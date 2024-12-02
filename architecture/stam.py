# Code based on https://github.com/lucidrains/STAM-pytorch

import torch
from torch import nn, einsum
from einops import rearrange, repeat

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# TODO implement fast attention?? https://github.com/HazyResearch/flash-attention
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., masking=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.masking = masking

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        # Use a single matrix to map the input tensor to Q, K and V
        # Chunk will split into 3 separate tensors
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # Go through all chunks (map) and rearrange their dimensions
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Masking
        # TODO: is there an efficient way of doing this that does not involve computing the whole matrix??
        if self.masking:
            # No need to account for CLS as temporal causal transformer does not have one! (add +1 to n)
            mask = (torch.full((b, h, n, n), -float('inf'), device=dots.device)).triu(diagonal=1)
            # If CLS token is allowed to attend to all tokens:
            # mask[:, :, 0, :] = 0
            # mask[:, :, 1:, 0] = -float('inf')
            dots += mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., causal_masking=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                       masking=causal_masking)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class RearrangeCustom(nn.Module):
    def __init__(self, r_str, **kwargs):
        super().__init__()
        self.r_str = r_str
        self.args = kwargs

    def forward(self, x, b):
        return rearrange(x, self.r_str, b=b, **self.args)


class IdentityCustom(nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityCustom, self).__init__()

    def forward(self, x, b):
        return x


class GeneralTransformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            type_space,         # True to indicate space, False to indicate time
            input_size,         # If spatial, this is image_size; If temporal, this is number of frames.
            size_ratio,         # If spatial, this is patch_size; If temporal, this is number of clips.
            layers,
            heads,
            head_dim,
            mlp_dim,
            dropout=0.,
            causal_mask=False,
            num_out_tokens=0,   # Only used for spatial transformer, to communicate number of output frames
            agg_token=-2         # See below, aggregation token is generally second to last, but in some cases it's last
    ):
        super().__init__()

        if type_space:
            # Number of tokens is number of patches per frame
            if type(input_size) is tuple:
                self.num_tokens = (input_size[0] // size_ratio) * (input_size[1] // size_ratio)
            else:
                self.num_tokens = (input_size // size_ratio) ** 2
            self.num_batched_groups = num_out_tokens
            self.format_input = RearrangeCustom('b f ... -> (b f) ...')
            if self.num_batched_groups == 1:
                self.format_output = IdentityCustom()
            else:
                self.format_output = RearrangeCustom('(b f) ... -> b f ...', f=self.num_batched_groups)
            self.format_output_agg = self.format_output
        else:
            # Number of tokens is frames per clip
            self.num_tokens = input_size // size_ratio
            self.num_batched_groups = size_ratio
            # Collapse clip and batch dimensions
            self.format_input = RearrangeCustom('b (c f) ... -> (b c) f ...', c=self.num_batched_groups)
            self.format_output = RearrangeCustom('(b c) f ... -> b (f c) ...', c=self.num_batched_groups)
            if self.num_batched_groups == 1:
                self.format_output_agg = self.format_output
            else:
                self.format_output_agg = RearrangeCustom('(b c) ... -> b c ...', c=self.num_batched_groups)


        self.is_causal = causal_mask

        # Initialize learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens + (0 if causal_mask else 1), dim))

        # Initialize CLS token
        # It will be -1 if causal and:
        #   - supervised training (this may change, if it's supervised but not from scratch, but fine-tunning)
        #   - ssl but the layer predicts upwards/side (still, -2 is getting more supervision)
        self.agg_token = agg_token  # If causal, we use either -2 or -1 as global aggregation token
        if not self.is_causal:
            # Define the aggregation token to be CLS
            self.agg_token = 0
            # TODO is this the best way to initialise?? is it same range as remaining net??
            #   It seems Linear and Conv use kaiming_uniform_ which is between -smth and +smth
            #   SVT uses truncated_normal_ for everything
            self.cls_token = nn.Parameter(torch.randn(1, dim))

        # Indicates whether tokens will be grouped into smaller sets for separate processing
        # (e.g., patches are separated by frame, so spatial transformer processes each one frame separately)
        self.has_batched_processing = size_ratio > 1

        self.transformer = Transformer(dim, layers, heads, head_dim, mlp_dim, dropout, causal_mask)

    def forward(self, x):
        b = x.shape[0]  # BatchSize

        # Rearrange input such that (Batch x Tokens x Dim)
        x = self.format_input(x, b)

        # Concat CLS token
        if not self.is_causal:
            # Repeat
            cls_tokens = repeat(self.cls_token, 'n d -> b n d', b=x.shape[0])
            # Concat
            x = torch.cat((cls_tokens, x), dim=-2)

        # Add positional embeddings (singleton dimensions gets broadcasted)
        x += self.pos_embedding[:]

        # Run through transformer layers
        x = self.transformer(x)

        # Return all tokens (format_output will be called from outside!)
        return x
