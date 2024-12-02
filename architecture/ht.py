from torch import nn, tensor
from torchvision import transforms as T
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from random import random
from enum import Enum

from architecture.stam import GeneralTransformer
from architecture.s3d import S3DStem

class Stage(Enum):
    TRAIN = 0
    PROBE = 1
    FTUNE = 2
    EVAL = 3
    FEATX = 4


class Task(Enum):
    MINE = 0
    CLS = 1
    VICREG = 2
    SVT = 3

    def str2task(text):
        return eval("Task." + text.upper())


# Standard ModuleDict but adds "get" functionality
class CustomModuleDict(nn.ModuleDict):
    def __init__(self, modules=None):
        super().__init__(modules)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random() > self.p:
            return x
        return self.fn(x)


def_aug = {"augs": {"color_jitter": {"p": 0.2,
                                     "brightness": 0.4,
                                     "contrast": 0.4,
                                     "saturation": 0.4,
                                     "hue": 0.1},
                    "grayscale": {"p": 0.15},
                    "horizontal_flip": {"p": 0.5},
                    "gaussian_blur": {"p": 0.30,
                                      "kernel": (3, 3),
                                      "sigma": (0.3, 2.0)},
                    "random_resized_crop": {"p": 1.0,
                                            "scale": (0.75, 0.95)}
                    },
           "normalization": 'imagenet'}


norms = {'imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
         'ucf101': {'mean': [0.37627297, 0.37627799, 0.37627846], 'std': [0.24465545, 0.24465971, 0.24466582]}}

class DataAugmentation(nn.Module):
    def __init__(self, h, w, augmentation=None):
        super().__init__()
        if augmentation is None:
            augmentation = def_aug

        augs = augmentation['augs']
        norm = norms[augmentation['normalization']]

        self.augment = nn.Sequential(
            RandomApply(
                T.ColorJitter(augs['color_jitter']['brightness'],
                              augs['color_jitter']['contrast'],
                              augs['color_jitter']['saturation'],
                              augs['color_jitter']['hue']),
                p=augs['color_jitter']['p']
            ),
            T.RandomGrayscale(p=augs['grayscale']['p']),
            T.RandomHorizontalFlip(p=augs['horizontal_flip']['p']),
            RandomApply(
                T.GaussianBlur(augs['gaussian_blur']['kernel'],
                               augs['gaussian_blur']['sigma']),
                p=augs['gaussian_blur']['p']
            ),
            RandomApply(
                T.RandomResizedCrop((h, w),
                                    scale=augs['random_resized_crop']['scale']),
                p=augs['random_resized_crop']['p']
            ),
            T.Normalize(    # TODO recompute these values for other datasets
                mean=tensor(norm['mean']),
                std=tensor(norm['std'])),
        )
        # self.rearrange1 = Rearrange('b f ... -> (b f) ...')
        # self.rearrange2 = RearrangeCustom('(b f) ... -> b f ...')

    # Expects video [B x F x H x W x C]
    def forward(self, x):
        x = x.permute(0,1,4,2,3)
        for i in range(x.shape[0]):
            x[i] = self.augment(x[i])
        return x.permute(0,1,3,4,2)


# Perform batch normalization. It is implemented for non-CNN layers where channel dimension tends to be last.
# dim             indicates number of channels
# D               indicates number of dimensions in expected tensor input
# channel_last    indicates if the channels are the last dimension (otherwise assumed to be in second dimension)
class BatchNormalization(nn.Module):
    def __init__(self, dim, D=2, channel_last=True):
        super(BatchNormalization, self).__init__()
        if D == 2 or D == 3:
            self.l = nn.BatchNorm1d(dim)   # Expects B x C  or  B x C x F
        elif D == 4:
            self.l = nn.BatchNorm2d(dim)   # Expects B x C x H x W
        elif D == 5:
            self.l = nn.BatchNorm3d(dim)   # Expects B x C x F x H x W
        else:
            raise(Exception, "No BatchNorm for so many dimensions! Provided " + str(D) + " but max is 5")

        self.channel_last = channel_last
        # Channel will either be in second position or last position
        if self.channel_last:
            # Permute so channels are in second position
            self.p1 = tuple([0, D - 1] + list(range(1, D - 1)))
            # Permute so channels are back in last position
            self.p2 = tuple([0] + list(range(2, D)) + [1])

    def forward(self, x):
        if self.channel_last:
            return self.l(x.permute(*self.p1)).permute(*self.p2)
        else:
            return self.l(x)


# TODO allow for predictions to be made on a smaller feature space by specifying a smaller d_out
#  (must also add a linear layer for the targets)
class Predictor(nn.Module):
    # d_mid is a list of hidden layer sizes (mutually exclusive with mult)
    def __init__(self, d_in, d_out, D=2, d_mid=None, mult=2, channel_last=True, out_bn=False):
        super(Predictor, self).__init__()
        if d_mid is not None:
            layers = list()
            for dim in d_mid:
                layers.append(nn.Linear(d_in, dim))
                layers.append(BatchNormalization(dim, D=D, channel_last=channel_last))
                layers.append(nn.GELU())
                d_in = dim
            layers.append(nn.Linear(d_in, d_out, bias=False))
            self.layers = nn.Sequential(*layers)
        else:

            self.layers = nn.Sequential(
                nn.Linear(d_in, d_in * mult),
                BatchNormalization(d_in * mult, D=D, channel_last=channel_last),
                nn.GELU(),  # We use GELU for consistency with activation used in last transformer layers
                nn.Linear(d_in * mult, d_out, bias=False)
            )

        if out_bn:
            self.layers.add_module('out_bn', BatchNormalization(d_out, D=D, channel_last=channel_last))

    def forward(self, x):
        return self.layers(x)


class HierarchicalTransformer(nn.Module):
    def __init__(self, config):
        super(HierarchicalTransformer, self).__init__()

        self.task = Task.str2task(config['train']['task'])
        self.output_bn = config['arch']['output_batchnorm']

        # ***************************************************************************
        # *************        GROUP HANDLING VARIABLES Pt.1      *******************
        # ***************************************************************************

        self.groups = eval(config['arch']['groups'])
        self.groups_causal = list()
        self.groups_non_causal = list()

        # ***************************************************************************
        # *************              SET UP MODEL LAYERS          *******************
        # ***************************************************************************

        # DICTS TO STORE DIFFERENT LAYERS FOR THE DIFFERENT GROUPS
        self.transformer_modules = CustomModuleDict()  # Stores transformer modules
        self.predictors = CustomModuleDict()           # Predictor networks. MLPs to be used at output of group or net
        self.output_norms = CustomModuleDict()         # Norm/regularizer before next group. Applied on aggregated outputs.
        self.linear_proj = CustomModuleDict()          # Used to change dimensionality of tokens between transformer modules

        ############
        # BACKBONE #
        ############
        bkbn_arch = config['arch']['architecture_backbone']
        self.use_backbone = bkbn_arch['use_cnn_stem']
        self.return_patches = 'backbone' in self.groups  # Indicates if backbone is a separate group from space
        self.space_pos = self.groups.index('space')   # This is used so backbone can be included/omitted in space group
        # Assume three input channels (RGB)
        channels = 3
        if self.use_backbone:
            channels = bkbn_arch['out_channels']
            # [B, out_c, T, H/4, W/4]
            self.backbone = S3DStem(bkbn_arch['in_channels'], channels, dropout=bkbn_arch['dropout'])


        ############
        # PATCHIFY #
        ############
        # Check that images will be divisible by patch size!
        space_arch = config['arch']['architecture_space']
        image_size = eval(space_arch['input_size'])
        patch_size = space_arch['size_ratio']
        dim = bkbn_arch['dim']
        dim_space = space_arch['dim']

        if isinstance(image_size, tuple):
            correct_dimensions = (image_size[0] % patch_size == 0) and (image_size[1] % patch_size == 0)
        else:
            correct_dimensions = image_size % patch_size == 0
        assert correct_dimensions, 'Image dimensions must be divisible by the patch size.'

        # PATCHIFY AND LINEAR PROJECTION TO SPATIAL TRANSFORMER
        patch_dim = channels * (patch_size ** 2)

        # Now (h w) indexes all patches
        # Each patch is represented by a set of p1 x p2 x c features
        self.to_patch_embedding = nn.Sequential(
                                Rearrange('b c f (h p1) (w p2) -> b f (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                                nn.Linear(patch_dim, dim))
        if dim != dim_space:
            self.linear_proj['backbone'] = nn.Linear(dim, dim_space)
        if self.use_backbone and (not self.return_patches) and self.output_bn:
            # output_norm for backbone will be added in main loop only if backbone is a group itself
            # otherwise add it here
            self.output_norms['backbone'] = BatchNormalization(dim, D=bkbn_arch['nD'])

        ################
        # TRANSFORMERS #
        ################
        if self.groups[-1] != 'space':
            assert config['arch']['architecture_' + self.groups[-1]]['size_ratio'] == 1, \
                   "Last group/module (" + self.groups[-1] + ") cannot have batched processing." + \
                   "Either set size_ratio:1 or implement average pooling for the multiple aggregation output tokens."
        # For each group
        for g, group in enumerate(self.groups):
            architecture = config['arch']['architecture_'+group]
            # Check that number of frames are divisible by clips in all temporal modules
            if group != 'space' and group != 'backbone':
                assert architecture['input_size'] % architecture['size_ratio'] == 0, \
                    "Number of frames must be divisible by number of clips"
            dim = architecture['dim']
            # CREATE A TRANSFORMER FOR EACH GROUP
            if group != 'backbone':
                i_size = eval(architecture['input_size']) if isinstance(architecture['input_size'], str) \
                                                          else architecture['input_size']
                if group == 'space':
                    # Number of frames that temporal transformer expects
                    # (which is equivalent to number of frames the spatial transformer will output)
                    # Unless spatial transformer is the last one, in which case num_out_t can only be 1
                    if self.groups[-1] == 'space':
                        n_out_t = 1
                    else:
                        n_out_t = config['arch']['architecture_' + self.groups[g + 1]]['input_size']
                else:
                    n_out_t = architecture['size_ratio']

                # Select a different aggregation token for causal layers in supervised training mode
                agg_token = 0
                if architecture['causal_mask']:
                    # TODO revisit this. depending on the number of losses, -1 might always get (some) supervision
                    if self.task == Task.MINE:
                        agg_token = -2
                    else:
                        agg_token = -1

                self.transformer_modules[group] = GeneralTransformer(
                                                      dim=dim,
                                                      type_space=(group == 'space'),
                                                      input_size=i_size,
                                                      size_ratio=architecture['size_ratio'],
                                                      layers=architecture['layers'],
                                                      heads=architecture['heads'],
                                                      mlp_dim=architecture['mlp_dim'],
                                                      head_dim=architecture['head_dim'],
                                                      dropout=architecture['dropout'],
                                                      causal_mask=architecture['causal_mask'],
                                                      num_out_tokens=n_out_t,
                                                      agg_token=agg_token)
                if g < (len(self.groups) - 1):
                    next_dim = config['arch']['architecture_' + self.groups[g + 1]]['dim']
                    # If token dimensionality of next module is different, add linear projection
                    if next_dim != dim:
                        self.linear_proj[group] = nn.Linear(dim, next_dim)

                # Increment counter of causal modules
                if architecture['causal_mask']:
                    self.groups_causal.append(group)
                else:
                    self.groups_non_causal.append(group)

            #########################################
            # REGULARIZERS AFTER EACH MODULE OUTPUT #
            #########################################
            if self.output_bn:
                self.output_norms[group] = BatchNormalization(dim, D=architecture['nD'])

        ###################
        # FINAL MLP LAYER #
        ###################

        # Indicates stage (train, probe, ftune or eval)
        self.stage = Stage.TRAIN

        # If causal, use last supervised token, otherwise use CLS
        # (from last group --> "architecture" from last loop iter)
        self.global_idx = 0           # Use CLS token?
        if architecture['causal_mask']:
            self.global_idx = -1  # Can use last token in causal setting if supervised

        # Probing/Tunning relevant variables
        self.num_classes = config['data'][config['train']['dataset']]['num_classes']
        self.final_output_dim = config['arch']['architecture_' + self.groups[-1]]['dim']

        # Create final classification layer
        self.mlp_head = nn.Linear(self.final_output_dim, self.num_classes)

        self.probe_augment = config['data']['data_augmentation']['use_datAug_probing']

    # To be implemented by child classes
    def forward(self,x):
        # Batch size
        b = x.shape[0]

        # If Probing for a second or more epochs, no need to run through whole network
        #  UNLESS data augmentation is used during probing!!
        if self.stage != Stage.PROBE or self.probe_augment:
            #############
            # EMBEDDING #
            #############

            # For simplicity, both the backbone and the patchify function expect B x C x F x H x W
            x = x.permute(0, 4, 1, 2, 3)
            if self.use_backbone:
                x = self.backbone(x)

            # Patchify / normalize
            x = self.to_patch_embedding(x)
            if self.output_bn:
                x = self.output_norms['backbone'](x)

            # Project to spatial dim
            if self.linear_proj.get('backbone', False):
                x = self.linear_proj['backbone'](x)

            #####################################
            # MAIN LOOP THROUGH ALL NET MODULES #
            #####################################

            for g, group in enumerate(self.groups[self.space_pos:]):
                current_transf = self.transformer_modules[group]
                # RUN THROUGH TRANSFORMER
                x = current_transf(x)

                # SELECT AGGREGATION TOKEN
                agg = current_transf.format_output_agg(x[:, current_transf.agg_token], b=b)

                # NORMALISE OUTPUT AND TARGETS
                if self.output_bn:
                    agg = self.output_norms[group](agg)

                # Reasign aggregated
                x = agg

                # Project to next module's dimensionality (if necessary)
                if self.linear_proj.get(group, False):
                    x = self.linear_proj[group](x)

        # MLP HEAD
        if self.task == Task.CLS or (self.stage != Stage.TRAIN and self.stage != Stage.FEATX):
            # Project last aggregation token
            out = self.mlp_head(x)
            # Softmax
            out = F.log_softmax(out, dim=-1)

        if self.task == Task.CLS or self.stage == Stage.PROBE or self.stage == Stage.FTUNE \
                or (self.stage == Stage.EVAL and not self.output_features):
            return out
        elif self.stage == Stage.EVAL and self.output_features:
            return out, x
        else:  # self.stage == Stage.TRAIN or self.stage == Stage.FEATX:
            return x

    def set_stage(self,stage,output_features=False):
        # self.output_features is only used during eval, so SSL losses can be computed on eval set
        self.output_features = False
        if stage == Stage.TRAIN or stage == Stage.FTUNE or stage == Stage.PROBE:
            self.train()
        elif stage == Stage.EVAL or stage == Stage.FEATX:
            self.eval()
            if output_features:
                self.output_features = True

        if stage == Stage.PROBE:
            # Freeze all network
            self.requires_grad_(False)
            # Put it in eval mode
            for module in self.children():
                module.eval()

        if stage == Stage.PROBE or stage == Stage.FTUNE:
            # Make sure the head is trained
            self.mlp_head.requires_grad_(True)
            self.mlp_head.train(True)

        self.stage = stage

    def probe(self):
        # Reset layer (reset_parameters only works on linear layers)
        #self.mlp_head = nn.Linear(self.final_output_dim, self.num_classe)
        self.mlp_head.reset_parameters()

    def get_groups(self):
        return self.groups

    def get_mlp_head(self):
        return self.mlp_head