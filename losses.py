from einops import rearrange
import torch
import torch.nn.functional as F

from architecture.ht_vicreg import off_diagonal

from math import isnan

# Flattens input tensor to be B x (T x D) (i.e., flattens all tokens into a single one)
def sequence_flatten(x):
    assert len(x.shape) > 1, 'sequence_flatten needs a tensor with at least 2 dimensions.'
    if len(x.shape) == 2:
        return x
    return rearrange(x, 'b ... -> b (...)')


# Flattens input tensor to be (B x T) x D (i.e., turns the input into a long batch of tokens)
def batch_flatten(x):
    assert len(x.shape) > 1, 'batch_flatten needs a tensor with at least 2 dimensions.'
    if len(x.shape) == 2:
        return x
    return rearrange(x, 'b ... d -> (b ...) d')


# Flattens input tensor to be B x (T x S) x D
def token_flatten(x):
    assert len(x.shape) > 2, 'token_flatten needs a tensor with at least 3 dimensions.'
    if len(x.shape) == 3:
        return x
    return rearrange(x, 'b ... d -> b (...) d')


# ***************************************************************************
# *************              LOSS FUNCTIONS               *******************
# ***************************************************************************
# kwargs is added so all loss functions have same interface
def cs_loss(tgt, pred, mult=1., **kwargs):
    return torch.tensor(mult, device=tgt.device) - mult*F.cosine_similarity(tgt, pred, dim=-1).mean()


def ce_loss(tgt, pred, tgt_temp=1, pred_temp=1, eps=0.000001, **kwargs):
    tgt = F.softmax(tgt / tgt_temp, dim=-1)
    pred = F.softmax(pred / pred_temp, dim=-1)
    return torch.mean(-torch.sum(tgt * torch.log(pred + eps), -1))


def mse_loss(tgt, pred, **kwargs):
    # MSE throws warning if it needs to broadcast, to supress it, do manual expansion of the target
    if pred.shape != tgt.shape:
        # When multiple tokens in one layer predict aggregation tokens in the next
        # Expects one dimension to be singleton (probably the token dimension), and the other two to have same size
        tgt = tgt.expand_as(pred)
    return F.mse_loss(pred, tgt)


# TODO the variance, could also be along T (token_flatten)
#       and then sum/mean (covariance still should be batch_flatten)
# TODO if this is applied, may make more sense to do so before/without BN
# Code slightly adapted from https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
def vicreg_loss(tgt, pred, eps=1e-4, tgt_already_reg=False, sim_coeff=25, std_coeff=25, cov_coeff=1, **kwargs):
    # TODO allow for the mse_loss to be other losses?
    repr_loss = mse_loss(tgt, pred)

    # Get tensor to shape (B x T) x D
    y = batch_flatten(pred)
    b, f2 = y.shape
    # TODO in the paper this is computed after the STD loss
    # Normalise
    y = y - y.mean(dim=0)
    # Compute variance
    std_y = torch.sqrt(y.var(dim=0) + eps)
    # Compute covariance
    cov_y = (y.T @ y) / (b - 1)

    if not tgt_already_reg:
        # In order to avoid regularizing target multiple times in a single pass, avoid computing this every time
        x = batch_flatten(tgt)
        b, f1 = x.shape
        # TODO in the paper this is computed after the STD loss
        x = x - x.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + eps)
        cov_x = (x.T @ x) / (b - 1)

        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(f1) + \
                   off_diagonal(cov_y).pow_(2).sum().div(f2)
    else:
        std_loss = torch.mean(F.relu(1 - std_y))
        cov_loss = off_diagonal(cov_y).pow_(2).sum().div(f2)

    loss = (
            sim_coeff * repr_loss
            + std_coeff * std_loss
            + cov_coeff * cov_loss
    )
    return loss


# Flattens each batch element and computes variance across them (returns 1 x D*T)
def sequence_variance(tgt):
    assert 1 < len(
        tgt.shape) < 5, 'Computing variance of tensor of 1 < len(shape) < 5 not implemented.'
    torch.var(sequence_flatten(tgt), dim=0)

# Compute variance of tokens across batch elements (returns 1 x D)
def batch_variance(tgt):
    assert 1 < len(
        tgt.shape) < 5, 'Computing variance of tensor of len(shape) < 1 or len(shape) > 5 not implemented.'
    return torch.mean(torch.var(batch_flatten(tgt), dim=0))

# Compute variance of tokens within one batch element, then average it across batches (returns 1 x D)
def token_variance(tgt):
    assert 1 < len(
        tgt.shape) < 5, 'Computing variance of tensor of 1 < len(shape) < 5 not implemented.'
    # If only 2 dims, impossible to get token variance, get batch variance instead
    if len(tgt.shape) == 2:
        return batch_variance(tgt)
    return torch.mean(torch.var(token_flatten(tgt), dim=1), dim=0)

# Expects targets and predictions dictionaries, computes specified loss
# model:        [nn.Module] the model, mostly to check if a specific module is causal or not
# target_agg:   [dict of nn.Tensor] dictionary of target tensors by module
# prediction:   [dict of nn.Tensor] dictionary of prediction tensors by module
# loss_fcn:     [python function] loss function to be used. Expects 3 parameters: target, prediction and temperature
# temp:         [float] Optional. temperature for the CrossEntropy loss function. Only used if loss is ce_loss
# target:       [dict of nn.Tensor] Optional. only used for causal layers, as some targets will be the full
#                     set of tokens (this variable), while others will be aggregated (target_agg).
# token_idxs:   [dict of ints or lists of ints] Optional. Indicates which positions the next module is trying ot predict
def compute_losses(model, target_agg, prediction, loss_fcn, side_prediction, loss_params, target=None, token_idxs=None):
    losses = dict()
    # TODO Not sure this initialization is really necessary,
    #       if they are needed and not passed, the function will fail either way
    # Really these are optional insofar not all model variants will need all of them, but the code is not prepared
    # to act accordingly if they are not passed
    if target is None:
        target = dict()
    if token_idxs is None:
        token_idxs = dict()

    # Keep track of which targets have already been regularised (for VICReg loss)
    already_regularised_tgt = dict()

    for g, group in enumerate(model.groups):
        losses[group] = dict()
        #####################################
        # All groups but last predict upwards
        if g < (len(model.groups) - 1):
            next_group = model.groups[g + 1]
            next_module = model.transformer_modules[next_group]

            # Prediction upward (if causal, prediction is to complete output - not aggregated)
            if next_module.is_causal:
                tgt = target[next_group]
            else:
                tgt = target_agg[next_group]
            pred = prediction[group + '_u']

            if group == 'backbone':
                # Roll the patches into a single row
                if next_module.is_causal:
                    # If 'space' module is not causal this is not necessary
                    tgt = token_flatten(tgt)
                pred = token_flatten(pred)

            # If the module that produced the target is causal
            if next_module.is_causal:
                # Roll targets forward to align with predictions
                # If 'space' module is causal, flatten_patches for targets too
                tgt = torch.roll(tgt, shifts=1, dims=1)[:, 1:, :]
                # First token does not get supervision
                pred = pred[:, 1:, ]
            else:
                if next_module.num_batched_groups > 1:
                    # Need to repeat_interleave aggregated target (dimensions won't match otherwise)
                    tgt = tgt.repeat_interleave(next_module.num_tokens, 1)
                else:
                    # Need to add singleton dimension to target (losses will broadcast)
                    tgt = tgt.unsqueeze(1)

            # Actually compute the loss
            losses[group].update({"up": loss_fcn(tgt, pred,
                                                 **loss_params,
                                                 tgt_already_reg=already_regularised_tgt.get(next_group, False))})
            assert not isnan(losses[group]['up'].item()), f"Woah, it seems you NaN-ed in {group}"
            if loss_fcn == vicreg_loss:
                already_regularised_tgt[next_group] = True

            if side_prediction:
                #####################################
                # All groups but last (also) predict sideways
                # Prediction sideways is always on aggregated targets
                tgt = target_agg[group]

                token_idx = token_idxs[group + '_s']
                # Get the temporal position to be predicted by each temporal position
                tgt = torch.stack([tgt[:, pos] for pos in token_idx], 1)

                if group == 'backbone':
                    # Although flattening should not be technically necessary,
                    #   for consistency with all other losses let's just work on B x T x D
                    tgt = token_flatten(tgt)
                    pred = token_flatten(prediction[group + '_s'])
                else:
                    pred = prediction[group + '_s']

                # Actually compute the loss
                losses[group].update({"side": loss_fcn(tgt, pred,
                                                       **loss_params,
                                                       tgt_already_reg=already_regularised_tgt.get(group, False))})
                assert not isnan(losses[group]['side'].item()), f"Woah, it seems you NaN-ed in {group}"
                if loss_fcn == vicreg_loss:
                    already_regularised_tgt[group] = True


        ########################################
        # All groups but first predict downwards
        if g > 0:
            prev_group = model.groups[g - 1]
            # Prediction downward (always predict aggregated output of previous group)
            tgt = target_agg[prev_group]
            pred = prediction[group + '_d']

            if group == 'space':
                # Roll the patches into a single row
                tgt = token_flatten(tgt)
                if model.transformer_modules['space'].is_causal:
                    # If 'space' module is not causal this is not necessary
                    pred = token_flatten(pred)

            if model.transformer_modules[group].is_causal:
                # Roll targets backward to align with predictions
                tgt = torch.roll(tgt, shifts=-1, dims=1)[:, :-1, :]
                # Last token does not get supervision
                pred = pred[:, :-1, ]
            else:
                # Need to sample the tokens from the target selected with token_idx above
                token_idx = token_idxs[group + '_d']
                if isinstance(token_idx, list):
                    tgt = torch.stack([tgt[:, pos] for pos in token_idx], 1)
                else:
                    tgt = tgt[:, token_idx]

            # Actually compute the loss
            losses[group].update({"down": loss_fcn(tgt, pred,
                                                   **loss_params,
                                                   tgt_already_reg=already_regularised_tgt.get(prev_group, False))})
            assert not isnan(losses[group]['down'].item()), f"Woah, it seems you NaN-ed in {group}"
            if loss_fcn == vicreg_loss:
                already_regularised_tgt[prev_group] = True

    return losses