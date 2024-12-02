# Augmentation code based on https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
import torch
from random import randint
from numpy.random import normal as npNormal
from architecture.ht import *


# Inefficient (but not so much so cannot be used for our purposes) way of truncating a normal
def truncated_normal_sampling(mean, std, mini, maxi):
    sample = round(npNormal(mean, std))
    while (not (mini <= sample < maxi)) or sample == mean:
        sample = round(npNormal(mean, std))
    return sample


def get_position_encoding(seq_len, dim, relative=False):
    if relative:
        return torch.nn.Parameter(torch.rand((1, seq_len*2-1, dim)))
    else:
        return torch.nn.Parameter(torch.rand((1, seq_len, dim)))

def get_relative_position(start, end, seq_len):
    # relative distance is the postion to be predicted - original position (index)
    # actual position in PE is seq_len + that relative position
    relative_p = seq_len + (end - start)
    # as position seq_length does not exist (that where one token is predicting itself)
    # subtract one to it if it is seq_length or larger
    if relative_p >= seq_len:
        relative_p -= 1

    return relative_p


class HT_Mine(HierarchicalTransformer):
    def __init__(self, config):
        super(HT_Mine, self).__init__(config)

        self.predicts_side = config['train']['use_temporal_prediction'] and \
                             config['data'][config['train']['dataset']]['sampler']['length'] > 1

        # Used to tell which position a token is tasked with predicting
        self.prediction_pos = dict()

        self.std_div = config['train']['std_divisor']

        self.always_compute_loss = config['train']['always_compute_loss']

        # ***************************************************************************
        # *************        GROUP HANDLING VARIABLES Pt.1      *******************
        # ***************************************************************************

        self.trainable_groups = dict()                    # each trainable group includes its projectors

        self.alternating_interval = config['train']['alternating_interval']

        # default requires_grad value True
        self.set_trainable_groups(True)

        # ***************************************************************************
        # *************              SET UP MODEL LAYERS          *******************
        # ***************************************************************************

        # DICTS TO STORE DIFFERENT LAYERS FOR THE DIFFERENT GROUPS
        self.target_norms = CustomModuleDict()         # Only used if module is causal, as targets won't be aggregated
        self.prediction_PEs = nn.ParameterDict()       # Used to store fixed PEs that will condition prediction

        ################
        #  PREDICTORS  #
        ################
        # For each group
        for g, group in enumerate(self.groups):
            architecture = config['arch']['architecture_'+group]
            dim = architecture['dim']
            pred_h_dim = architecture['pred_h_dim']
            if pred_h_dim is not None:
                if isinstance(pred_h_dim, int):
                    pred_h_dim = [pred_h_dim]
                elif isinstance(pred_h_dim, str):
                    pred_h_dim = eval(pred_h_dim)
            else:
                pred_h_dim = []
            if self.task == Task.MINE:
                ###########################
                # CREATE PROJECTOR LAYERS #
                ###########################
                # Every group but the last predicts upwards (and sideways)
                if g != (len(self.groups) - 1):
                    next_dim = config['arch']['architecture_' + self.groups[g + 1]]['dim']
                    self.predictors[group + '_u'] = Predictor(dim,
                                                              next_dim,
                                                              D=architecture['nD'],
                                                              d_mid=pred_h_dim,
                                                              out_bn=self.output_bn)
                    if self.predicts_side:
                        self.predictors[group + '_s'] = Predictor(dim,
                                                                  dim,
                                                                  D=architecture['nD'],
                                                                  d_mid=pred_h_dim,
                                                                  out_bn=self.output_bn)
                        # Clip will never predict sideways
                        if group == 'time':
                            seq_length = config['arch']['architecture_clip']['input_size']
                        else:
                            # Both space and backbone need number of frames input to time transformer
                            seq_length = config['arch']['architecture_time']['input_size']
                        # (1 x seq_len x dim)
                        self.prediction_PEs[group + '_s'] = get_position_encoding(seq_length, dim, relative=True)

                # Every group but the first predicts downward
                if g != 0:
                    prev_dim = config['arch']['architecture_' + self.groups[g - 1]]['dim']
                    # If it's causal, number of dimensions will match that of aggregated
                    #    output from previous group
                    if architecture.get('causal_mask', False):
                        aux_nD = config['arch']['architecture_' + self.groups[g - 1]]['nD']
                    else:
                        aux_nD = architecture['nD']
                    self.predictors[group + '_d'] = Predictor(dim,
                                                              prev_dim,
                                                              D=aux_nD,
                                                              d_mid=pred_h_dim,
                                                              out_bn=self.output_bn)
                    # If the module has batched processing (all but last), is not causal and sideways prediction is off
                    if (not self.predicts_side) and \
                            (not architecture.get('causal_mask', False)) \
                            and g != (len(self.groups) - 1):
                        # then we need 2D PEs for the prediction
                        dim_t = (dim // 3) * 2  # TODO unharcode % of PEs used to identify the token and the group
                        dim_g = dim - dim_t
                        seq_length_t = self.transformer_modules[group].num_tokens
                        seq_length_g = self.transformer_modules[group].num_batched_groups
                        self.prediction_PEs[group + '_d_t'] = get_position_encoding(seq_length_t, dim_t)
                        self.prediction_PEs[group + '_d_g'] = get_position_encoding(seq_length_g, dim_g)
                    # Causal mode does not require this PE for down prediction (it predicts a fixed position)
                    elif not architecture.get('causal_mask', False):
                        seq_length = self.transformer_modules[group].num_tokens
                        self.prediction_PEs[group + '_d'] = get_position_encoding(seq_length, dim)

            #########################################
            # REGULARIZERS AFTER EACH MODULE OUTPUT #
            #########################################
            if self.output_bn:
                if architecture.get('causal_mask', False):
                    # Only need this in causal mode, otherwise output_norm plays double role
                    # Number of dimensions will match that of aggregated output from previous module
                    if group == 'space' and g == 0:
                        # If backbone is trained with space it won't be in self.groups
                        mod = 'backbone'
                    else:
                        mod = self.groups[g - 1]
                    prev_nD = config['arch']['architecture_' + mod]['nD']
                    self.target_norms[group] = BatchNormalization(dim, D=prev_nD)

        ###################
        ####   OTHERS  ####
        ###################

        # If causal, use last supervised token, otherwise use CLS
        # (from last group --> "architecture" from last loop iter)
        if architecture['causal_mask'] and self.task == Task.MINE:
            self.global_idx = -2  # Use last self-supervised token

        if self.task == Task.MINE:
            # Get parameters if explicit freezing is on
            # (if alternating_interval is 0, all modules are trained on every iteration)
            self.params = dict()
            self.explicit_freeze = config['train']['explicit_freeze'] and config['train']['alternating_interval'] > 0
            if self.explicit_freeze:
                for group in self.groups:
                    mod, pred = self.get_params_for_group(group, named=False)
                    self.params[group] = mod + pred

    # If probing, (first epoch) also return last layer features
    # If probing, (second epoch and next ones) only pass through MLP head
    # If eval after probing or training, store targets/predictions for ssl loss computation
    # If training, do not use MLP unless 'cls' task
    # If 'cls' task, do not store anything
    def forward(self, x):
        # Batch size
        b = x.shape[0]
        # Should intermediate module outputs, as well as predictions, be kept for loss computation?
        keep_and_predict = self.task == Task.MINE and \
                           (self.stage == Stage.TRAIN or (self.stage == Stage.EVAL and self.output_features))
        # Do we need to train each module separately?
        do_detach = self.task == Task.MINE and self.stage == Stage.TRAIN

        # INITIALISE TARGET / PREDICTION DICTIONARIES
        if keep_and_predict:
            # Used to store aggregated targets (general case)
            output_targets_agg = dict()
            # Used to store full-length targets (predicting upwards to a causal layer)
            output_targets = dict()
            # Used to store each module's predictions (module_u or module_p)
            output_predictions = dict()

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

            if keep_and_predict and self.return_patches:
                if self.trainable_groups['backbone'] or self.always_compute_loss:
                    # Predict frames from patches and store
                    output_predictions['backbone_u'] = self.predictors['backbone_u'](x)
                    # Predict & save sideways if applies
                    if self.predicts_side:
                        output_predictions['backbone_s'] = self.predictors['backbone_s'](
                                                            x + self.get_pos_enc('backbone_s')[:].unsqueeze(2))

                # Also, this means backbone is trained separately from spatial transformer
                if do_detach:
                    x = x.detach()

                # Patches are only targets if backbone is trained separately from spatial transformer
                output_targets_agg['backbone'] = x

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
                # TODO This won't work if last group has batched processing
                if current_transf.is_causal and current_transf.has_batched_processing:
                    # All tokens get supervision except last one,
                    # so take token -1 for each batched set of tokens except last one
                    agg2 = current_transf.format_output_agg(x[:, -1], b=b)
                    agg = torch.cat([agg2[:, :-1], agg[:, -1:]], dim=1)

                # NORMALISE OUTPUT AND TARGETS
                if self.output_bn:
                    # Apparently BatchNorm returns a new copy of the tensor, so this should not be a problem
                    if current_transf.is_causal:
                        x = self.target_norms[group](current_transf.format_output(x, b=b))
                    agg = self.output_norms[group](agg)

                # Store module's output to be used as targets
                if keep_and_predict:
                    output_targets_agg[group] = agg
                    if current_transf.is_causal:
                        output_targets[group] = x
                    if self.trainable_groups[group] or self.always_compute_loss:
                        # PREDICTOR NETWORKS
                        # Run through predictors
                        if self.predictors.get(group + '_u', False):
                            output_predictions[group + '_u'] = self.predictors[group + '_u'](agg)

                        if self.predictors.get(group + '_s', False):
                            output_predictions[group + '_s'] = self.predictors[group + '_s'](
                                                                agg + self.get_pos_enc(group + '_s')[:])

                        # When predicting down, a random token is selected as target,
                        #   add positional encoding of that token
                        if self.predictors.get(group+'_d', False):
                            if current_transf.is_causal:
                                output_predictions[group + '_d'] = self.predictors[group + '_d'](x)
                            else:
                                # Expects that train function will have set self.prediction_pos before
                                # running the forward function.
                                output_predictions[group+'_d'] = self.predictors[group+'_d'](
                                                        agg + self.get_pos_enc(group + '_d')[:])

                # Detach tensor, so each module gets independent supervision
                if do_detach:
                    x = agg.detach()
                else:
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

        if self.task == Task.CLS or self.stage == Stage.PROBE or self.stage == Stage.FTUNE\
                or (self.stage == Stage.EVAL and not self.output_features):
            return out
        elif self.stage == Stage.EVAL and self.output_features:
            return out, output_targets, output_targets_agg, output_predictions
        elif self.stage == Stage.FEATX:
            return x
        else: # if self.stage == Stage.TRAIN:
            return output_targets, output_targets_agg, output_predictions

    # If loading model for curriculum learning PE length may have changed
    #   2 main cases:
    #    - APE of transformers or conditioning adds length, copy the first prev_seq_length
    # TODO NOT SURE THE RPE ONE IS HANDLED PROPERLY!
    #    - RPE for conditioning side prediction adds length, copy the middle ones!
    #   First copy weights, then remove them from old_arch so load doesn't fail
    def load_model_weights(self, old_arch, strict=False):
        new_arch = self.state_dict()
        # CHECK TRANSFORMER PEs
        for group in self.groups:
            if group != 'backbone':
                pe_tensor = 'transformer_modules.' + group + '.pos_embedding'
                if pe_tensor in old_arch:
                    if new_arch[pe_tensor].shape != old_arch[pe_tensor].shape:
                        old_len = old_arch[pe_tensor].shape[1]
                        new_arch[pe_tensor][:, :old_len] = old_arch[pe_tensor]
                        del old_arch[pe_tensor]

        # CHECK PREDICTOR CONDITIONING PEs
        for name in self.prediction_PEs:
            pe_tensor = 'prediction_PEs.' + name
            # General case: PEs keep number of dimensions but may increase length
            if pe_tensor in old_arch:
                if new_arch[pe_tensor].shape != old_arch[pe_tensor].shape:
                    old_len = old_arch[pe_tensor].shape[1]
                    new_arch[pe_tensor][:, :old_len] = old_arch[pe_tensor]
                    del old_arch[pe_tensor]
            # If not sideways: down PEs may have gone from 1D (spatial only) to 2D.
            #   As dims may have changed, copy token PE up to 2D dim
            elif not self.predicts_side and name.endswith('_t'):
                pe_tensor = 'prediction_PEs.' + name.split('_')[0] + '_d'
                # If down PE existed as 1D
                if pe_tensor in old_arch:
                    old_len = old_arch[pe_tensor].shape[1]
                    new_dim = new_arch[pe_tensor + '_t'].shape[-1]
                    new_arch[pe_tensor + '_t'][:, :old_len] = old_arch[pe_tensor][:, :, :new_dim]
                    del old_arch[pe_tensor]

        # Filter out any other unnecessary keys
        old_arch = {k: v for k, v in old_arch.items() if k in new_arch}
        new_arch.update(old_arch)
        # TODO load mlp_head if it is loading only for eval?
        print("WARNING: loading the mlp_head is disabled!! if loading for evaluation this behaviour should be changed.")
        del new_arch['mlp_head.weight']
        del new_arch['mlp_head.bias']

        # Once PEs have been cleaned and handled, just load everything else
        self.load_state_dict(new_arch, strict=strict)

    def set_stage(self, stage, output_features=False):
        super(HT_Mine, self).set_stage(stage, output_features)

        if self.task == Task.MINE and stage == Stage.TRAIN and self.alternating_interval > 0:
            self.set_trainable_groups(True)

            if self.explicit_freeze:
                self.set_requires_grad_from_trainable_groups()

    def set_trainable_groups(self, d):
        for g in self.groups:
            self.trainable_groups[g] = d            # need this cos linear probing will set all to False
            # TODO allow for a schedule of intervals, slightly increasing/decreasing it over time
            # If training is done by alternating, each group should have opposite requires_grad than previous group
            if self.task == Task.MINE and self.alternating_interval > 0:
                d = not d

    def switch_training_groups(self):
        if self.task != Task.MINE:
            raise(Exception, "switching_training_groups only works if selected task is 'mine'")

        for group, on in self.trainable_groups.items():
            self.trainable_groups[group] = not on

        if self.explicit_freeze:
            self.set_requires_grad_from_trainable_groups()

    def set_requires_grad_from_trainable_groups(self):
        for group in self.groups:
            to = self.trainable_groups[group]
            for param in self.params[group]:
                param.requires_grad_(to)

    def get_trainable_groups(self):
        return [k for k, v in self.trainable_groups.items() if v]

    def gather_2d_pos_enc(self, p, group):
        num_tokens = self.prediction_PEs[group+'_t'].shape[1]
        token_pos = (p % num_tokens)
        group_pos = p // num_tokens
        return torch.cat(
            [self.prediction_PEs[group+'_t'][:, token_pos],
             self.prediction_PEs[group+'_g'][:, group_pos]],
            dim=-1)

    def get_pos_enc(self, group):
        pos = self.prediction_pos[group]
        if isinstance(pos, list):
            is_relative = group.split('_')[1] == 's'
            positions = list()
            # If 2D positional encoding
            if (group + '_t') in self.prediction_PEs:
                for i, p in enumerate(pos):
                    if is_relative:
                        p = get_relative_position(i, p, len(pos))
                    positions.append(self.gather_2d_pos_enc(p, group).unsqueeze(1))
            else:
                for i, p in enumerate(pos):
                    if is_relative:
                        p = get_relative_position(i, p, len(pos))
                    positions.append(self.prediction_PEs[group][:, p].unsqueeze(1))
            return torch.cat(positions, dim=1)
        else:
            # This is for sure predicting down from the last group (no sideways, hence no relative possible)
            if (group + '_t') in self.prediction_PEs:
                return self.gather_2d_pos_enc(pos, group)
            else:
                # Pos embedding has singleton batch dimension
                return self.prediction_PEs[group][:, pos]

    #   In non-causal mode, module L predicts random token from module L-1
    #   If a module is causal each token predicts next token, no need for random one
    def set_random_prediction_pos(self):
        token_idxs = dict()
        # Include backbone if we are using sideways prediction
        groups = (['backbone'] if self.return_patches and self.predicts_side else []) + self.groups_non_causal
        for group in groups:
            # Get current transformer to know number of batched_groups
            if group != 'backbone':
                curr_trans = self.transformer_modules[group]
            else:
                # If it's backbone we need to know number of frames, get it from batched_groups of 'space'
                curr_trans = self.transformer_modules['space']

            if self.predicts_side:
                # Prediction down (except first group which never predicts down)
                if group != groups[0]:
                    if curr_trans.num_batched_groups > 1:
                        # Get one random position per each batched_group
                        token_idx = [randint(0, curr_trans.num_tokens - 1)
                                     for _ in range(curr_trans.num_batched_groups)]
                    else:
                        token_idx = randint(0, curr_trans.num_tokens - 1)

                    # Store in prediction_pos of corresponding group
                    self.set_prediction_pos(group + '_d', token_idx)
                    token_idxs[group + '_d'] = token_idx

                # Prediction side
                # If only one output token it cannot be applied
                if curr_trans.num_batched_groups > 1:
                    tot_tokens = curr_trans.num_batched_groups
                    # Sample a temporal position to be predicted from a
                    #   normal distribution that decreases with distance with the position making the prediction
                    token_idx = [truncated_normal_sampling(p, max(p, tot_tokens-p)/self.std_div, 0, tot_tokens)
                                  for p in range(tot_tokens)]
                    self.set_prediction_pos(group + '_s', token_idx)
                    token_idxs[group + '_s'] = token_idx

            else:
                # TODO implement truncated_normal_sampling for this too
                #  (so nearby (and within) groups get predicted with higher probability)
                if curr_trans.num_batched_groups > 1:
                    # Number of tokens
                    tot_tokens = curr_trans.num_batched_groups * curr_trans.num_tokens

                    # Get one random position per each batched_group
                    token_idx = [randint(0, tot_tokens - 1) for _ in range(curr_trans.num_batched_groups)]
                else:
                    token_idx = randint(0, curr_trans.num_tokens - 1)

                # Store in prediction_pos of corresponding group
                self.set_prediction_pos(group + '_d', token_idx)

                token_idxs[group + '_d'] = token_idx

        return token_idxs

    # Used to indicate position of the spatial token that the temporal module is tasked with predicting
    def set_prediction_pos(self, group, pos):
        self.prediction_pos[group] = pos

    # The order in which parameters are given may be important (?), so trying to preserve it here.
    def get_params_for_group(self, group, named=False):
        params_module = list()
        params_pred = list()

        def call_params(x):
            if named:
                return x.named_parameters()
            else:
                return x.parameters()

        if group == 'backbone' or (group == 'space' and (not self.return_patches)):
            params_module += list(call_params(self.backbone))

            # Add BN (if any)
            if self.output_bn:
                params_module += list(call_params(self.output_norms['backbone']))

            # Add patchify (has a linear layer)
            params_module += list(call_params(self.to_patch_embedding))

            # Add linear projection
            if self.linear_proj.get('backbone', False):
                params_module += list(call_params(self.linear_proj['backbone']))

            # Add predictor (if any)
            if self.return_patches:
                params_pred += list(call_params(self.predictors['backbone_u']))
                if self.predicts_side:
                    params_pred += list(call_params(self.predictors['backbone_s']))
                    params_pred += [self.prediction_PEs['backbone_s']]

        if group != 'backbone':
            # Add the Transformer
            params_module += list(call_params(self.transformer_modules[group]))

            # Add the BN (if any)
            if self.output_bn:
                params_module += list(call_params(self.output_norms[group]))

                if self.transformer_modules[group].is_causal:
                    params_module += list(call_params(self.target_norms[group]))

            # Add linear projection (if any)
            if self.linear_proj.get(group, False):
                params_module += list(call_params(self.linear_proj[group]))

            # Add predictors
            for ending in ['_d', '_u', '_s']:
                if self.predictors.get(group + ending, False):
                    params_pred += list(call_params(self.predictors[group + ending]))
                    if ending == '_d' and (not self.predicts_side) and (
                            (group + ending + '_t') in self.prediction_PEs):
                        # 2D prediction has extra last names (t-oken and g-roup)
                        params_pred += [self.prediction_PEs[group + ending + '_t']]
                        params_pred += [self.prediction_PEs[group + ending + '_g']]
                    elif (ending == '_d' or ending == '_s') and (group + ending) in self.prediction_PEs:
                        params_pred += [self.prediction_PEs[group + ending]]

        return params_module, params_pred
