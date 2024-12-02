import yaml
import os
from datetime import timedelta
import wandb
import itertools
from copy import deepcopy, copy
import random
from torch import save, load

# Although pycharm regards these as not being used, they are!!
import torch.optim as optim
import torch.optim.lr_scheduler as sched
# TODO flash and ignite can work together??
# Commented out for now, as it is not used and current dockerfile fails to import it
# from flash.core.optimizers import LARS
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

DEBUG = False

def set_debug(to):
    global DEBUG
    DEBUG = to

# Sets new value for a nested key in the dictionary (has to already exist). Key depth separated by ':'
def set_nested_dict(d,k,v):
    temp = d
    all_keys = k.split(':')
    for key in all_keys[:-1]:
        temp = temp[key]
    temp[all_keys[-1]] = v


# given a configuration dictionary and a dictionary of parameters (each with a list of values to try)
# returns a list of config dictionaries with all combinations of those parameters
def sweep_params(config, sweep, names):
    all_configs = list()
    params = sweep.keys()
    # Iterate through whole combination of parameters
    combinations = itertools.product(*sweep.values())
    for combo in combinations:
        base_name = config['train']['model_name']
        curr_dict = deepcopy(config)
        for x, param in enumerate(params):
            set_nested_dict(curr_dict, param, combo[x])
            name = names[param]
            value = combo[x]
            if isinstance(value, bool):
                value = int(value)
            # If a value is str, remove possible underscores to avoid confusion
            #   replace them by camel case
            elif isinstance(value, str):
                # TODO find a better way to name values, this is very dirty and very specific
                #   for now it is just differentiating backbone or no backbone
                # The "arch:groups" variable is a list in the form of a string
                #   find a better way to name it (just first letter of each group)
                if '[' in value and isinstance(eval(value), list) and 'backbone' in value:
                    value = 'B'
                elif '[' in value and isinstance(eval(value), list):
                    value = 'noB'
                else:
                    pos = value.find('_')
                    if pos > 0:
                        value = value[:pos] + value[pos + 1].upper() + value[pos+2:]

            base_name += '_' + name + ('-' if name else '') + str(value)
        curr_dict['train']['model_name'] = base_name
        all_configs.append(curr_dict)
    return all_configs


def print_config_dict(c, prepend='', flag=True):
    out_text = ''
    for k, v in c.items():
        if flag:
            print(prepend, k, ': ', end='')
        out_text += prepend + str(k) + ' : '
        if isinstance(v,dict):
            if flag:
                print('\n')
            out_text += '\n'
            out_text += print_config_dict(v, prepend=prepend+'  ', flag=flag)
        else:
            if isinstance(v, str):
                out_text += '"'+v+'"\n'
                if flag:
                    print('"'+v+'"')
            elif v is None:
                out_text += ' \n'
                if flag:
                    print(v)
            else:
                out_text += str(v) + '\n'
                if flag:
                    print(v)
    return out_text


def save_config_file(config, file):
    with open(file, 'w') as f:
        f.write(print_config_dict(config, flag=False))


def update_config_dicts(default,custom):
    for k, v in custom.items():
        if k == 'name':
            return None
        if isinstance(v, dict):
            # Probably the .get wouldn't be necessary, as custom should always contain keys already on default
            ret = update_config_dicts(default.get(k, {}), custom[k])
            if ret is not None:
                default[k] = ret
            else:
                default[k] = custom[k]
        else:
            default[k] = v
    return default


def select_normalization_dataset(config,norms):
    if config['data']['data_augmentation']['use_datAug']:
        if config['data']['data_augmentation']['normalization'] == 'custom':
            if config['train']['dataset'] in norms:
                config['data']['data_augmentation']['normalization'] = config['train']['dataset']
            else:
                print("WARNING: Seleced 'custom' normalization dataset for data augmentation (" + config['train'][
                    'dataset'] +
                      ") was not found. Defaulting to ImageNet")
                config['data']['data_augmentation']['normalization'] = 'imagenet'
        elif config['data']['data_augmentation']['normalization'] != 'imagenet':
            if config['data']['data_augmentation']['normalization'] not in norms:
                print(
                    "WARNING: Selected normalization dataset (" + config['data']['data_augmentation']['normalization'] +
                    ") was not found. Defaulting to ImageNet")
                config['data']['data_augmentation']['normalization'] = 'imagenet'


def read_n_setup_params(current=None, path="./config", load_eval=False):
    config = dict()

    with open(os.path.join(path, 'data_config.yaml'), 'r') as f:
        config['data'] = yaml.load(f, Loader=yaml.loader.SafeLoader)

    with open(os.path.join(path, 'train_config.yaml'), 'r') as f:
        config['train'] = yaml.load(f, Loader=yaml.loader.SafeLoader)

    with open(os.path.join(path, 'arch_config.yaml'), 'r') as f:
        config['arch'] = yaml.load(f,Loader=yaml.loader.SafeLoader)

    if load_eval:
        with open(os.path.join(path, 'eval_config.yaml'), 'r') as f:
            config['eval'] = yaml.load(f,Loader=yaml.loader.SafeLoader)

    if current:
        with open(os.path.join(path, current), 'r') as f:
            custom = yaml.load(f, Loader=yaml.loader.SafeLoader)

        update_config_dicts(config, custom)

    return config


def build_args_as_str(params):
    args = ''
    for param,value in params.items():
        # If value is None, that param should not be utilized
        if param != 'name' and (value is not None):
            args += ',' + param + '=' + str(value)
    return args + ')'


# Expects same set of keys in acum dictionary and in new one (at all levels)
#    alternatively, acum can be empty, in which case new is copied into acum
# Used to produce summary losses at the end of an epoch
def update_acum_dict(acum, new):
    if acum:
        for k, v in acum.items():
            if isinstance(v, dict):
                update_acum_dict(v, new[k])
            else:
                acum[k] += new[k].item()
    else:  # acum is an empty dict, so just copy new
        for k, v in new.items():
            if isinstance(v, dict):
                acum[k] = dict()
                update_acum_dict(acum[k], v)
            else:
                acum[k] = new[k].item()


# TODO This logging is outdated: update it in case Wandb is not used!
# For test, batch_idx and total_batch are overloaded and used for accuracy
def log_training(task, time_count=None, batch_idx=-1, total_batch=-1, epoch=-1,
                 loss=float('inf')):
    if time_count is None:
        time_count = dict()
    if task == 'test':
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, batch_idx, total_batch,
            100. * batch_idx / total_batch))
        return

    eta = (time_count.get('total', 0) / (batch_idx + 1)) * (total_batch - batch_idx)
    if task.startswith('mine'):
        # In order for the logs to not get cluttered, only print with a 5% probability
        if random.random() > .95:
            print(('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}' +
                   '\tAvg. batch time: {:.3f}s / Epoch ETA: {}; DL: {:.3f}s; F: {:.3f}s; L: {:.3f}s; B: {:.3f}s').format(
                epoch, batch_idx, total_batch, 100. * batch_idx / total_batch, loss,
                # time_count['total'] / (batch_idx + 1), str(timedelta(seconds=eta)) ,
                # time_count['data']/(batch_idx + 1), time_count['forward']/(batch_idx + 1),
                # time_count['losses']/(batch_idx + 1), time_count['backprop']/(batch_idx + 1) ))
                time_count.get('total', 0), str(timedelta(seconds=eta)),
                time_count.get('data', 0), time_count.get('forward', 0),
                time_count.get('losses', 0), time_count.get('backprop', 0)))
    elif task == 'cls':
        # In order for the logs to not get cluttered, only print with a 5% probability
        if random.random() > .95:
            print(('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}' +
                  '\tAvg.BatchTime: {:.3f}s / Epoch ETA: {}').format(
                      epoch, batch_idx, total_batch,
                      100. * batch_idx / total_batch, loss,
                      time_count['total'] / (batch_idx + 1), str(timedelta(seconds=eta))
                  ))
    else:
        print("Logging function received wrong task: " + task)

# Function used to simulate training steps to see the progression the learning rate will have
def print_lr_over_epochs(optimizer, groups, instr_update_sch):
    for epoch in range(40):
        if isinstance(optimizer,dict):
            print(epoch,'space', round(optimizer['space'].param_groups[0]['lr'],3),
                           round(optimizer['s_proj'].param_groups[0]['lr'],3),
                  'time', round(optimizer['time'].param_groups[0]['lr'],3),
                           round(optimizer['t_proj'].param_groups[0]['lr'],3))
            for group in groups.keys():
                eval(instr_update_sch)
        else:
            print(epoch,round(optimizer.param_groups[0]['lr'],3), round(optimizer.param_groups[1]['lr'],3))
            eval(instr_update_sch)

# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
# https://pytorch.org/tutorials/beginner/saving_loading_models.html#warmstarting-model-using-parameters-from-a-different-model
# TODO Allow for changes when preloading a checkpoint
#   e.g. change the dataset (need different MLP) or number of frames/resolution
#   e.g. now I want a different alternation_interval
#   e.g. I want to change the groups, either by adding/removing
#       (having the same but restarting some would be different),
#       may be useful to check the "strict" param on the load funcion
#   In case the model class has changed (and possibly, consequently some config file variables)
#       either it is compatible --> use new variables from new config file
#       it is not, some fundamental change --> checkpoint is no longer usable
# Loads a model either from a checkpoint or from the tmp file
# If checkpoint, subsequent args must be provided
# Else, epoch must be provided
def load_model(model, config, epoch=0, extra='', checkpoint=False,
               load_optim_sched=True, optim=None, sched=None,
               strict=True):
    # Setup path, either checkpoint folder or tmp folder
    base_dir = config['train']['save_path'] if checkpoint else config['train']['tmp_path']
    base_name = config['train']['resume_path'] if checkpoint else config['train']['model_name'] + extra
    full_path = os.path.join(base_dir, base_name)
    if checkpoint:
        full_name = base_name + '_' + config['train']['resume_mod'] + '_e' + str(config['train']['resume_epoch'])
    else:
        full_name = base_name + '_ssl_e' + str(epoch)
    path_n_name = os.path.join(full_path, full_name + '.pt')

    if checkpoint:
        saved_dict = load(path_n_name)
        model.load_model_weights(saved_dict['model_state_dict'], strict=strict)
        if load_optim_sched:
            # TODO if checkpoint is supervised optim won't be a dictionary
            for o in optim:
                optim[o].load_state_dict(saved_dict['optim_state_dict'][o])
            if type(saved_dict['sched_state_dict']) is dict:
                for s in sched:
                    sched[s].load_state_dict(saved_dict['sched_state_dict'][s])
            else:
                sched.load_state_dict(saved_dict['sched_state_dict'])
        return saved_dict['loss']
    else:
        model.load_state_dict(load(path_n_name))

# If checkpoint, subsequent args must be provided
def save_model(model, config, epoch, extra='', mod='ssl', checkpoint=False, best_loss=-1, optim=None, sched=None):
    base_dir = config['train']['save_path'] if checkpoint else config['train']['tmp_path']
    base_name = config['train']['model_name'] + extra
    full_path = os.path.join(base_dir, base_name)
    full_name = base_name + '_' + mod + '_e' + str(epoch) + '.pt'
    path_n_name = os.path.join(full_path,
                               full_name)

    # Check if there exists a folder for this experiment
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    if checkpoint:
        # When saving a checkpoint, the configuration is only saved once
        config_file_path = os.path.join(full_path, 'config.yaml')
        if not os.path.exists(config_file_path):
            save_config_file(config, config_file_path)

        if isinstance(optim, dict):
            save_optim = {k: v.state_dict() for k, v in optim.items()}
        else:
            save_optim = optim.state_dict()

        if isinstance(sched, dict):
            save_sched = {k: v.state_dict() for k, v in sched.items()}
        else:
            save_sched = sched.state_dict()

        # See https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
        #   for an example on saving/loading all these
        save({'model_state_dict': model.state_dict(),
              'optim_state_dict': save_optim,
              'sched_state_dict': save_sched,
              'loss': best_loss}, path_n_name)
        print(f"Saved checkpoint in {path_n_name}")
    else:
        # Just saving temporally for sanity check
        save(model.state_dict(), path_n_name)


def save_model_if_better(model, optim, sched, config, epoch, loss, best, mod='ssl', extra=''):
    if loss < best:
        save_model(model, config, epoch, extra, mod, True, loss, optim, sched)
        return loss, True
    else:
        return best, False


# Step towards uniform loss weighting (all losses have weight of 1). Linear interpolation.
def step_loss_weights(loss, e, te):
    for l, weight in loss.items():
        loss[l] = round(weight - ((weight - 1) / (te - e)),3)

# counts total number of parameters of a model
# code inspired on accepted answer in:
#   https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def create_classification_optimizer_and_scheduler(config, model, mod=''):
    instruction = 'optim.' + config['train']['val_opt']['name'] + '(model.parameters()'
    instruction += build_args_as_str(config['train']['val_opt'])
    optimizer = eval(instruction)

    instruction = 'sched.' + config['train']['val_sch']['name'] + '(optimizer'
    instruction += build_args_as_str(config['train']['val_sch'])
    scheduler = eval(instruction)

    instr_update_sch = 'scheduler'+mod+'.step()'

    return optimizer, scheduler, instr_update_sch


def create_general_optimizer_and_scheduler(args_opt, args_sch, args_wup, parameters, mod=''):
    # LARS optimizer is form different library
    if args_opt['name'] == 'LARS':
        instr_opt = 'LARS('
    else:
        instr_opt = 'optim.' + args_opt['name'] + '('

    instr_opt += 'parameters' + build_args_as_str(args_opt)
    instr_sch = 'sched.' + args_sch['name'] + '(optimizer' + build_args_as_str(args_sch)

    optimizer = eval(instr_opt)
    scheduler = eval(instr_sch)

    instr_update_sch = 'scheduler' + mod

    # SETUP WARMUP
    if args_wup['name']:
        instr_wup = 'create_lr_scheduler_with_warmup(scheduler' + \
                    build_args_as_str(args_wup)
        scheduler = eval(instr_wup)

        # Initialise the warmup scheduler
        scheduler(None)

        instr_update_sch += '(None)'
    else:
        instr_update_sch += '.step()'

    return optimizer, scheduler, instr_update_sch

# TODO allow for different optimizer params for the different groups
def select_right_optimizer(config, groups, model):
    optimizer = dict()
    scheduler = dict()

    # Also see https://pytorch.org/docs/stable/optim.html#per-parameter-options for other options
    args_opt = config['train']['optimizer']
    args_sch = config['train']['scheduler']
    args_wup = config['train']['warmup']

    all_params = list()
    all_params_mod = dict()
    all_params_pred = dict()
    # all_params_by_group = list()
    if config['train']['task'] == 'mine':
        for group in groups:
            all_params_mod[group], all_params_pred[group] = model.get_params_for_group(group, named=False)
            all_params += (all_params_mod[group] + all_params_pred[group])
            # all_params_by_group += [(group, x[0]) for x in all_params_mod[group]] + \
            #                        [(group, x[0]) for x in all_params_pred[group]]
        mlp_params = list(model.mlp_head.parameters())
    else:
        all_params_mod, all_params_pred = model.get_params(named=False)
        all_params = all_params_mod + all_params_pred
        mlp_params = []

    ###############################
    # This is a helper snippet to help find the inconsistencies in missing parameters
    # all_params_by_group above also needs to be uncommented
    # Also, it expects model.get_params_for_group() to return named_parameters (change named to True)

    # Take all detected params
    # all_names = sorted([p for p in all_params_by_group], key=lambda x: (x[1],x[0]))
    # module_names = list()
    #
    # # Take all actual params
    # for p in model.named_parameters():
    #     name = p[0]
    #     if name.startswith('backbone'):
    #         base = 'backbone'
    #         name = '.'.join(name.split('.')[1:])
    #     else:
    #         base = '.'.join(name.split('.')[:2])
    #         name = '.'.join(name.split('.')[2:])
    #     module_names.append((base, name))
    #
    # module_names = sorted(module_names, key=lambda x: (x[1], x[0].split('.')[-1]))
    #
    #
    # x = 0
    # y = 0
    # while x < len(all_names) and y < len(module_names):
    #     # If a param is in both lists, print that
    #     if all_names[x][1] == module_names[y][1]:
    #         print(all_names[x], module_names[y][1], module_names[y][0])
    #         x += 1
    #         y += 1
    #     # If a param was found, but it is not on the actual param list
    #     #  i.e., pytorch missed some layers or something
    #     elif len(all_names) > len(module_names):
    #         print(all_names[x])
    #         x += 1
    #     # If a param was not found
    #     #  i.e., it does not belong to any group
    #     else:
    #         print('\t\t\t', module_names[y][1], module_names[y][0])
    #         y += 1
    #
    # if x < (len(all_names)-1):
    #     while x < len(all_names):
    #         print("an", all_names[x])
    #         x += 1
    #
    # if y < (len(module_names)-1):
    #     while y < len(module_names):
    #         print("mn", module_names[x])
    #         y+=1
    ##############################

    # Account for MLP layers, which do not belong to any group
    #   (no need to have them here as they will be provided their own optimizer later)
    assert (len(all_params) + len(mlp_params)) == len(list(model.parameters())), \
           "Some parameters won't be trained!\n" + \
           "If layers were added/removed from the model, get_params_for_group may need updating."

    # Baselines only need a single optimizer
    # TODO allow for different optimizers for module/predictor for the baselines too!!!
    if config['train']['task'] != 'mine':
        optimizer, scheduler, instr_update_sch = \
            create_general_optimizer_and_scheduler(args_opt, args_sch, args_wup, all_params)

    # Train predictors together with corresponding module
    elif config['train']['predictor']['pred_aside'] == 0:
        # All modules are trained with the same optimizer
        if config['train']['use_single_optimizer']:
            ###################################
            # Single optimizer for everything #
            ###################################
            optimizer['module'], scheduler, instr_update_sch = \
                create_general_optimizer_and_scheduler(args_opt, args_sch, args_wup, all_params)

        # Each module has its own optimizer
        else:
            ######################################
            # Different optimizer for each group #
            ######################################
            for group in groups:
                optimizer[group], scheduler[group], instr_update_sch = \
                    create_general_optimizer_and_scheduler(args_opt, args_sch, args_wup,
                                                           all_params_mod[group] + all_params_pred[group],
                                                           mod='[group]')
    else:
        pred_args_opt = copy(args_opt)
        pred_args_sch = copy(args_sch)
        pred_args_wup = copy(args_wup)

        # Apply learning rate multiplier for predictors
        mult = config['train']['predictor']['pred_multiplier']
        pred_args_opt['lr'] *= mult
        pred_args_opt['weight_decay'] = pred_args_opt.get('weight_decay', 0) * mult
        pred_args_wup['warmup_start_value'] = pred_args_wup.get('warmup_start_value', 0) * mult
        if args_sch['name'] == 'CosineAnnealingLR':
            pred_args_sch['eta_min'] *= mult

        if config['train']['use_single_optimizer']:
            apm = list()
            app = list()
            for group in groups:
                apm += all_params_mod[group]
                app += all_params_pred[group]

        # Train predictor together with corresponding module (but in different param group)
        if config['train']['predictor']['pred_aside'] == 1:
            # All modules are trained with the same optimizer
            if config['train']['use_single_optimizer']:
                #############################################################
                # Single optimizer, but predictors in different param_group #
                #############################################################
                # TODO LR parameters of scheduler and warmup are same regardless of group!!
                #  optimizer initial will help, but!
                param_set = [{'params': apm},
                             {'params': app,
                              'lr': pred_args_opt['lr'],
                              'weight_decay': pred_args_opt['weight_decay']}]
                optimizer['module'], scheduler, instr_update_sch = \
                    create_general_optimizer_and_scheduler(args_opt, args_sch, args_wup,
                                                           param_set)

            # Each module has its own optimizer
            else:
                #############################################################################
                # Different optimizer for each group and predictors in separate param_group #
                #############################################################################
                for group in groups:
                    # TODO LR parameters of scheduler and warmup are same regardless of group!!
                    #  optimizer initial will help, but!
                    param_set = [{'params': all_params_mod[group]},
                                 {'params': all_params_pred[group], 'lr': pred_args_opt['lr'],
                                  'weight_decay': pred_args_opt['weight_decay']}]
                    optimizer[group], scheduler[group], instr_update_sch = \
                        create_general_optimizer_and_scheduler(args_opt, args_sch, args_wup,
                                                               param_set, mod='[group]')

        # Train predictors with separate optimizers
        else:
            # All modules are trained with the same optimizer
            if config['train']['use_single_optimizer']:
                #####################################################
                # Two optimizers, one for model, one for predictors #
                #####################################################
                optimizer['module'], scheduler['module'], instr_update_sch = \
                    create_general_optimizer_and_scheduler(args_opt, args_sch, args_wup,
                                                           apm,
                                                           mod='[group]')
                optimizer['predictor'], scheduler['predictor'], instr_update_sch = \
                    create_general_optimizer_and_scheduler(pred_args_opt, pred_args_sch, pred_args_wup,
                                                           app,
                                                           mod='[group]')
            # Each module has its own optimizer
            else:
                ####################################################
                # Different optimizer for each group and predictor #
                ####################################################
                for group in groups:
                    optimizer[group + '_mod'], scheduler[group + '_mod'], instr_update_sch = \
                        create_general_optimizer_and_scheduler(args_opt, args_sch, args_wup,
                                                               all_params_mod[group],
                                                               mod='[group]')
                    optimizer[group + '_pred'], scheduler[group + '_pred'], instr_update_sch = \
                        create_general_optimizer_and_scheduler(pred_args_opt, pred_args_sch, pred_args_wup,
                                                               all_params_pred[group],
                                                               mod='[group]')

    return optimizer, scheduler, instr_update_sch



if __name__ == '__main__':
    param_names = {'arch:groups': 'gr',
                   'train:use_single_optimizer': 'opt',
                   'train:predictor:pred_aside': 'pA',
                   'train:loss_fn': 'LFn'}
    sweep_vals = {'arch:groups': ["['space', 'time']",
                                  "['backbone', 'space', 'time']"],
                  'train:use_single_optimizer': [True, False],
                  'train:predictor:pred_aside': [0,1,2],
                  'train:loss_fn': ['CE','MSE','CS']}

    # LOAD ONLY THE CUSTOM ONE SO REMAINING FIXED PARAMETERS CAN BE MODIFIED IN DEFAULT FILES
    print("################################################ \n ST")
    config_name = 'custom_christmas_st{}.yaml'
    with open(os.path.join('./config', 'custom_christmas_st.yaml'), 'r') as f:
        original_config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    for x, cfg in enumerate(sweep_params(original_config, sweep_vals, param_names)):
        #print(cfg)
        save_config_file(cfg, './config/'+config_name.format(x))

    print("################################################ \n ST_causal")
    config_name = 'custom_christmas_st_causal{}.yaml'
    with open(os.path.join('./config', 'custom_christmas_st_causal.yaml'), 'r') as f:
        original_config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    for x, cfg in enumerate(sweep_params(original_config, sweep_vals, param_names)):
        #print(cfg)
        save_config_file(cfg, './config/'+config_name.format(x))

    sweep_vals['arch:groups'] = ["['space', 'time', 'clip']",
                                 "['backbone', 'space', 'time', 'clip']"]

    print("################################################ \n STC")
    config_name = 'custom_christmas_stc{}.yaml'
    with open(os.path.join('./config', 'custom_christmas_stc.yaml'), 'r') as f:
        original_config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    for x, cfg in enumerate(sweep_params(original_config, sweep_vals, param_names)):
        #print(cfg)
        save_config_file(cfg, './config/'+config_name.format(x))

    print("################################################ \n STC_causal")
    config_name = 'custom_christmas_stc_causal{}.yaml'
    with open(os.path.join('./config', 'custom_christmas_stc_causal.yaml'), 'r') as f:
        original_config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    for x, cfg in enumerate(sweep_params(original_config, sweep_vals, param_names)):
        #print(cfg)
        save_config_file(cfg, './config/'+config_name.format(x))
