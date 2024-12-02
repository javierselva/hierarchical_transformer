import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.profiler import profile, record_function, ProfilerActivity

from losses import compute_losses, token_variance
from eval import test

import time as tiempo
from random import random
# import gc

from utils import log_training, create_classification_optimizer_and_scheduler, update_acum_dict
from data_utils.precomputed_features import PreComputedFeatures
from architecture.ht_mine import Stage, Task, DataAugmentation
from wandb_log import wandb_logging, set_debug

# Epoch loops are broken during debug if number of batches reaches MAX_DEBUG_ITERS
MAX_DEBUG_ITERS = 150
DEBUG = False

def set_wandb_log_off(on):
    global DEBUG
    DEBUG = on
    set_debug(on)

def check_dict_nan(d):
    failed_keys = list()
    for k, v in d.items():
        is_nan = torch.sum(torch.isnan(v))
        if is_nan > 0:
            failed_keys.append(k)
    return failed_keys

# ***************************************************************************
# *************            TRAINING FUNCTIONS             *******************
# ***************************************************************************
def train_cls(model, device, data_loader, optimizer, epoch, stage, log_interval, scaler,
              sepoch=-1, num_sepochs=0, val_rounds=-1, save_path='/tmp', save_file='',
              use_datAug=(0.0,60,60,None), num_iters_grad_accum=1, use_mixed_prec=(False, "float16")):
    model.set_stage(stage)
    time_count = dict()
    total_loss = 0

    epoch_time = tiempo.time()

    use_datAug, h, w, aug_params = use_datAug
    if use_datAug:
        datAug = DataAugmentation(h, w, aug_params)

    use_mixed_prec, data_type = use_mixed_prec
    if data_type == 'float16':
        data_type = torch.float16
    elif data_type == 'bfloat16':
        data_type = torch.bfloat16

    # Total number of batches in a single epoch
    num_batches_per_epoch = len(data_loader)

    if DEBUG:
        iterat = 0

    optimizer.zero_grad()
    
    for batch_idx, (data, label) in enumerate(data_loader):

        if DEBUG:
            if iterat > MAX_DEBUG_ITERS:
                break
            iterat += 1

        s0 = tiempo.time()

        data, label = data.to(device), label.to(device)
        if data.shape[0] == 1:
            print(f"Warning! Only one sample in batch {batch_idx}/{num_batches_per_epoch}!")
            print(f"Importantly, the dataset had {len(data_loader.dataset)} samples!")
            continue
        
        
        # Data augmentation (unless probing without augmentation)
        if random() < use_datAug and (stage != Stage.PROBE or model.probe_augment):
            data = datAug(data)
        
        
        # time_count['data'] = time_count.setdefault('data', 0) + (tiempo.time() - s0)

        # s = tiempo.time()
        with torch.autocast(device_type='cuda', dtype=data_type, enabled=use_mixed_prec):
            # If it's first linear probing round, store features for faster next iterations as rest of model is frozen
            output = model(data)
            # time_count['forward'] = time_count.setdefault('forward', 0) + (tiempo.time() - s)
            
            # CrossEntropy expects a class number, actually
            loss = F.nll_loss(output, label) / num_iters_grad_accum
            

        # s = tiempo.time()
        scaler.scale(loss).backward()
        
        # Backpropagate if number of batches for gradient accumulation is reached or it is last batch
        if (batch_idx + 1) % num_iters_grad_accum == 0 or batch_idx == num_batches_per_epoch - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        # time_count['backward'] = time_count.setdefault('backward', 0) + (tiempo.time() - s)
        
        # TODO should total loss be summed before scaling the loss??
        total_loss += loss.item()

        time_count['total'] = time_count.setdefault('total', 0) + (tiempo.time() - s0)
        if batch_idx % log_interval == 0:

            log_training('cls', epoch=epoch, batch_idx=batch_idx, total_batch=num_batches_per_epoch,
                         loss=loss.item(), time_count=time_count)
            wandb_logging(stage, Task.CLS if stage == Stage.TRAIN else Task.MINE,
                          epoch, batch_idx, num_batches_per_epoch, {'cls': loss},
                          sepoch=sepoch, num_sepochs=num_sepochs, val_rounds=val_rounds)
    epoch_time = tiempo.time() - epoch_time
    wandb_logging(stage, Task.CLS if stage == Stage.TRAIN else Task.MINE,
                  epoch, batch_idx, num_batches_per_epoch, {'cls': total_loss},
                  summary=True, sepoch=sepoch, num_sepochs=num_sepochs, val_rounds=val_rounds,
                  epoch_time=epoch_time)

    del loss
    del output

    if is_first_probing_round:
        new_dataset.finished_saving()
        return new_dataset


def train_mine(model, device, data_loader, optimizer, epoch, log_interval, scaler,
               loss_fcn, loss_weights, loss_params,
               use_datAug=(0.0,60,60,None), num_iters_grad_accum=1,
               use_mixed_prec=(False, "float16")):
    model.train(Stage.TRAIN)
    # Dictionary to store time elapsed for different parts of the training
    time_count = dict()
    # Number of optimizers
    num_opt = len(optimizer)
    # Indicates whether predictors are trained separately from modules
    preds_aside = list(optimizer.keys())[0].find('_') != -1
    # Total number of batches in a single epoch
    num_batches_per_epoch = len(data_loader)
    # Switch to swap which groups are being rained
    switch = round(num_batches_per_epoch * (model.alternating_interval / 100.))
    if switch > 0:
        model.set_trainable_groups(True)
    # Dictionary to accumulate the losses throughout an epoch
    epoch_losses = dict()
    epoch_time = tiempo.time()

    # TODO datAug is always the same, could be created outside
    use_datAug, h, w, aug_params = use_datAug

    if use_datAug:
        datAug = DataAugmentation(h, w, aug_params)

    batch_idx = 0

    use_mixed_prec, data_type = use_mixed_prec
    if data_type == 'float16':
        data_type = torch.float16
    elif data_type == 'bfloat16':
        data_type = torch.bfloat16
    else:
        raise(Exception, "Wrong data type selected for mixed precision. Got:", data_type,
                         "but expected float16 or bfloat16")

    if DEBUG:
        iterat = 0

    for o in optimizer.values():
        o.zero_grad()
    
    for batch_idx, (data, _) in enumerate(data_loader):
        if DEBUG:
            if iterat > MAX_DEBUG_ITERS:
                break
            iterat += 1

        if data.shape[0] == 1:
            print(f"Warning! Only one sample in batch {batch_idx}/{num_batches_per_epoch}!")
            print(f"Importantly, the dataset had {len(data_loader.dataset)} samples!")
            continue

        s0 = tiempo.time()
        # No need for the targets in this setting
        data = data.to(device)
        
        if random() < use_datAug:
            data = datAug(data)
        
        #time_count['data'] = time_count.setdefault('data', 0) + (tiempo.time() - s0)
        time_count['data'] = (tiempo.time() - s0)

        # Switch the groups being optimized
        if switch > 0 and batch_idx > 0 and batch_idx % switch == 0:
            model.switch_training_groups()

        # PICK PREDICTION TOKENS FOR NON-CAUSAL MODE
        token_idxs = model.set_random_prediction_pos()
        
        # RUN THROUGH MODEL
        # TODO If problems observed, may need gradient clipping, then unscale gradients!!
        #   https://blog.paperspace.com/automatic-mixed-precision-using-pytorch/
        with torch.autocast(device_type='cuda', dtype=data_type, enabled=use_mixed_prec):
            # Compute output
            s = tiempo.time()
            # SNIPPET TO PROFILE GPU/CPU USAGE DURING THE FORWARD PASS (Include the print prof.key_averages() below)
            # with_stack=True can be added to help find culprits in the code;
            #   in such case add "group_by_stack_n=5" to the key_averages call in the print below
            # with profile(activities=[ProfilerActivity.CPU], #, ProfilerActivity.CUDA],
            #              profile_memory=True) as prof:
            ############
            target, target_agg, prediction = model(data)
            
            #time_count['forward'] = time_count.setdefault('forward', 0) + (tiempo.time() - s)
            time_count['forward'] = (tiempo.time() - s)
            
            # NaN SANITY CHECKS
            out_str = ''
            which_nan = check_dict_nan(target)
            if len(which_nan) > 0:
                out_str += f'targets NaN-ed in {which_nan};'
            which_nan = check_dict_nan(target_agg)
            if len(which_nan) > 0:
                out_str += f'targets_agg NaN-ed in {which_nan};'
            which_nan = check_dict_nan(prediction)
            if len(which_nan) > 0:
                out_str += f'predictions NaN-ed in {which_nan};'
            assert len(out_str) == 0, 'Woah!!' + out_str
            
            #############
            # print(prof.key_averages().table(sort_by="cpu_time_total"))
            #############
            
            # COMPUTE THE LOSSES
            s = tiempo.time()
            losses = compute_losses(model, target_agg, prediction, loss_fcn,
                                    model.predicts_side, loss_params, target, token_idxs)
            
            #time_count['losses'] = time_count.setdefault('losses', 0) + (tiempo.time() - s)
            time_count['losses'] = (tiempo.time() - s)

            # Sum all losses (per predictor and group) of the groups being trained
            total_loss = 0
            for group, pred_loss in losses.items():
                if model.trainable_groups[group]:
                    for pred in pred_loss:
                        losses[group][pred] *= loss_weights[group + '_' + pred]
                        losses[group][pred] /= num_iters_grad_accum
                        total_loss += losses[group][pred]
            

        ################################
        # SNIPPET TO SEE TENSORS AND THEIR MEMORY SIZE IN MB
        # objects = list()
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             objects.append((obj, obj.element_size() * obj.nelement() / (1024**2)))
        #     except:
        #         pass
        ################################

        s = tiempo.time()
        if num_opt == 1 or (num_opt == 2 and 'predictor' in optimizer):
            scaler['module'].scale(total_loss).backward()
            
            # Backpropagate
            if (batch_idx + 1) % num_iters_grad_accum == 0 or batch_idx == num_batches_per_epoch - 1:
                for opt in optimizer.values():
                    scaler['module'].step(opt)
                    opt.zero_grad()

                scaler['module'].update()
        else:
            # Compute gradient for each group separately
            for idx, (group, pred_loss) in enumerate(losses.items()):
                if model.trainable_groups[group]:
                    # Aggregate per group loss
                    current_loss = sum([loss for loss in pred_loss.values()])
                    # Only retain graph if it is not the last call to backward
                    scaler[group].scale(current_loss).backward(retain_graph=idx < len(losses) - 1)

            
            # Backpropagate only the trainable ones!!
            if (batch_idx + 1) % num_iters_grad_accum == 0 or batch_idx == num_batches_per_epoch - 1:
                for group in model.get_trainable_groups():
                    if preds_aside:
                        scaler[group].step(optimizer[group+'_pred'])
                        scaler[group].step(optimizer[group+'_mod'])

                        optimizer[group + '_pred'].zero_grad()
                        optimizer[group + '_mod'].zero_grad()
                    else:
                        scaler[group].step(optimizer[group])
                        optimizer[group].zero_grad()

                    scaler[group].update()

        

        #time_count['backprop'] = time_count.setdefault('backprop', 0) + (tiempo.time() - s)
        time_count['backprop'] = (tiempo.time() - s)

        # COMPUTE TOKEN VARIANCE FOR TARGETS (MONITOR COLLAPSE)
        # TODO Token variance is measured after BN, not sure that makes sense (??)
        #   I am tracking the variance of fatures geting in the next group, but not the output ones, which
        #   are the ones that may be collapsing
        for t, tgt in target_agg.items():
            losses[t]['var'] = torch.mean(token_variance(tgt))

        #time_count['total'] = time_count.setdefault('total', 0) + (tiempo.time() - s0)
        time_count['total'] = (tiempo.time() - s0)

        update_acum_dict(epoch_losses, losses)

        if batch_idx % log_interval == 0:
            log_training('mine', epoch=epoch, batch_idx=batch_idx, total_batch=num_batches_per_epoch,
                         loss=total_loss.item(), time_count=time_count)

            wandb_logging(Stage.TRAIN, Task.MINE, epoch, batch_idx, num_batches_per_epoch, losses, loss_fcn=loss_fcn)
    epoch_time = tiempo.time() - epoch_time
    wandb_logging(Stage.TRAIN, Task.MINE, epoch, batch_idx, num_batches_per_epoch, epoch_losses,
                  summary=True, loss_fcn=loss_fcn, epoch_time=epoch_time)
    del losses, epoch_losses, total_loss
    del target, target_agg, prediction


# ***************************************************************************
# *************            TESTING/VALIDATION             *******************
# ***************************************************************************

# TODO Pass only train config
# Fine-tune the model for downstream supervised task
def fine_tune(config, model, device, train_loader, test_loader, epoch, scaler, loss_params,
              val_rounds=-1, use_datAug=(0.0,60,60), num_iters_grad_accum=1,
              use_mixed_prec=False):

    if hasattr(train_loader.dataset,'set_single_sequence'):
        train_loader.dataset.set_single_sequence()

    # Create optimizer for supervised training
    optimizer_ft, scheduler_ft, ins_upd_sch_ft = create_classification_optimizer_and_scheduler(config, model,
                                                                                               mod='_ft')
    for e in range(1, config['train']['ftune_epochs'] + 1):
        # Train using pre-computed features
        train_cls(model, device, train_loader, optimizer_ft, epoch,
                  Stage.FTUNE, config['train']['log_interval'], scaler, e,
                  config['train']['ftune_epochs'],
                  val_rounds, use_datAug=use_datAug,
                  num_iters_grad_accum=num_iters_grad_accum,
                  use_mixed_prec=use_mixed_prec)
        # Scheduler step
        eval(ins_upd_sch_ft)

    # TEST
    l = test(model, device, test_loader, epoch, loss_params, task=Task.str2task(config['train']['task']))

    # Free optimizer as it will be reset?
    del optimizer_ft
    del scheduler_ft

    if hasattr(train_loader.dataset,'reset_single_sequence'):
        train_loader.dataset.reset_single_sequence()

    return l


# Fit linear layer at frozen model's output for downstream supervised task
def linear_probe(config, model, device, train_loader, test_loader, epoch, scaler, loss_fcn,
                 loss_params, rand_num=-1, val_rounds=-1, use_datAug=(0.0,60,60),
                 num_iters_grad_accum=1, use_mixed_prec=False):
    # Reset mlp_head in case it is needed
    model.probe()
    
    if hasattr(train_loader.dataset, 'set_single_sequence'):
        train_loader.dataset.set_single_sequence()

    # Get mlp_head for the optimizer
    layers_to_train = model.get_mlp_head()

    # Create new dataloader with test_batch_size, so it is a bit faster
    # if config['train']['test_batch_size'] > 0 and config['train']['test_batch_size'] != config['train']['batch_size']:
    #     # from data_utils.data_handler_frames import CustomDataset
    #     # daset = CustomDataset(config, 'train', dataset=config['train']['dataset'])
    #     # TODO, actually doing this seems to cause some OOM problems...
    #     aux_data_loader = DataLoader(train_loader.dataset,
    #                                  batch_size=config['train']['test_batch_size'],
    #                                  shuffle=True,
    #                                  num_workers=config['data']['dl_workers'],
    #                                  persistent_workers=config['data']['dl_workers'] > 0)
    # else:
    aux_data_loader = train_loader

    # Create optimizer for supervised training of classification layer
    optimizer_lp, scheduler_lp, ins_upd_sch_lp = create_classification_optimizer_and_scheduler(config, layers_to_train,
                                                                                               mod='_lp')

    # TODO extracting features should not be necessary if using data augmentation during probing,
    #  but as it breaks the logic of everything else it is kept
    # Train final layer supervisedly and extract transformer features
    
    new_dataset = train_cls(model, device, aux_data_loader, optimizer_lp, epoch,
                            Stage.PROBE, config['train']['log_interval'], scaler, 1,
                            config['train']['probe_epochs'],
                            val_rounds,
                            config['train']['tmp_path'],
                            config['train']['model_name'] + str(rand_num),
                            use_datAug=use_datAug,
                            num_iters_grad_accum=num_iters_grad_accum,
                            use_mixed_prec=use_mixed_prec)
    
    if model.probe_augment:
        if config['train']['probe_batch_size'] != -1 and \
                config['train']['batch_size'] != config['train']['probe_batch_size']:
            new_data_loader = DataLoader(train_loader.dataset,
                                         batch_size=config['train']['probe_batch_size'],
                                         shuffle=True,
                                         num_workers=config['data']['dl_workers'],
                                         persistent_workers=config['data']['dl_workers'] > 0)
        else:
            new_data_loader = train_loader
    else:
        # Now the workers are in charge of actually pre-loading the dataset on the first call to __getitem__
        # TODO if only 1 epoch of probing, creating a new dataLoader is not necessary
        bs = config['train']['batch_size'] if config['train']['probe_batch_size'] == -1 else config['train']['probe_batch_size']
        new_data_loader = DataLoader(new_dataset,
                                     batch_size=bs,
                                     shuffle=True,
                                     num_workers=config['data']['dl_workers'],
                                     persistent_workers=config['data']['dl_workers'] > 0)
    
    for e in range(2, config['train']['probe_epochs'] + 1):

        # Scheduler step
        eval(ins_upd_sch_lp)
        # Train using pre-computed features
        train_cls(model, device, new_data_loader, optimizer_lp, epoch,
                  Stage.PROBE, config['train']['log_interval'], scaler, e,
                  config['train']['probe_epochs'],
                  val_rounds, use_datAug=use_datAug,
                  num_iters_grad_accum=num_iters_grad_accum,
                  use_mixed_prec=use_mixed_prec)
    

    # TEST
    l = test(model, device, test_loader, epoch, loss_params, task=Task.str2task(config['train']['task']), loss_fcn=loss_fcn)

    
    # Free optimizer as it will be reset?
    del optimizer_lp
    del scheduler_lp
    del aux_data_loader
    if not model.probe_augment:
        # Remove HDF5 tmp file
        new_dataset.delete_file()
        del new_dataset
        # Remove data_loader
        del new_data_loader

    if hasattr(train_loader.dataset,'reset_single_sequence'):
        train_loader.dataset.reset_single_sequence()

    return l
