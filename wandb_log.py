import wandb

from architecture.ht_mine import Stage, Task

DEBUG = False

def set_debug(to):
    global DEBUG
    DEBUG = to


def get_ssl_batch(epoch, num_batches, batch_idx):
    return (epoch-1)*num_batches + batch_idx


def get_supv_epoch(val_rounds, sepoch, num_sepochs):
    # how many epochs depending on the number of val_rounds
    return val_rounds * num_sepochs + sepoch


def get_supv_batch(val_rounds, sepoch, num_sepochs, num_batches, batch_idx):
    # how many validation epochs have I done
    total_val_epochs = get_supv_epoch(val_rounds, sepoch, num_sepochs)
    # how many batches would that be
    return total_val_epochs * num_batches + batch_idx


def generate_per_predictor_metrics(base_name, step_metric, modules, loss):
    if base_name.endswith('/'):
        base_name = base_name[:-1]

    base_name += '/' + loss
    # Total loss for the network
    wandb.define_metric(base_name + "/total", step_metric=step_metric)

    # Per-module losses
    for m, module in enumerate(modules):
        # Per predictor Losses
        for predictor in predictors:
            # Unless it is a module in the extremes, add a total for both its predictors
            if m != 0 and m != (len(modules) - 1):
                wandb.define_metric(base_name + '/' + module + '/total', step_metric=step_metric)
            if (m == 0 and predictor == 'down') or \
                    (m == (len(modules) - 1) and (predictor == 'up' or predictor == 'side')):
                continue  # First module doesn't predict down, last doesn't predict up or sideways
            wandb.define_metric(base_name + '/' + module + '/' + predictor,
                                step_metric=step_metric)


def create_wandb_metrics(modules, also_ftune, task, loss=None, pred_aside=False):
    # STEPS
    # Batch
    wandb.define_metric("c_step/train/batch")
    if task != Task.CLS:
        wandb.define_metric("c_step/probe/batch")
        if also_ftune:
            wandb.define_metric("c_step/ftune/batch")

    # Epoch
    wandb.define_metric("c_step/train/epoch", step_metric="c_step/train/batch")
    if task != Task.CLS:
        wandb.define_metric("c_step/probe/epoch", step_metric="c_step/probe/batch")
        if also_ftune:
            wandb.define_metric("c_step/ftune/epoch", step_metric="c_step/ftune/batch")

    # Learning Rate
    wandb.define_metric("lr", step_metric="c_step/train/epoch")
    if pred_aside:
        wandb.define_metric("lr_p", step_metric="c_step/train/epoch")

    # TOTAL LOSS FOR THE NETWORK
    # Loss per batch
    if task != Task.CLS:
        if task == Task.MINE:
            generate_per_predictor_metrics("train/batch", "c_step/train/batch", modules, loss)
        else:
            wandb.define_metric(f"train/batch/{str(task).split('.')[-1].lower()}", step_metric="c_step/train/batch")
        wandb.define_metric("probe/batch/cls", step_metric="c_step/probe/batch")
        if also_ftune:
            wandb.define_metric("ftune/batch/cls", step_metric="c_step/ftune/batch")
    else:
        wandb.define_metric("train/batch/cls", step_metric="c_step/train/batch")

    # Loss per epoch (mean over epoch)
    if task != Task.CLS:
        if task == Task.MINE:
            generate_per_predictor_metrics("train/epoch", "c_step/train/epoch", modules, loss)
        else:
            wandb.define_metric(f"train/epoch/{str(task).split('.')[-1].lower()}", step_metric="c_step/train/epoch")
        wandb.define_metric("probe/epoch/cls", step_metric="c_step/probe/epoch")
        wandb.define_metric("train/epoch/time", step_metric="c_step/train/epoch")
        wandb.define_metric("probe/epoch/time", step_metric="c_step/probe/epoch")
        if also_ftune:
            wandb.define_metric("ftune/epoch/time", step_metric="c_step/ftune/epoch")
            wandb.define_metric("ftune/epoch/cls", step_metric="c_step/ftune/epoch")
    else:
        wandb.define_metric("train/epoch/time", step_metric="c_step/ftune/epoch")
        wandb.define_metric("train/epoch/cls", step_metric="c_step/train/epoch")

    # Loss/Acc per validation epoch (mean over all epoch)
    if task != Task.CLS:
        wandb.define_metric("eval/epoch/probe/acc", step_metric="c_step/train/epoch")
        wandb.define_metric("eval/epoch/probe/acc_mv", step_metric="c_step/train/epoch")
        wandb.define_metric("eval/epoch/probe/cls", step_metric="c_step/train/epoch")
        if task == Task.MINE:
            generate_per_predictor_metrics("eval/epoch/probe", "c_step/train/epoch", modules, loss)
        if also_ftune:
            wandb.define_metric("eval/epoch/ftune/acc", step_metric="c_step/train/epoch")
            wandb.define_metric("eval/epoch/ftune/acc_mv", step_metric="c_step/train/epoch")
            wandb.define_metric("eval/epoch/ftune/cls", step_metric="c_step/train/epoch")
    else:
        wandb.define_metric("eval/epoch/cls", step_metric="c_step/train/epoch")
        wandb.define_metric("eval/epoch/acc", step_metric="c_step/train/epoch")
        wandb.define_metric("eval/epoch/acc_mv", step_metric="c_step/train/epoch")


predictors = ['up', 'down', 'side', 'var']


# Auxiliary function to wandb_logging. Logs individual losses when they depend on the modules
# num_batches indicates over how many batches should the loss be averaged
def log_separate_losses(losses, log_data, stage, base, num_batches=1, loss_name='', is_summary=True):
    total_loss = 0
    non_module_losses = ['acc', 'acc_mv', 'cls']
    if loss_name:
        base += loss_name + '/'
    for module, pred_losses in losses.items():
        # This function is also called for validation after probing, and these are reported independently
        if module in non_module_losses:
            continue
        m_total = 0
        # For each module, iterate over each predictor's losses
        for predictor, loss in pred_losses.items():
            if predictor != 'var':
                m_total += loss if is_summary else loss.item()
            # LOG PREDICTOR LOSS
            log_data[base + module + '/' + predictor] = (loss if is_summary else loss.item()) / num_batches

        # LOG TOTAL MODULE LOSS
        m_total /= len(pred_losses) - (1 if 'var' in pred_losses else 0)
        if len(pred_losses) > (2 if 'var' in pred_losses else 1):
            log_data[base + module + '/total'] = m_total / num_batches

        total_loss += m_total

    # LOG TOTAL NETWORK LOSS
    total_loss /= len(losses) - (len(non_module_losses) if stage == Stage.EVAL else 0)
    log_data[base + 'total'] = total_loss / num_batches


stage_names = {Stage.TRAIN: 'train', Stage.PROBE: 'probe', Stage.FTUNE: 'ftune', Stage.EVAL: 'eval'}


# stage:        [Stage] defines whether it is 'train', 'probe', 'ftune', or 'eval'
# epoch:        [int] current overall epoch
# batch:        [int] indicates batch_idx within a given epoch. if s_epoch > -1, indicates batch within s_epoch
# total_batch:  [int] total number of batches in an epoch
# losses:       [dict of floats] a dictionary of all current losses.
#               keys are organised by module_predictor. e.g. "space_u". Possible predictors: (u)p, (d)own
#               if probing/fine-tunning/evaluating, should also include "cls" to indicate supervised loss
#               if evaluating, should also include "acc" to indicate prediction accuracy
# summary:      [bool] indicates if reported losses are over batch or epoch summaries
# s_epoch:      [int] Optional. secondary epoch used only in probing/fine_tunning/validating
# num_sepochs:  [int] Optional. used to indicate how many probing/fine_tunning epochs are run each round
# val_mod:      [str] Optional. used to indicate whether validation round is within (probe)ing or (f)ine_(tune)ning
# val_rounds:   [int] Optional. used to indicate number of validation rounds (probing/fine_tunning) already ocurred
def wandb_logging(stage, task, epoch, batch, total_batch, losses, summary=False, loss_fcn='',
                  sepoch=-1, num_sepochs=0, val_mod=None, val_rounds=0, epoch_time=0):
    if DEBUG:
        return

    if val_mod is None:
        val_mod = ''

    loss_name = 'none'
    if loss_fcn:
        loss_name = str(loss_fcn).split(' ')[1].split('_')[0]

    log_data = dict()

    # LOG EPOCH
    log_data["c_step/train/epoch"] = epoch

    stage_name = stage_names[stage]

    if task != Task.CLS:
        if summary:
            base = stage_name + '/epoch/' + ((val_mod + '/') if val_mod else '')
            log_data[base + 'time'] = epoch_time
            if stage == Stage.TRAIN or (stage == Stage.EVAL and val_mod == 'probe'):
                if task == Task.MINE:
                    log_separate_losses(losses, log_data, stage, base, total_batch, loss_name=loss_name)
                else:
                    log_data[base + str(task).split('.')[1].lower()] = losses

            if stage == Stage.EVAL:
                log_data[base + 'cls'] = losses['cls']
                log_data[base + 'acc'] = losses['acc']
                log_data[base + 'acc_mv'] = losses['acc_mv']

            if stage == Stage.PROBE or stage == Stage.FTUNE:
                # LOG EPOCH
                log_data["c_step/" + stage_name + "/epoch"] = get_supv_epoch(val_rounds, sepoch, num_sepochs)

                log_data[base + 'cls'] = losses['cls'] / total_batch

        else:
            if stage == Stage.PROBE or stage == Stage.FTUNE:
                # LOG SECONDARY EPOCH
                log_data["c_step/" + stage_name + "/epoch"] = get_supv_epoch(val_rounds, sepoch, num_sepochs)
                # LOG BATCH
                log_data["c_step/" + stage_name + "/batch"] = get_supv_batch(val_rounds, sepoch, num_sepochs,
                                                                             total_batch, batch)
                # LOG LOSS
                log_data[stage_name + '/batch/cls'] = losses['cls']
            else:  # This should be 'train'
                # LOG BATCH
                log_data["c_step/" + stage_name + "/batch"] = get_ssl_batch(epoch, total_batch, batch)
                # LOG LOSSES
                base = 'train/batch/'
                if task == Task.MINE:
                    log_separate_losses(losses, log_data, stage, base, loss_name=loss_name, is_summary=False)
                else:
                    log_data[base + str(task).split('.')[1].lower()] = losses
    else:
        if summary:
            log_data[stage_name + '/epoch/cls'] = losses['cls'] / total_batch
            log_data[stage_name + '/epoch/time'] = epoch_time
            if stage == Stage.EVAL:
                log_data['eval/epoch/acc'] = losses['acc']
                log_data['eval/epoch/acc_mv'] = losses['acc_mv']
        else:
            log_data['c_step/train/batch'] = batch
            log_data['train/batch/cls'] = losses['cls'].item()

    wandb.log(log_data)

def wandb_log_lr(config, optimizer, epoch):
    if config['train']['task'] == 'cls' or config['train']['task'] == 'vicreg':
        wandb.log({'c_step/train/epoch': epoch,
                   'lr': optimizer.param_groups[0]['lr']}, commit=False)
    elif config['train']['use_single_optimizer']:
        if config['train']['predictor']['pred_aside'] == 0:
            wandb.log({'c_step/train/epoch': epoch,
                       'lr': optimizer['module'].param_groups[0]['lr']}, commit=False)
        elif config['train']['predictor']['pred_aside'] == 1:
            wandb.log({'c_step/train/epoch': epoch,
                       'lr': optimizer['module'].param_groups[0]['lr'],
                       'lr_p': optimizer['module'].param_groups[1]['lr']}, commit=False)
        else:
            wandb.log({'c_step/train/epoch': epoch,
                       'lr': optimizer['module'].param_groups[0]['lr'],
                       'lr_p': optimizer['predictor'].param_groups[0]['lr']}, commit=False)
    else:
        if config['train']['predictor']['pred_aside'] == 0:
            wandb.log({'c_step/train/epoch': epoch,
                       'lr': optimizer['space'].param_groups[0]['lr']}, commit=False)
        elif config['train']['predictor']['pred_aside'] == 1:
            wandb.log({'c_step/train/epoch': epoch,
                       'lr': optimizer['space'].param_groups[0]['lr'],
                       'lr_p': optimizer['space'].param_groups[1]['lr']}, commit=False)
        else:
            wandb.log({'c_step/train/epoch': epoch,
                       'lr': optimizer['space_mod'].param_groups[0]['lr'],
                       'lr_p': optimizer['space_pred'].param_groups[0]['lr']}, commit=False)


# def aux_ssl_loss(tr_epochs, epoch, num_batches, batch_idx):
#     actual_batch = get_ssl_batch(epoch, num_batches, batch_idx)
#     total_batches = tr_epochs * num_batches
#
#     return round(math.log(total_batches) - math.log(actual_batch + 1), 3)
#     # return round(math.log(total_batches) - math.log(actual_batch + 1) + (random.random() - 0.5), 3)
#
#
# def aux_sup_loss(epoch, interval, sepoch, num_sepochs, num_batches, batch_idx):
#     actual_batch = get_supv_batch(epoch, interval, sepoch, num_sepochs, num_batches, batch_idx)
#
#     return round(1 / (actual_batch + 1), 3)


# if __name__ == "__main__":
#     with open('wandb.key', 'r') as k:
#         wandb_key = k.read().strip()
#     config = dict()
#     wandb_run = wandb.init(project="WANDB_TEST")
#
#     modules = config['arch']['groups']
#
#     also_ftune = config['train']['val_config'] == 'ftune'
#
#     create_wandb_metrics(modules, also_ftune)
#
#     tr_epochs = 20
#
#     probe_epochs = 5
#     ftune_epochs = 7
#
#     probe_interval = 1
#     ftune_interval = 5
#
#     num_batches = 100
#
#     # MAIN LOOP
#     for tr_epoch in range(tr_epochs):
#         # TRAIN
#         acum = 0
#         for tr_b in range(num_batches):
#             loss = aux_ssl_loss(tr_epochs, tr_epoch, num_batches, tr_b)
#             acum += loss
#             wandb.log({"train/batch/total": loss,
#                        "c_step/train/batch": get_ssl_batch(tr_epoch, num_batches, tr_b),
#                        "c_step/train/epoch": tr_epoch})
#
#         wandb.log({"train/epoch/loss": acum / num_batches,
#                    "c_step/train/epoch": tr_epoch})
#
#         # VALIDATION
#         if tr_epoch % probe_interval == 0:
#             # PROBE
#             for pr_epoch in range(probe_epochs):
#
#                 current_epoch = get_supv_epoch(tr_epoch, probe_interval, pr_epoch, probe_epochs)
#                 acum = 0
#                 for val_b in range(num_batches):
#                     loss = aux_sup_loss(tr_epoch,
#                                         probe_interval,
#                                         pr_epoch,
#                                         probe_epochs,
#                                         num_batches,
#                                         val_b)
#                     acum += loss
#                     wandb.log({"probe/batch/cls": loss,
#                                "c_step/probe/batch": get_supv_batch(tr_epoch,
#                                                                     probe_interval,
#                                                                     pr_epoch,
#                                                                     probe_epochs,
#                                                                     num_batches,
#                                                                     val_b),
#                                "c_step/probe/epoch": current_epoch})
#
#                 wandb.log({"probe/epoch/cls": acum / num_batches,
#                            "c_step/probe/epoch": current_epoch,
#                            "c_step/train/epoch": tr_epoch})
#
#             # TEST
#             wandb.log({"val/probe_acc": tr_epoch/tr_epochs,
#                        "val/probe_loss": 1 + acum / num_batches,
#                        "c_step/train/epoch": tr_epoch})


'''
# tr_batch = (tr_epoch * num_batches) + batch_idx
# probe_batch = ((tr_epoch // probe_interval) * probe_epochs * num_batches) + batch_idx
# ftune_batch = ...

# MAIN TRAIN LOOP
best_p = float('inf')
best_f = float('inf')
for epoch in ssl_supervised_epochs:
    train_ssl(data_tr)                      --> report per batch (ssl) loss and epoch average  (tr_batch)
    
    # VALIDATION
    if epoch % probe_interval == 0:        
        model.save("/tmp/"+model_name)
        
        # PROBE
        for pr_epoch in probe_epochs:
            probe(data_tr,label_tr)          --> report per batch (sup) loss and epoch average
        l = test(data_val)                   --> report epoch (ssl/sup) average loss / acc
        
        # SAVE CHECKPOINT
        if l < best_p:
            model.save("/checkpoints/probing_"+model_name)
        
    # FINE-TUNE
    if epoch % ftune_interval == 0:
        model.load("/tmp/"+model_name)
        
        for ft_epoch in ftune_epochs:
            ftune(data_tr,label_tr)          --> report per batch (sup) loss and epoch average
        
        l = test_sup(data_val)               --> report epoch (ssl/sup) average loss / acc
        
        # SAVE CHECKPOINT
        if l < best_f:
            model.save("/checkpoints/ftunning_"+model_name)
            
    
    model.load("/tmp/"+model_name)

TEST LOOP
model.load()
test_sup(data_te)                        --> report epoch (sup) average loss / acc
        
    
'''