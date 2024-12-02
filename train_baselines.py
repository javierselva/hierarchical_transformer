from architecture.ht import Stage, Task, DataAugmentation
from utils import log_training
import time as tiempo
import torch
from wandb_log import wandb_logging, set_debug

# Epoch loops are broken during debug if number of batches reaches MAX_DEBUG_ITERS
MAX_DEBUG_ITERS = 150
DEBUG = False

def set_wandb_log_off(on):
    global DEBUG
    DEBUG = on
    set_debug(on)

# TODO setup the debug thing! Could be done more generally?

def train_vicreg(model, device, data_loader, optimizer, epoch, log_interval, scaler,
               datAug=(1.0,60,60,None), num_iters_grad_accum=1, use_mixed_prec=(False, "float16")):
    model.train(Stage.TRAIN) # TODO Check this!
    # Dictionary to store time elapsed for different parts of the training
    time_count = dict()
    # Total number of batches in a single epoch
    num_batches_per_epoch = len(data_loader)
    # Dictionary to accumulate the losses throughout an epoch
    epoch_loss = 0
    epoch_time = tiempo.time()

    # TODO datAug is always the same, could be created outside
    _, h, w, aug_params = datAug
    datAug = DataAugmentation(h, w, aug_params)

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

    optimizer.zero_grad()

    for batch_idx, (data1,data2, _) in enumerate(data_loader):
        if DEBUG:
            if iterat > MAX_DEBUG_ITERS:
                break
            iterat += 1

        if data1.shape[0] == 1:
            print(f"Holy cow! Only one sample in batch {batch_idx}/{num_batches_per_epoch}!")
            print(f"Importantly, the dataset had {len(data_loader.dataset)} samples!")
            continue

        s0 = tiempo.time()
        # No need for the targets in this setting
        data1 = data1.to(device)
        data2 = data2.to(device)

        data1 = datAug(data1)
        data2 = datAug(data2)

        #time_count['data'] = time_count.setdefault('data', 0) + (tiempo.time() - s0)
        time_count['data'] = (tiempo.time() - s0)

        # RUN THROUGH MODEL
        # TODO If problems observed, may need gradient clipping, then unscale gradients!!
        #   https://blog.paperspace.com/automatic-mixed-precision-using-pytorch/
        with torch.autocast(device_type='cuda', dtype=data_type, enabled=use_mixed_prec):
            # Compute output
            s = tiempo.time()
            # TODO expose variance to be able to log it!
            loss = model(data1, data2)
            time_count['forward'] = (tiempo.time() - s)
            # NaN SANITY CHECKS
            assert not torch.isnan(loss), "Loss is NaN!"

            loss /= num_iters_grad_accum
            epoch_loss += loss.item()

        s = tiempo.time()
        scaler.scale(loss).backward()
        if (batch_idx + 1) % num_iters_grad_accum == 0 or batch_idx == num_batches_per_epoch - 1:
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()

        time_count['backprop'] = (tiempo.time() - s)

        #time_count['total'] = time_count.setdefault('total', 0) + (tiempo.time() - s0)
        time_count['total'] = (tiempo.time() - s0)

        if batch_idx % log_interval == 0:
            log_training('mine', epoch=epoch, batch_idx=batch_idx, total_batch=num_batches_per_epoch,
                         loss=loss.item(), time_count=time_count)

            wandb_logging(Stage.TRAIN, Task.VICREG, epoch, batch_idx, num_batches_per_epoch, loss)
    epoch_time = tiempo.time() - epoch_time
    wandb_logging(Stage.TRAIN, Task.VICREG, epoch, batch_idx, num_batches_per_epoch, epoch_loss/num_batches_per_epoch,
                  summary=True, epoch_time=epoch_time)
