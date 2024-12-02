import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import no_grad, device, cuda
import wandb

from architecture.s3d import S3D
from architecture.ht_mine import DataAugmentation
from utils import create_classification_optimizer_and_scheduler, count_parameters, save_model_if_better
from wandb_log import wandb_logging, create_wandb_metrics
from architecture.ht_mine import Stage, Task
from data_utils.data_handler import UCF101

import sys
import os.path as op
import yaml
import traceback
from random import random

DEBUG = False

def train(model, data_loader, optim, log_interval, device=None, use_datAug=(0.0,120,160)):
    model.train()
    if device is None:
        device = model.device

    num_batches_per_epoch = len(data_loader)
    use_datAug, h, w = use_datAug

    if use_datAug:
        datAug = DataAugmentation(h,w)

    # Acumulate loss for epoch summary
    total_loss = 0

    if DEBUG:
        max_iter_for_debugging = 150
        iterat = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        if DEBUG:
            if iterat > max_iter_for_debugging:
                break
            iterat += 1

        data, label = data.to(device), label.to(device)

        if random() < use_datAug:
            data = datAug(data)

        optim.zero_grad()

        out = model(data.permute(0, 4, 1, 3, 2))

        loss = F.nll_loss(F.log_softmax(out, dim=-1), label)

        loss.backward()
        optim.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            print("TRAIN: Iteration", batch_idx, 'of epoch', epoch, 'loss:', round(loss.item(),3))
            if not DEBUG:
                wandb_logging(Stage.TRAIN, Task.CLS, epoch, batch_idx, num_batches_per_epoch, {'cls': loss})
    if not DEBUG:
        wandb_logging(Stage.TRAIN, Task.CLS, epoch, batch_idx, num_batches_per_epoch, {'cls': total_loss}, summary=True)

def validate(model, data_loader, epoch, device=None):
    model.eval()

    if device is None:
        device = model.device

    total_loss = 0
    correct = 0

    num_samples_in_dataset = len(data_loader)*data_loader.batch_size

    with no_grad():
        if DEBUG:
            max_iter_for_debugging = 150
            iterat = 0

        for batch_idx, (data, label) in enumerate(data_loader):
            if DEBUG:
                if iterat > max_iter_for_debugging:
                    break
                iterat += 1

            data, label = data.to(device), label.to(device)

            out = model(data.permute(0, 4, 1, 3, 2))

            total_loss += F.nll_loss(F.log_softmax(out, dim=-1), label, reduction='sum').item()
            pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    total_loss /= num_samples_in_dataset
    acc = 100. * correct / num_samples_in_dataset

    # In order to keep log_training from growing in optional parameters, I am overloading batch_idx for accuracy
    print('TEST: Acc.' + str(round(acc, 3)) + '%', 'loss', round(total_loss, 3))
    if not DEBUG:
        wandb_logging(Stage.EVAL, Task.CLS, epoch, -1, len(data_loader), {'cls': total_loss, 'acc': acc}, summary=True)
    return total_loss

if __name__ == '__main__':
    try:
        custom_config_file = sys.argv[1]
    except:
        print("No custom config file provided. Running with default.")
        custom_config_file = 's3d_config.yaml'

    gettrace = getattr(sys, 'gettrace', None)
    if (gettrace is not None and gettrace()) or DEBUG:
        DEBUG = True

    # ***************************************************************************
    # *************                   SETUP                   *******************
    # ***************************************************************************

    # Load config files
    with open(op.join('./config', custom_config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.loader.SafeLoader)

    run = False
    if not DEBUG:
        with open('wandb.key', 'r') as k:
            wandb_key = k.read().strip()
        try:
            wandb.login(key=wandb_key)
            run = wandb.init(project=config['train']['wb_project'], config=config)
            wandb.run.name = config['train']['model_name']
            print("Successfully connected with wandb")
        except Exception:
            print(traceback.format_exc())
            print("Failed to setup wandb!")

    num_epochs = config['train']['epochs']
    val_interval = config['train']['val_interval']
    log_interval = config['train']['log_interval']
    n_classes = config['data'][config['train']['dataset']]['num_classes']

    device = device("cuda" if cuda.is_available() else "cpu")

    # CREATE S3D MODEL
    # In order to pre-load weights, may need to initialise model with 400 output dim
    if config['train']['resume']:
        model = S3D(400, True, True)
        model.load_weights()
        model.reset_out_layer(n_classes)
    else:
        model = S3D(n_classes, True, True)

    model.to(device)
    print("Model ready with ", count_parameters(model), 'parameters')

    if run:
        wandb.watch(model, log='all', log_freq=config['train']['log_interval']*10)
        create_wandb_metrics([], False, Task.CLS, None, False)

    # TODO Keep only the stem?

    # PREPARE OPTIMIZER / SCHEDULER
    optimizer, scheduler, ins_sched_step = create_classification_optimizer_and_scheduler(config, model)

    # PREPARE DATA LOADER
    dataset_train = UCF101(config, 'train')
    dataset_test = UCF101(config, 'test')
    train_loader = DataLoader(dataset_train,
                              batch_size=config['train']['batch_size'],
                              shuffle=True,
                              num_workers=config['data']['dl_workers'],
                              persistent_workers=config['data']['dl_workers'] > 0)
    test_loader = DataLoader(dataset_test,
                             batch_size=config['train']['batch_size'],
                             shuffle=False,
                             num_workers=config['data']['dl_workers'],
                             persistent_workers=config['data']['dl_workers'] > 0)

    # MAIN LOOP
    best_loss = float('inf')
    if config['train']['val_init']:
        best_loss = validate(model, test_loader, 0, device)

    for epoch in range(1, num_epochs+1):
        train(model, train_loader, optimizer, log_interval, device)

        if epoch % val_interval == 0:
            l = validate(model, test_loader, epoch, device)
            if l < best_loss:
                best_loss = save_model_if_better(model,optimizer,scheduler,config,epoch,l,best_loss,'s3d')

        eval(ins_sched_step)
