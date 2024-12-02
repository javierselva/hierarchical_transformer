from torch.utils.data import DataLoader, Subset
import torch

from utils import *
from architecture.ht import HierarchicalTransformer
from architecture.ht_mine import HT_Mine, Task, Stage, norms
from architecture.ht_vicreg import VICReg
from losses import mse_loss, ce_loss, cs_loss, vicreg_loss
from train import train_mine, train_cls, fine_tune, linear_probe, test
from train_baselines import train_vicreg

import sys
import wandb_log
from random import randint
import traceback

run = None
DEBUG = False

def main():
    global run
    global DEBUG
    try:
        custom_config_file = sys.argv[1]
    except:
        print("No custom config file provided. Running with default.")
        custom_config_file = None

    # ***************************************************************************
    # *************                   SETUP                   *******************
    # ***************************************************************************

    # Load config files
    config = read_n_setup_params(custom_config_file)
    print('Configuration loaded from ' + (custom_config_file if custom_config_file else 'default.'))
    print("Running model with base name " + config['train']['model_name'])

    if config['train']['dataset'] == 'imagenet':
        from data_utils.imagenet_handler import CustomDataset
    else:
        if config['data']['data_loader'] == 'hdf5':
            from data_utils.data_handler import CustomDataset
        elif config['data']['data_loader'] == 'video':
            from data_utils.data_handler_decord import CustomDataset
        elif config['data']['data_loader'] == 'frames':
            if config['train']['task'] == 'vicreg':
                from data_utils.data_handler_frames_2seq import CustomDataset
            else:
                from data_utils.data_handler_frames import CustomDataset
        else:
            raise NotImplementedError(f"Data loader {config['data']['data_loader']} not implemented")

    task = Task.str2task(config['train']['task'])
    running_baseline = task != Task.MINE and task != Task.CLS

    gettrace = getattr(sys, 'gettrace', None)
    # TODO find a better way to do this
    if (gettrace is not None and gettrace()) or DEBUG:
        DEBUG = True
        set_debug(True)
        from train import set_wandb_log_off as set_debug_main
        from eval import set_wand_log_off as set_debug_eval
        if running_baseline:
            from train_baselines import set_wandb_log_off as set_debug_baselines
            set_debug_baselines(True)
        set_debug_main(True)
        set_debug_eval(True)

    if not DEBUG:
        with open('wandb.key','r') as k:
            wandb_key = k.read().strip()
        try:
            wandb.login(key=wandb_key)
            run = wandb.init(project=config['train']['wb_project'], config=config)
            wandb.run.name = config['train']['model_name']
            print("Successfully connected with wandb")
        except Exception:
            print(traceback.format_exc())
            print("Failed to setup wandb!")

    torch.manual_seed(config['train']['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on " + str(device))

    # ***************************************************************************
    # *************                 DATA LOADING              *******************
    # ***************************************************************************

    # Create data_loader
    if config['train']['dataset'] in ['ucf101', 'ssv2', 'k400']:
        if config['train']['dataset'] == 'ucf101':
            eval_on = 'test'
        else:
            eval_on = 'val'
        dataset_train = CustomDataset(config, 'train', dataset=config['train']['dataset'])
        dataset_test = CustomDataset(config, eval_on, dataset=config['train']['dataset'])

    elif config['train']['dataset'] == 'imagenet':
        groups = eval(config['arch']['groups'])
        assert len(groups) == 2 and 'backbone' in groups and 'space' in groups, \
             "When using 'imagenet' dataset, only 2 groups are allowed ('backbone' and 'space')"

        dataset_train = CustomDataset(config, 'train')
        if config['data']['imagenet']['sampler']['step'] > 1:
            dataset_train_bak = dataset_train
            dataset_train = Subset(dataset_train,
                                   range(0,
                                         len(dataset_train),
                                         config['data']['imagenet']['sampler']['step']))
        dataset_test = CustomDataset(config, 'val')

    else:
        raise Exception('Selected dataset '+config['train']['dataset']+' not found')

    train_loader = DataLoader(dataset_train,
                              batch_size=config['train']['batch_size'],
                              shuffle=True,
                              num_workers=config['data']['dl_workers'],
                              persistent_workers=config['data']['dl_workers'] > 0)
    test_loader = DataLoader(dataset_test,
                             batch_size=config['train']['test_batch_size'] if config['train'][
                                                                                  'test_batch_size'] > 0 else
                             config['train']['batch_size'],
                             shuffle=False,
                             num_workers=config['data']['dl_workers'],
                             persistent_workers=config['data']['dl_workers'] > 0)

    print('Data loader for '+config['train']['dataset']+' ready')

    # DATA AUGMENTATION
    if running_baseline and config['data']['data_augmentation']['use_datAug'] != 1.0:
        print(f"WARINING: Data augmentation is necessary for baselines (currently {task}) "
              f"but was set to {config['data']['data_augmentation']['use_datAug']}. "
              f"As it is required, it will be set to 1.")
        config['data']['data_augmentation']['use_datAug'] = 1.0

    # Select re-normalization mean and std of a specific dataset (used right after data augmentation)
    select_normalization_dataset(config, norms)

    data_augmentation = (config['data']['data_augmentation']['use_datAug'],
                         config['data'][config['train']['dataset']]['sampler']['height'],
                         config['data'][config['train']['dataset']]['sampler']['width'],
                         {'augs': config['data']['data_augmentation']['augs'],
                          'normalization': config['data']['data_augmentation']['normalization']})

    # ***************************************************************************
    # *************                 MODEL LOADING             *******************
    # ***************************************************************************

    # Create model
    if task == Task.MINE:
        model = HT_Mine(config)
    elif task == Task.CLS:
        model = HierarchicalTransformer(config)
    elif task == Task.VICREG:
        model = VICReg(config)
    model.to(device)

    # Attach the model to wandb
    if run:
        wandb.watch(model, log='all', log_freq=config['train']['log_interval']*10)
        wandb_log.create_wandb_metrics(eval(config['arch']['groups']),
                                       config['train']['ftune_interval'],
                                       task,
                                       config['train']['loss_fn'].lower() if task != Task.CLS else None,
                                       config['train']['predictor']['pred_aside'] and task != Task.CLS)

    print("Initialised model with {} trainable parameters".format(count_parameters(model, True)))

    # ***************************************************************************
    # *************            OPTIMIZERS / SCHEDULERS        *******************
    # ***************************************************************************
    groups = model.get_groups()

    # CLASSIFICATION OBJECTIVE
    if task == Task.CLS:
        optimizer, scheduler, instr_update_sch = create_classification_optimizer_and_scheduler(config, model)
        main_scaler = torch.cuda.amp.GradScaler(enabled=config['train']['use_mixed_precision'])
    # SELF-PREDICTION
    elif task == Task.MINE:
        optimizer, scheduler, instr_update_sch = select_right_optimizer(config, groups, model)
        # Build one scaler per module
        main_scaler = dict()
        if 'module' in optimizer:
            main_scaler['module'] = torch.cuda.amp.GradScaler(enabled=config['train']['use_mixed_precision'])
        else:
            for group in groups:
                main_scaler[group] = torch.cuda.amp.GradScaler(enabled=config['train']['use_mixed_precision'])
    # OTHER BASELINES
    elif task == Task.VICREG or task == Task.SVT:
        optimizer, scheduler, instr_update_sch = select_right_optimizer(config, groups, model)
        main_scaler = torch.cuda.amp.GradScaler(enabled=config['train']['use_mixed_precision'])
    else:
        raise (Exception, "Selected task '" + task + "' does not exist!")

    num_iters_grad_accum = config['train']['num_iters_grad_accum']
    use_mixed_precision = (config['train']['use_mixed_precision'], config['train']['type_for_mixed_prec'])

    print("Optimizer and Schedulers ready.")
    print("Selected task:", task)
    random_value = str(randint(0, 1000000))
    print("Assigned random number to model: "+random_value)

    # ***************************************************************************
    # *************             RESUME TRAINING               *******************
    # ***************************************************************************
    # If resuming from checkpoint, load model, optimizer and scheduler state
    # TODO this won't work for baselines, must pre-load model.backbone instead
    if config['train']['resume_path']:
        print("Loading model from checkpoint")
        # TODO also load optimizer / epoch information for proper resuming
        if config['train']['resume_all']:
            l = load_model(model, config, checkpoint=True, optim=optimizer, sched=scheduler)
        else:
            l = float('inf')
            load_model(model, config, checkpoint=True, load_optim_sched=False,
                       strict=not config['train']['use_curriculum'])
    else:
        l = float('inf')

    model.to(device)

    # ***************************************************************************
    # *************             LOSS SELECTION                *******************
    # ***************************************************************************

    # Setup loss function
    loss_fcn = None
    loss_params = dict()
    if task == Task.MINE:
        if config['train']['loss_fn'].upper() == 'CE':
            loss_fcn = ce_loss
            loss_params = config['train']['loss_params']['CE']
            assert loss_params['tgt_temp'] != 0.0, "Cross Entropy selected but temperature is set to 0"
            assert loss_params['pred_temp'] != 0.0, "Cross Entropy selected but temperature is set to 0"
        elif config['train']['loss_fn'].upper() == 'CS':
            loss_fcn = cs_loss
            loss_params = config['train']['loss_params']['CS']
        elif config['train']['loss_fn'].upper() == 'MSE':
            loss_fcn = mse_loss
            # MSE has no furhter parameters!
            #loss_params = config['train']['loss_params']['MSE']
        elif config['train']['loss_fn'].upper() == 'VICREG':
            loss_fcn = vicreg_loss
            loss_params = config['train']['loss_params']['VICREG']
        else:
            raise Exception("Selected loss " + config['train']['loss_fn'] + " is not implemented.")
    if loss_fcn is not None:
        print(f"Loss function {loss_fcn} with params {loss_params} selected.")

    # ***************************************************************************
    # *************         INITIAL VALIDATION ROUND          *******************
    # ***************************************************************************

    is_val_interval_type_linear = config['train']['val_interval_type'] == 'linear'
    if not is_val_interval_type_linear and task == Task.CLS:
        config['train']['val_interval'] = round(config['train']['val_interval'])
        config['train']['val_interval_type'] = 'linear'
        is_val_interval_type_linear = True
        print("WARNING: exponential validation schedule type is not implemented for supervised runs. Using linear"
              f"with interval {config['train']['val_interval']}")
    val_round_count = 0
    prev_error_val = float('inf')

    # Exponential interval for validation won't work if interval is 1.5 and no initial validation round is run
    if not is_val_interval_type_linear and config['train']['val_interval'] == 1.5:
        val_round_count = 1

    # ***************************************************************************
    # *************            MAIN TRAINING LOOP             *******************
    # ***************************************************************************
    print("Starting training loop")
    best_epoch = 0  # Used for curriculum learning to load best epoch from previous step!
    for epoch in range(1, config['train']['epochs'] + 1):
        if not DEBUG and run:
            wandb_log.wandb_log_lr(config, optimizer, epoch)
        
        # TRAIN
        if task == Task.CLS:
            train_cls(model, device, train_loader, optimizer, epoch, Stage.TRAIN,
                      config['train']['log_interval'], main_scaler, use_datAug=data_augmentation,
                      num_iters_grad_accum=num_iters_grad_accum,
                      use_mixed_prec=use_mixed_precision)
        elif task == Task.MINE:
            train_mine(model, device, train_loader, optimizer, epoch,
                       config['train']['log_interval'],
                       main_scaler,
                       loss_fcn,
                       config['train']['loss_weights'],
                       loss_params,
                       use_datAug=data_augmentation,
                       num_iters_grad_accum=num_iters_grad_accum,
                       use_mixed_prec=use_mixed_precision)
        elif task == Task.VICREG:
            train_vicreg(model, device, train_loader, optimizer,
                         epoch, config['train']['log_interval'], main_scaler,
                         datAug=data_augmentation,
                         num_iters_grad_accum=num_iters_grad_accum,
                         use_mixed_prec=use_mixed_precision)
        else:
            raise(Exception, "Selected task '"+task+"' has no associated train function!!")
        
        # VALIDATION
        # If task is SSL do KNN
        if task != Task.CLS:
            evalt = False
            is_linear_val_interval_turn = (config['train']['val_interval'] and
                                           (epoch % config['train']['val_interval'] == 0))
            is_first_or_last_epoch = epoch == 1 or epoch == config['train']['epochs']
            is_expon_val_interval_turn = (config['train']['val_interval'] and
                                          (int(config['train']['val_interval']**val_round_count) == epoch or
                                                is_first_or_last_epoch))

            # KNN VALIDATION ROUND
            if ((not is_val_interval_type_linear) and is_expon_val_interval_turn) or \
               (is_val_interval_type_linear and is_linear_val_interval_turn):
                print(f"Doing KNN validation round {val_round_count}")
                if running_baseline:
                    aux_model = model.backbone
                else:
                    aux_model = model

                # TODO compute ssl losses on eval set!
                # Extract train and test features
                train_features, train_labels, test_features, test_labels = extract_features(aux_model,
                                                                                            device,
                                                                                            train_loader,
                                                                                            test_loader)
                l = knn_eval(train_features, train_labels, test_features, test_labels)

                # Update best and save checkpoint
                if config['train']['save_model']:
                    # TODO update this, knn returns acc (so acc > best)
                    # TODO what dataloader should I use? all features?
                    prev_error_val, did_save = save_model_if_better(aux_model, optimizer, scheduler, config, epoch, l,
                                                            prev_error_val, extra=random_value, mod='probe')
                    if did_save:
                        best_epoch = epoch

                val_round_count += 1



        elif task == Task.CLS:
            print("Doing validation round")
            if epoch % config['train']['val_interval'] == 0:
                # Simply test
                l = test(model, device, test_loader, epoch, loss_params, task)
                if config['train']['save_model']:
                    prev_error_val, did_save = save_model_if_better(model, optimizer, scheduler, config, epoch, l,
                                                            prev_error_val, extra=random_value, mod='cls')
                    if did_save:
                        best_epoch = epoch

        else:
            print("Warning, no validation function for selected task ", task)

        # SCHEDULER STEP
        if (config['train']['use_single_optimizer'] and
                config['train']['predictor']['pred_aside'] != 2) or \
                task == Task.CLS or running_baseline:
            eval(instr_update_sch)
        else:
            for group in scheduler.keys():
                eval(instr_update_sch)

        if config['train']['interpolate_loss_weights_to_uniformity']:
            step_loss_weights(config['train']['loss_weights'], epoch, config['train']['epochs'] + 1)

        # TODO could do something similar with video data loader so it shuffles every epoch when max_seq_per_video is > 0
        #   or simply set persistent_workers to False!!
        if config['train']['dataset'] == 'imagenet' and config['data']['imagenet']['sampler']['step'] > 1:
            del dataset_train
            dataset_train = Subset(dataset_train_bak,
                                   range(epoch % config['data']['imagenet']['sampler']['step'],
                                         len(dataset_train_bak),
                                         config['data']['imagenet']['sampler']['step']))
            del train_loader
            train_loader = DataLoader(dataset_train,
                                      batch_size=config['train']['batch_size'],
                                      shuffle=True,
                                      num_workers=config['data']['dl_workers'],
                                      persistent_workers=config['data']['dl_workers'] > 0)
        elif config['train']['dataset'] != 'imagenet' and \
            config['data']['dl_workers'] > 0 and \
            config['data']['max_seq_per_video'] > 0:
            del train_loader
            # Create again so internal shuffle is redone for all workers
            train_loader = DataLoader(dataset_train,
                                      batch_size=config['train']['batch_size'],
                                      shuffle=True,
                                      num_workers=config['data']['dl_workers'],
                                      persistent_workers=config['data']['dl_workers'] > 0)
        elif config['train']['dataset'] != 'imagenet' and \
            config['data']['dl_workers'] == 0 and \
            config['data']['max_seq_per_video'] > 0:
            # Shuffle train loader
            train_loader.dataset.compute_mapping_and_offset()


    if not DEBUG and run:
        run.finish()

    # ***************************************************************************
    # ********* CREATE NEW CONFIG FILE FOR CURRICULUM LEARNING  *****************
    # ***************************************************************************
    if config['train']['use_curriculum']:
        # Increment curriculum step
        config['train']['curriculum_step'] += 1

        # Update architecture, training and other data config parameters
        update_config_dicts(config, config['train']['curriculum']['step_' + str(config['train']['curriculum_step'])])

        # Save model info to be loaded back
        config['train']['resume_path'] = config['train']['model_name'] + random_value
        config['train']['resume_epoch'] = best_epoch
        # Curriculum has to be done with probing (fine-tunning will save fine-tuned model)
        config['train']['resume_mod'] = 'cls' if task == Task.CLS else 'probe'

        # Add which step of the curriculum this is to the end of the name
        if '--' in config['train']['model_name']:
            base_name = config['train']['model_name'].split('--')[0]
        else:
            base_name = config['train']['model_name']
        config['train']['model_name'] = base_name + '--CLp-' + \
                                        str(config['train']['curriculum_step'])

        if custom_config_file is None:
            print("ERROR: No custom config file was provided. Curriculum learning will now fail.")
            print("\t What follows are the necessary data to load the proper model")
            print("\t\tresume_path: ", config['train']['resume_path'])
            print("\t\tresume_epoch: ", config['train']['resume_epoch'])
            print("\t\tresume_mod: ", config['train']['resume_mod'])
            print("\t\tcurriculum_step: ", config['train']['curriculum_step'])
        else:
            if '--' in custom_config_file:
                base_name = custom_config_file.split('--')[0]
            else:
                base_name = custom_config_file.split('.')[0]

            new_config_file_name = './config/' + base_name + \
                                   '--CLp-' + str(config['train']['curriculum_step']) + '.yaml'
            save_config_file(config, new_config_file_name)
            print("Saved new config for curriculum learning in ", new_config_file_name)

    eval_scaler = torch.cuda.amp.GradScaler(enabled=config['train']['use_mixed_precision'])
    if config['train']['test_type'] == 'probe' and config['train']['probe_epochs']:
        # DO PROBING
        linear_probe(config, aux_model, device, train_loader, test_loader,
                     epoch, eval_scaler, loss_fcn, loss_params,
                     random_value, probe_round_count, use_datAug=data_augmentation,
                     num_iters_grad_accum=num_iters_grad_accum,
                     use_mixed_prec=use_mixed_precission)

    elif config['train']['test_type'] == 'ftune' and config['train']['ftune_epochs']:
        # DO FINE-TUNNING
        l = fine_tune(config, aux_model, device, train_loader, test_loader,
                      epoch, eval_scaler, loss_params,
                      ftune_round_count, use_datAug=data_augmentation,
                      num_iters_grad_accum=num_iters_grad_accum,
                      use_mixed_prec=use_mixed_precision)

if __name__ == '__main__':
    main()
