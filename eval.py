import time as tiempo
from losses import compute_losses
from architecture.ht import Task, Stage
from utils import log_training, update_acum_dict
from wandb_log import wandb_logging, set_debug

import torch
import torch.nn.functional as F

# Epoch loops are broken during debug if number of batches reaches MAX_DEBUG_ITERS
MAX_DEBUG_ITERS = 150
DEBUG = False

def set_wandb_log_off(on):
    global DEBUG
    DEBUG = on
    set_debug(on)

# Loss_fcn is only used if evaluating after probing, so SSL losses can also be computed.
# In those lines, temp will be used if loss_fcn is ce_loss
def test(model, device, data_loader, epoch, loss_params, task=Task.MINE, loss_fcn=None):
    model.eval()
    test_losses = dict()
    losses = dict()
    correct = 0
    # Total number of batches in a single epoch
    num_batches_per_epoch = len(data_loader)
    num_samples_in_dataset = len(data_loader)*data_loader.batch_size
    test_time = tiempo.time()

    multiview_out = dict()
    multiview_label = dict()
    multiview_count = dict()

    # TODO Inference mode may be better!!! (Faster!)
    # https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc
    with torch.no_grad():
        if DEBUG:
            iterat = 0

        for batch_idx, (data, label, idx) in enumerate(data_loader):
            if DEBUG:
                if iterat > MAX_DEBUG_ITERS:
                    break
                iterat += 1

            data, label = data.to(device), label.to(device)
            if model.probe_epoch >= 0 and task == Task.MINE:
                # If we're evaluating after probing, also take intermediate features
                # to see how the SSL loss behaves in ODD data
                token_idxs = model.set_random_prediction_pos()
                output, targets, targets_agg, predictions = model(data)
                losses = compute_losses(model, targets_agg, predictions, loss_fcn,
                                        model.predicts_side, loss_params, targets, token_idxs)
            else:
                output = model(data)

            # ce_loss = CrossEntropyLoss()  # si que te softmax
            # ce_fn = F.cross_entropy()  # does not applies softmax within it
            losses['cls'] = F.nll_loss(output, label, reduction='sum')  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

            update_acum_dict(test_losses, losses)

            # MULTIVIEW ACCUMULATING!
            # TODO find a better sampling for testing! like 10 temporal x 3 spatial crops or smth
            for b in range(output.shape[0]):
                i = idx[b].item()
                out = output[b]
                lab = label[b]
                multiview_out[i] = multiview_out.get(i, torch.zeros(out.shape, device=output.device)) + out
                multiview_count[i] = multiview_count.get(i, 0) + 1
                if multiview_label.get(i, None) is None:
                    multiview_label[i] = lab
                else:
                    assert lab.eq(multiview_label[i]).item(), "Something happend, same sequence {} " \
                                                          "got two labels {} and {}".format(
                                                             i, lab.item(), multiview_label[i].item())

    # MULTIVIEW COMPUTING!
    correct_multiview = 0
    for idx in multiview_out:
        multiview_out[idx] /= multiview_count[idx]
        pred = multiview_out[idx].argmax(keepdim=True)  # get the index of the max log-probability
        correct_multiview += pred.eq(multiview_label[idx]).item()

    test_losses['cls'] /= num_samples_in_dataset
    test_losses['acc'] = 100. * correct / num_samples_in_dataset
    test_losses['acc_mv'] = 100. * correct_multiview / len(multiview_out)

    # In order to keep log_training from growing in optional parameters, I am overloading batch_idx for accuracy
    log_training('test', total_batch=num_samples_in_dataset, batch_idx=correct,
                 loss=test_losses['cls'], epoch=epoch)
    wandb_logging(Stage.EVAL, task, epoch, -1, num_batches_per_epoch, test_losses,
                  summary=True, loss_fcn=loss_fcn,
                  val_mod='probe' if model.probe_epoch >= 0 else 'ftune',
                  epoch_time=tiempo.time() - test_time)

    return test_losses['cls']

@torch.no_grad()
def get_features(model, data_loader):
    model.eval()
    features = list()
    labels = list()
    for d,l in data_loader:
        # TODO will i get oom?
        labels.append(l.cpu())
        features.append(model(d).cpu())
    return torch.stack(features), torch.stack(labels)


def extract_features(config, model, data_loader_train, data_loader_test, device):
    if model is None or data_loader_train is None or data_loader_test is None:
        # TODO Create model / data_loaders
        raise NotImplementedError("Feature extraction outside of main training loop not yet implemented")

    # TODO maybe use an hdf5 if they don't fit in memory
    # new_dataset = PreComputedFeatures(len(data_loader) * data_loader.batch_size, feature_dim=model.final_output_dim,
    #                                   save_path=save_path, save_file=save_file)

    # Extract features
    train_features, train_labels = get_features(model, data_loader_train)
    test_features, test_labels = get_features(model, data_loader_test)
    return train_features, train_labels, test_features, test_labels


# Function adapted from https://github.com/kahnchana/svt/blob/master/eval_knn.py
@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5
