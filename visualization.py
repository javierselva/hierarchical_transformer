from os.path import join as pjoin
import time as tiempo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import sys
import wandb
import traceback
from numpy import frombuffer

from data_utils.precomputed_features import PreComputedFeatures
from architecture.ht_mine import HierarchicalTransformer
from utils import read_n_setup_params

from torch import load
from torch import device as tdevice
import torch.nn.functional as F
#from tsnecuda import TSNE
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
import umap


def load_model(file_name,config,path='/output'):
    model = HierarchicalTransformer(config)
    try:
        model.load_state_dict(load(pjoin(path,file_name)))
    except:
        print("Could not load model "+pjoin(path,file_name)+". Running with randomly initialised net.")
    return model

# Go through whole dataset once
# Only returns labels if return_features is true
def dry_run(model,data_loader,device,return_features=False,return_labels=False):
    model.eval()
    time_count = dict()
    if return_features:
        # Create a new data handler to store output features
        new_data_handler = PreComputedFeatures(data_loader.get_num_batches(), data_loader.get_batch_size())

    test_loss = 0
    correct = 0
    for batch_idx in range(data_loader.get_num_batches()):
        s0 = tiempo.time()
        data, target = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        # time_count['data'] = time_count.setdefault('data', 0) + (tiempo.time() - s0)

        # s = tiempo.time()
        if model.linear_probe:
            output, features = model(data)
        else:
            # TODO not sure this will work
            features = model(data)
        if return_features:
            new_data_handler.add_batch(features.detach(), target.detach())
        # time_count['forward'] = time_count.setdefault('forward', 0) + (tiempo.time() - s)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        time_count['total'] = time_count.setdefault('total', 0) + (tiempo.time() - s0)

    test_loss /= data_loader.get_size()
    acc = 100. * correct / data_loader.get_size()

    if return_features:
        return test_loss, acc, new_data_handler.get_all(get_labels=return_labels)

    else:
        return test_loss, acc

def generate_plot(data,labels,title):
    # Create the figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, title=title)
    canvas = FigureCanvas(fig)

    # Create the scatter
    ax.scatter(
        x=data[:, 0],
        y=data[:, 1],
        c=labels,
        cmap=plt.cm.get_cmap('Paired'),
        alpha=0.4,
        s=0.5)
    # plt.savefig(pjoin(config['train']['save_path'],config['eval']['model_name']+'.png'))

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()

    return frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

def sweep_tsne(reduced_features,targets,model_name,dims):
    for p in range(10,50,10):
        for n in range(32,1024,256):
            for l in range(10,1000,200):
                for e in range(3,50,10):
                    try:
                        tsne = TSNE(perplexity=p, n_iter=10000, verbose=1, num_neighbors=n, learning_rate=l, early_exaggeration=e)
                        # Features shape should be #samples x #dims
                        tsne_results = tsne.fit_transform(reduced_features)
                        print("TSNE FEATURES"+'_pca'+str(dims)+'_perp'+str(p)+'_neigh'+str(n)+'_lr'+str(l)+'_ee'+str(e),tsne_results.shape,flush=True)
                    except Exception:
                        print(traceback.format_exc())
                        print("Something went wrong with TSNE")
                        continue

                    name = "TSNE " + model_name + '_pca' + str(dims) + '_perp' + str(p) + '_neigh' + str(
                        n) + '_lr' + str(l) + '_ee' + str(e)
                    img = generate_plot(tsne_results, targets, name)
                    image = wandb.Image(img, caption=name)
                    wandb.log({"tsne_final": image})

def sweep_umap(reduced_features, targets, model_name,dims):
    for n in (2, 5, 10, 20, 50, 100, 200):
        for d in (0.0, 0.1, 0.25, 0.5, 0.8, 0.99):
            reducer = umap.UMAP()
            scaled_data = StandardScaler().fit_transform(reduced_features)
            umap_results = reducer.fit_transform(scaled_data)

            name = 'UMAP ' + model_name + '_pca' + str(dims) + '_neigh' + str(n) + '_minDist' + str(d)
            img = generate_plot(umap_results, targets, name)
            image = wandb.Image(img, caption=name)

            wandb.log({"tsne_final": image})

if __name__ == '__main__':
    print("Welcome to a TSNE evaluation")
    # ***************************************************************************
    # *************                   SETUP                   *******************
    # ***************************************************************************
    try:
        custom_config_file = sys.argv[1]
    except:
        print("No custom config file provided. Running with default.")
        custom_config_file = None

    # Load config files
    config = read_n_setup_params(custom_config_file)
    print("Config file loaded!")

    device = tdevice("cuda")
    model = load_model(config['eval']['model_name']+'.pt',config).to(device)
    print("Model ready")

    # set linear probe so final output features are returned
    # TODO this method does not exist anymore
    model.set_linear_probe(True,device,keep_mlp=(config['train']['task'] == 'cls'))

    data_loader = LoaderUcf101(config, mode='test')
    print("Data loader ready")

    # START WANDB
    with open('wandb.key', 'r') as k:
        wandb_key = k.read().strip()
    try:
        wandb.login(key=wandb_key)
        run = wandb.init(project=config['eval']['wb_project'], config=config)
        wandb.run.name = config['train']['model_name'] + '_tsnePlot'
        print("Successfully connected with wandb")
    except Exception:
        print(traceback.format_exc())
        print("Failed to setup wandb!")

    # ***************************************************************************
    # *************          PERFORM DRY RUN                  *******************
    # ***************************************************************************

    loss,acc,(features,targets) = dry_run(model,data_loader,device,True,True)
    print("Features extracted",flush=True)
    print("Num Features",data_loader.get_size(),flush=True)
    print("loss",loss,"acc",acc,flush=True)
    wandb.log({'test_loss':loss,'acc':acc})
    # labels = dict()
    # for target in targets:
    #     labels[target.item()] = labels.get(target.item(),0) + 1
    # print(labels)
    # sys.exit()
    # ***************************************************************************
    # *************                    TSNE                   *******************
    # ***************************************************************************
    print("FEATURES",features.shape,flush=True)
    # Try just PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features.cpu())
    targets = targets.cpu()
    print("PCA FEATURES",reduced_features.shape,flush=True)
    name = "PCA " + config['train']['model_name']
    img = generate_plot(reduced_features,targets,name)
    image = wandb.Image(img, caption=name)

    wandb.log({"pca_final": image,
               '_pca': 2})

    for dims in range(128, 7, -20):
        if dims != 128:
            pca = PCA(n_components=dims)
            reduced_features = pca.fit_transform(features.cpu())
        else:
            reduced_features = features.cpu()

        if config['eval']['dim_red_method'] == 'umap':
            sweep_umap(reduced_features,targets,config['train']['model_name'],dims)
        elif config['eval']['dim_red_method'] == 'tsne':
            sweep_tsne(reduced_features, targets, config['train']['model_name'],dims)

    run.finish()