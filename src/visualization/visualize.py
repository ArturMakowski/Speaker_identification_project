import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_embeddings(embeddings, targets, xlim=None, ylim=None, fig_save = True):
    
    uamp = umap.UMAP()
    embeddings_umap = uamp.fit_transform(embeddings)
    
    tsne = TSNE(init = "pca", learning_rate = 1000, perplexity = 30, n_iter = 5000)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10,10))
    for i in range(len(set(targets))):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings_umap[inds,0], embeddings_umap[inds,1], alpha=0.5) # type: ignore
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(set(targets))
    plt.title('UMAP projection of the test set', fontsize=24)   
    if fig_save: plt.savefig('../reports/figures/umap_test.png')
    
    plt.figure(figsize=(10,10))
    for i in range(len(set(targets))):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings_tsne[inds,0], embeddings_tsne[inds,1], alpha=0.5) # type: ignore
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(set(targets))
    plt.title('tSNE projection of the test set', fontsize=24)   
    if fig_save: plt.savefig('../reports/figures/tSNE_test.png')
    
    plt.show()

def extract_embeddings(dataloader, model, cuda):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 128))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels