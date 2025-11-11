from __future__ import division
from __future__ import print_function

from construction import features_construct_graph, spatial_construct_graph
import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from config import Config


def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)

    return adata


def load_ST_file(path, highly_genes, k, k1):
    labels_path = os.path.join(path, "metadata.tsv")

    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["ground_truth"].copy()

    ground = labels
    _, ground = np.unique(ground, return_inverse=True)

    cell_labels = labels.copy()
    for j in range(len(cell_labels)):
        cell_labels[j] = cell_labels[j][0]
    cell_labels = cell_labels.replace('D', '1')
    cell_labels = cell_labels.replace('H', '0')
    cell_labels = cell_labels.replace('I', '1')
    cell_labels = cell_labels.replace('T', '1')

    adata = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()

    adata.obs['ground_truth'] = labels.values
    adata.obs['ground'] = ground
    # adata.obs['annot_type'] = cell_labels.values.astype(int)
    adata.var_names_make_unique()

    adata.X = np.array(sp.csr_matrix(adata.X, dtype=np.float32).todense())
    print(adata)
    adata = normalize(adata, highly_genes=highly_genes)

    fadj = features_construct_graph(adata.X, k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph(adata.obsm['spatial'], k=k1)

    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    return adata


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['Mouse_Brain_Anterior']
    for i in range(len(datasets)):
        dataset = datasets[i]
        print(dataset)
        raw_data_dir = rf"data/raw/{dataset}"
        savepath = f"generate_data/{dataset}/"
        config_file = f'config/{dataset}.ini'
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        config = Config(config_file)
        adata = load_ST_file(raw_data_dir, config.fdim, config.k, config.radius)
        print("saving")
        adata.write(savepath + 'data.h5ad')
        print("done")
