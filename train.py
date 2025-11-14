from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import warnings

import pandas as pd
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from tqdm import tqdm
import scanpy as sc

from models import SemST
from preprocess.config import Config
from utils import *

warnings.filterwarnings("ignore")


def load_data(dataset, data_path_prefix="DLPFC/"): 
    print("load data:")

    base_path = "./generate_data/"
    if data_path_prefix:
        path = os.path.join(base_path, data_path_prefix, dataset, "data.h5ad")
    else:
        path = os.path.join(base_path, dataset, "data.h5ad")

    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg


def train(model, features, sadj, fadj, precomputed_llm_base_embeddings, graph_nei, graph_neg, config,
          optimizer):  
    model.train()
    optimizer.zero_grad()
    emb, pi, disp, mean, emb1, emb2 = model(features, sadj, fadj, precomputed_llm_base_embeddings)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0.05).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)

    dcir_loss = dicr_loss(emb1, emb2)

    total_loss = config.alpha * zinb_loss + config.gamma * reg_loss + config.beta * dcir_loss
    total_loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        emb, _, _, mean, _, _ = model(features, sadj, fadj, precomputed_llm_base_embeddings)

    return emb, mean, zinb_loss, reg_loss, dcir_loss, total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, default='Mouse_Brain_Anterior')
    parser.add_argument('--use_llm', action='store_true', default=True, help='Whether to use LLM embeddings.')
    parser.add_argument('--llm_emb_dir', type=str, default='data/npys/', help='Directory for LLM embeddings.')

    args = parser.parse_args()

    dataset_id = args.dataset_id

    # dataset_id = '151672'
    # dataset_id = 'Human_Breast_Cancer'
    # dataset_id = 'Mouse_Brain_Anterior'
    # dataset_id = 'MVC'
    # dataset_id = 'E1S1'

    if dataset_id in ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672']:
        config_name = 'DLPFC'
    elif dataset_id in ['E1S1']:
        config_name = 'Embryo'
    else:
        config_name = dataset_id

    args.use_llm = True

    # Determine path prefixes based on config_name
    # Assuming "DLPFC, Embryo" uses a prefix and others don't for data/results
    data_path_prefix_str = config_name + "/" if config_name in ["DLPFC", "Embryo"] else ""
    result_path_prefix_str = config_name + "/" if config_name in ["DLPFC", "Embryo"] else ""

    print(f"Processing dataset: {dataset_id}")

    config_file_path = f'./config/{config_name}.ini'
    config = Config(config_file_path)

    llm_emb_file_path = os.path.join(args.llm_emb_dir, f'embeddings_{dataset_id}.npy')

    adata, features, labels, fadj, sadj, graph_nei, graph_neg = load_data(dataset_id,
                                                                          data_path_prefix=data_path_prefix_str)

    plt.rcParams["figure.figsize"] = (3, 4)
    current_savepath = f'./result/{result_path_prefix_str}{dataset_id}'
    if not os.path.exists(current_savepath):
        os.makedirs(current_savepath)

    cuda_enabled = not config.no_cuda and torch.cuda.is_available()

    _, ground_truth_labels = np.unique(np.array(labels, dtype=str), return_inverse=True)
    ground_truth_labels = torch.LongTensor(ground_truth_labels)
    config.n = len(ground_truth_labels) 
    config.class_num = len(ground_truth_labels.unique())
    
    print('seed:', config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    if cuda_enabled:
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    print(f"{dataset_id} LR: {config.lr}, Alpha: {config.alpha}, Beta: {config.beta}, Gamma: {config.gamma}")

    if cuda_enabled:
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        graph_nei = graph_nei.cuda()
        graph_neg = graph_neg.cuda()

    precomputed_llm_base_embeddings, llm_base_emb_dim = run_precomputation_llm_base(llm_emb_file_path)
    if cuda_enabled:
        precomputed_llm_base_embeddings = precomputed_llm_base_embeddings.cuda()

    model_instance = SemST(nfeat=config.fdim,
                           nhid1=config.nhid1,
                           nhid2=config.nhid2,
                           dropout=config.dropout,
                           llm_dim=llm_base_emb_dim,
                           llm_modulation_ratio=config.llm_modulation_ratio,
                           use_llm=args.use_llm)
    if cuda_enabled:
        model_instance.cuda()

    optimizer_instance = optim.Adam(model_instance.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    current_dataset_best_ari = 0.0
    current_dataset_best_nmi = 0.0
    current_dataset_best_acc = 0.0
    current_dataset_best_f1_w = 0.0
    current_dataset_best_f1_m = 0.0
    best_idx_for_dataset = []
    best_mean_for_dataset = []
    best_emb_for_dataset = []

    for epoch in tqdm(range(config.epochs)):
        emb, mean, zinb_loss, reg_loss, dcir_loss, total_loss = train(
            model_instance, features, sadj, fadj, precomputed_llm_base_embeddings,
            graph_nei, graph_neg, config, optimizer_instance
        )
        print(f"{dataset_id} epoch: {epoch}, zinb_loss = {zinb_loss:.2f}"
              f", reg_loss = {reg_loss:.2f}, dcir_loss= {dcir_loss:.2f}, total_loss = {total_loss:.2f}")

        emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
        mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values

        kmeans = KMeans(n_clusters=config.class_num, n_init='auto', random_state=config.seed).fit(emb)
        idx = kmeans.labels_

        ari_res = metrics.adjusted_rand_score(labels, idx) * 100
        nmi_res = metrics.normalized_mutual_info_score(labels, idx) * 100
        ami_res = metrics.adjusted_mutual_info_score(labels, idx) * 100

        if config_name == 'DLPFC':
            _, labels = np.unique(labels, return_inverse=True)
        idx_aligned = hungarian_match(labels, idx)
        acc_res = metrics.accuracy_score(labels, idx_aligned) * 100
        f1_res_w = metrics.f1_score(labels, idx_aligned, average='weighted') * 100
        f1_res_m = metrics.f1_score(labels, idx_aligned, average='macro') * 100

        if ari_res > current_dataset_best_ari:
            current_dataset_best_ari = ari_res
            current_dataset_best_nmi = nmi_res
            current_dataset_best_acc = acc_res
            current_dataset_best_f1_w = f1_res_w
            current_dataset_best_f1_m = f1_res_m

            best_idx_for_dataset = idx
            best_mean_for_dataset = mean
            best_emb_for_dataset = emb

    print(f"Best ARI for {dataset_id}: {current_dataset_best_ari}")

    title_str = (f'SemST: ARI={current_dataset_best_ari:.2f}\n'
                 f'NMI={current_dataset_best_nmi:.2f}, ACC={current_dataset_best_acc:.2f}\n'
                 f'F1_m={current_dataset_best_f1_m:.2f}, F1_w={current_dataset_best_f1_w:.2f}')

    adata.obs['idx'] = best_idx_for_dataset.astype(str)
    adata.obsm['emb'] = best_emb_for_dataset
    adata.obsm['mean'] = best_mean_for_dataset

    if config_name in ['DLPFC', 'Human_Breast_Cancer', 'Mouse_Brain_Anterior']:
        sc.pl.spatial(adata,
                      img_key='hires',
                      color=['idx'],
                      title=title_str,
                      show=False
                      )
    else:
        sc.pl.embedding(adata,
                        basis="spatial",
                        color="idx",
                        s=25,
                        show=False,
                        title=title_str
                        )
    plot_filename = f'SemST' if args.use_llm else 'SemST_wo_llm'
    plt.savefig(os.path.join(current_savepath, f'{plot_filename}.png'),
                bbox_inches='tight',
                dpi=300)

    plt.show()
