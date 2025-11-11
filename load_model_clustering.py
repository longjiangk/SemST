from __future__ import division
from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
import scanpy as sc
from models import SemST
from preprocess.config import Config
from utils import *


def load_data(dataset, data_path_prefix="DLPFC/"):  # Added data_path_prefix argument
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, default='Mouse_Brain_Anterior')
    parser.add_argument('--llm_emb_dir', type=str, default='data/npys/',
                        help='Directory for LLM embeddings.')
    parser.add_argument('--weights_dir', type=str, default=r'data/weights/',
                        help='Directory for model weights.')

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

    # Determine path prefixes based on config_name
    # Assuming "DLPFC" uses a prefix and others don't for data/results
    data_path_prefix_str = config_name + "/" if config_name in ["DLPFC", "Embryo"] else ""
    result_path_prefix_str = config_name + "/" if config_name in ["DLPFC", "Embryo"] else ""

    print(f"Processing dataset: {dataset_id}")

    config_file_path = f'./config/{config_name}.ini'
    llm_emb_file_path = os.path.join(args.llm_emb_dir, f'embeddings_{dataset_id}.npy')

    adata, features, labels, fadj, sadj, graph_nei, graph_neg = load_data(dataset_id,
                                                                          data_path_prefix=data_path_prefix_str)
    print(adata)

    config = Config(config_file_path)
    cuda_enabled = not config.no_cuda and torch.cuda.is_available()
    # use_seed = not config.no_seed # This was not used, config.seed is used directly

    _, ground_truth_labels = np.unique(np.array(labels, dtype=str), return_inverse=True)
    ground_truth_labels = torch.LongTensor(ground_truth_labels)
    config.n = len(ground_truth_labels)  # Make sure this is set based on actual data
    config.class_num = len(ground_truth_labels.unique())

    if cuda_enabled:
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        graph_nei = graph_nei.cuda()
        graph_neg = graph_neg.cuda()

    # Seeding (ensure this is done correctly and consistently)
    import random

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    if cuda_enabled:
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Benchmark should be False for reproducibility if True

    print(f"{dataset_id} LR: {config.lr}, Alpha: {config.alpha}, Beta: {config.beta}, Gamma: {config.gamma}")

    precomputed_llm_base_embeddings, llm_base_emb_dim = run_precomputation_llm_base(llm_emb_file_path)

    if cuda_enabled:
        precomputed_llm_base_embeddings = precomputed_llm_base_embeddings.cuda()

    model_instance = SemST(nfeat=config.fdim,
                           nhid1=config.nhid1,
                           nhid2=config.nhid2,
                           dropout=config.dropout,
                           llm_dim=llm_base_emb_dim,
                           llm_modulation_ratio=config.llm_modulation_ratio,
                           use_llm=True)
    if cuda_enabled:
        model_instance.cuda()

    model_instance.load_state_dict(
        torch.load(os.path.join(args.weights_dir, rf'model_{dataset_id}.pth'))
    )
    with torch.no_grad():
        model_instance.eval()
        emb, mean, _, _, _, _ = model_instance(features, sadj, fadj, precomputed_llm_base_embeddings)

    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values

    kmeans = KMeans(n_clusters=config.class_num, n_init='auto', random_state=100).fit(emb)
    idx = kmeans.labels_

    ari_res = metrics.adjusted_rand_score(labels, idx) * 100
    nmi_res = metrics.normalized_mutual_info_score(labels, idx) * 100
    ami_res = metrics.adjusted_mutual_info_score(labels, idx) * 100

    if config_name == 'DLPFC':
        _, labels = np.unique(labels, return_inverse=True)
    idx_aligned = hungarian_match(labels, idx)
    acc_res = metrics.accuracy_score(labels, idx_aligned) * 100
    f1_res = metrics.f1_score(labels, idx_aligned, average='weighted') * 100
    f1_res_m = metrics.f1_score(labels, idx_aligned, average='macro') * 100

    print('ari:', ari_res)
    print('nmi:', nmi_res)
    print('acc:', acc_res)
    print('ami:', ami_res)
    print('f1_w:', f1_res)
    print('f1_m:', f1_res_m)

    adata.obs['idx'] = idx.astype(str)
    plt.rcParams["figure.figsize"] = (4, 4)  # 3, 4

    # adata.obsm["spatial"][:, 1] *= -1
    title_str = (f'SemST: ARI={ari_res:.2f}\n'
                 f'NMI={nmi_res:.2f}, ACC={acc_res:.2f}\n'
                 f'F1_m={f1_res_m:.2f}, F1_w={f1_res:.2f}')
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
    plt.tight_layout()
    plt.show()
