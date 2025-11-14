import os

import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def generate_llm_emb(llm_model_name, all_gene_texts, batch_size, device_for_precomputation,
                     prompt_system, prompt_user, max_length=256, npy_path=None):
    if not os.path.exists(npy_path):
        print(f"Starting LLM base embedding precomputation on device: {device_for_precomputation}")

        tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        llm_base = AutoModel.from_pretrained(llm_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for param in llm_base.parameters():  # Freeze LLM base
            param.requires_grad = False
        llm_base.eval().to(device_for_precomputation)

        llm_hidden_dim = llm_base.config.hidden_size
        all_base_embs_list = []

        for i in tqdm(range(0, len(all_gene_texts), batch_size), desc="Precomputing LLM base embeddings"):
            batch_texts = all_gene_texts[i: i + batch_size]
            prompts = [
                f"<|im_start|>system\n{prompt_system}<|im_end|>\n<|im_start|>user\n{prompt_user}{text}<|im_end|>\n<|im_start|>assistant\n"
                for text in batch_texts
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(
                device_for_precomputation)

            with torch.no_grad():
                outputs = llm_base(**inputs, output_hidden_states=False)

            last_hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size())
            masked_hidden_states = last_hidden_states * attention_mask
            summed_hidden_states = masked_hidden_states.sum(dim=1)
            num_non_padding_tokens = attention_mask.sum(dim=1)
            num_non_padding_tokens = torch.clamp(num_non_padding_tokens, min=1e-9)
            pooled_output = summed_hidden_states / num_non_padding_tokens

            all_base_embs_list.append(pooled_output.cpu())

        del llm_base, tokenizer  # Free memory
        if torch.cuda.is_available() and device_for_precomputation.type == 'cuda':
            torch.cuda.empty_cache()

        final_base_embeddings = torch.cat(all_base_embs_list, dim=0)
        print(
            f"Finished LLM base embedding precomputation. "
            f"Shape: {final_base_embeddings.shape}, "
            f"Dimension: {llm_hidden_dim}, "
            f"path: {npy_path}"
        )
        np.save(npy_path, final_base_embeddings)


if __name__ == '__main__':

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_TOP_GENES_LLM = 10
    LLM_MAX_LENGTH = 2560
    LLM_PRECOMPUTATION_BATCH_SIZE = 8

    LLM_MODEL_NAME = "Qwen3-4B"
    datasets = ['Mouse_Brain_Anterior']
    # datasets = ['Human_Breast_Cancer']
    for dataset_name in datasets:

        NPY_PATH = rf'data/npys/embeddings_{dataset_name}.npy'
        if not os.path.exists(os.path.dirname(NPY_PATH)):
            os.makedirs(os.path.dirname(NPY_PATH))

        if dataset_name in ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672']:
            path = f"generate_data/DLPFC/{dataset_name}/data.h5ad"
        elif dataset_name == 'E1S1':
            path = f"generate_data/Embryo/{dataset_name}/data.h5ad"
        else:
            path = f"generate_data/{dataset_name}/data.h5ad"
        adata = sc.read_h5ad(path)

        gene_expression_raw = adata.X.toarray() if isinstance(adata.X, np.ndarray) is False else adata.X
        top_n_gene_indices = np.argsort(gene_expression_raw, axis=1)[:, -N_TOP_GENES_LLM:]
        cell_gene_texts = []
        gene_names_array = adata.var_names.to_numpy()
        for i in range(adata.n_obs):
            top_genes_for_cell = gene_names_array[top_n_gene_indices[i]]
            cell_gene_texts.append(", ".join(reversed(top_genes_for_cell)))

        prompt_system = (
            'You are an expert in bioinformatics. Represent the biological state of a cell characterized '
            'by the following highly expressed genes. Focus on capturing the functional essence relevant '
            'for spatial domain identification. Output a dense vector representation.')

        prompt_user = 'Highly expressed genes: '

        generate_llm_emb(
            llm_model_name=LLM_MODEL_NAME,
            all_gene_texts=cell_gene_texts,
            batch_size=LLM_PRECOMPUTATION_BATCH_SIZE,
            device_for_precomputation=DEVICE,
            prompt_system=prompt_system,
            prompt_user=prompt_user,
            max_length=LLM_MAX_LENGTH,
            npy_path=NPY_PATH
        )
