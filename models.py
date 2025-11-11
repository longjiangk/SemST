import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_ratio=0.5):
        super().__init__()

        hidden_dim = max(1, int(input_dim * hidden_dim_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class ZINBDecoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(ZINBDecoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class SemST(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout, llm_dim, llm_modulation_ratio, use_llm=True):
        super(SemST, self).__init__()

        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = ZINBDecoder(nfeat, nhid1, nhid2)
        self.dropout = dropout

        self.use_llm = use_llm
        if use_llm:
            self.llm_head = LLMHead(llm_dim, nhid2)

            mod_mlp_input_dim = nhid2
            mod_mlp_output_dim = 4 * nhid2
            mod_mlp_hidden_dim = max(1, int(mod_mlp_input_dim * llm_modulation_ratio))
            self.modulation_mlp = nn.Sequential(
                nn.Linear(mod_mlp_input_dim, mod_mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mod_mlp_hidden_dim, mod_mlp_output_dim)
            )

        self.MLP = nn.Sequential(
            nn.Linear(nhid2 * 2, nhid1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid1, nhid2)
        )

    def forward(self, x, sadj, fadj, llm_emb):
        emb1 = self.SGCN(x, sadj)
        emb2 = self.FGCN(x, fadj)

        emb_combined = torch.cat((emb1, emb2), dim=1)

        if self.use_llm:
            processed_llm_emb = self.llm_head(llm_emb)
            modulation_params = self.modulation_mlp(processed_llm_emb)

            alpha, beta = torch.chunk(modulation_params, 2, dim=1)
            emb = (1 + alpha) * emb_combined + beta
        else:
            emb = emb_combined

        latent_emb = self.MLP(emb)
        [pi, disp, mean] = self.ZINB(latent_emb)

        return latent_emb, pi, disp, mean, emb1, emb2
