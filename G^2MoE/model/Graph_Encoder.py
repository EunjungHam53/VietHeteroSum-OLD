import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, layers=2, act=nn.LeakyReLU(), dropout_p=0.3, keep_last_layer=False):
        super(MLP, self).__init__()
        self.layers = layers
        self.act = act
        self.dropout = nn.Dropout(dropout_p)
        self.keep_last = keep_last_layer

        self.mlp_layers = nn.ModuleList([])
        if layers == 1:
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.mlp_layers.append(nn.Linear(in_dim, hid_dim))
            for i in range(self.layers - 2):
                self.mlp_layers.append(nn.Linear(hid_dim, hid_dim))
            self.mlp_layers.append(nn.Linear(hid_dim, out_dim))

    def forward(self, x):
        for i in range(len(self.mlp_layers) - 1):
            x = self.dropout(self.act(self.mlp_layers[i](x)))
        if self.keep_last:
            x = self.mlp_layers[-1](x)
        else:
            x = self.act(self.mlp_layers[-1](x))
        return x


# borrowed from labml.ai
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.

        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))
        # del g_concat, g_repeat, g_repeat_interleave
        # torch.cuda.empty_cache()
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        # $e_{ij}$ is set to $- \infty$ if there is no edge from $i$ to $j$.
        e = e.masked_fill(adj_mat == 0, float(-1e9))
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=1)


class GAT(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        super().__init__()
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()
        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        x = x.squeeze(0)
        adj_mat = adj_mat.squeeze(0)
        adj_x = adj_mat.clone().sum(dim=1, keepdim=True).repeat(1, x.shape[1]).bool()
        adj_mat = adj_mat.unsqueeze(-1).bool()
        x = self.dropout(x)
        x = self.layer1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x, adj_mat).masked_fill(adj_x == 0, float(0))
        return x.unsqueeze(0)
    
class NodeScorer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
    def forward(self, x):
        return torch.tanh(self.linear(x))

class Gate(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.scorer = NodeScorer(in_features)
    def forward(self, x):
        scores = self.scorer(x).squeeze(-1)
        x = x * scores.view(-1,1)
        return x

class Contrast_Encoder(nn.Module):
    def __init__(self, graph_encoder, hidden_dim, bert_hidden=768, in_dim=768, dropout_p=0.3):
        super(Contrast_Encoder, self).__init__()
        self.graph_encoder = graph_encoder
        self.common_proj_mlp = MLP(in_dim, in_dim, hidden_dim, dropout_p=dropout_p, act=nn.LeakyReLU())

    def forward(self, p_gfeature, p_adj, docnum, secnum):
        pg = self.graph_encoder(p_gfeature.float(), p_adj.float(), docnum, secnum)
        pg = self.common_proj_mlp(pg)
        return pg


class End2End_Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_p):
        super(End2End_Encoder, self).__init__()
        # self.graph_encoder = graph_encoder
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj_layer_mlp = MLP(in_dim, in_dim, hidden_dim, act=nn.LeakyReLU(), dropout_p=dropout_p, layers=2)
        self.final_layer = nn.Linear(in_dim, 1)

    def forward(self, x, adj, docnum, secnum):
        # x = self.graph_encoder(x.float(), adj.float(), docnum, secnum)
        x = x[:, :-docnum-secnum-1, :]
        x = self.out_proj_layer_mlp(x)
        x = self.final_layer(x)
        return x