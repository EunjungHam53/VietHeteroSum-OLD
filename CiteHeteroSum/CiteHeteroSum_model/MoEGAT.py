import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# MatrixRouter với top_k cố định
class FixedMatrixRouter(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Ma trận W để nhân với input
        self.W = nn.Parameter(torch.randn(input_dim, num_experts) * 0.1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, dim]
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # [batch*seq_len, dim]
        
        # Nhân ma trận W
        gate_scores = torch.matmul(x_flat, self.W)
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        # Top-k selection cố định
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Renormalize
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Create routing mask
        routing_mask = torch.zeros_like(gate_probs).scatter_(1, top_k_indices, 1.0)
        
        return top_k_weights, top_k_indices, routing_mask.view(batch_size, seq_len, self.num_experts)


# MatrixRouter với top_k động
class DynamicMatrixRouter(nn.Module):
    def __init__(self, input_dim, num_experts, max_top_k=3):
        super().__init__()
        self.num_experts = num_experts
        self.max_top_k = max_top_k
        
        # Ma trận W để nhân với input
        self.W = nn.Parameter(torch.randn(input_dim, num_experts) * 0.1)
        
        # Network để quyết định top_k cho mỗi token
        self.top_k_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, dim]
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # [batch*seq_len, dim]
        
        # Dự đoán top_k động cho mỗi token: từ 1 đến max_top_k
        top_k_scores = self.top_k_predictor(x_flat)  # [batch*seq_len, 1]
        dynamic_top_k = torch.clamp(
            torch.round(top_k_scores * self.max_top_k) + 1, 
            min=1, 
            max=self.max_top_k
        ).long().squeeze(-1)  # [batch*seq_len]
        
        # Nhân ma trận W
        gate_scores = torch.matmul(x_flat, self.W)
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        # Top-k selection với k động
        # Để xử lý k khác nhau, ta lấy max_top_k rồi mask
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.max_top_k, dim=-1)
        
        # Tạo mask dựa trên dynamic_top_k
        k_mask = torch.arange(self.max_top_k, device=x.device).unsqueeze(0) < dynamic_top_k.unsqueeze(1)
        top_k_probs = top_k_probs * k_mask.float()
        
        # Renormalize
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Create routing mask
        routing_mask = torch.zeros_like(gate_probs).scatter_(1, top_k_indices, 1.0)
        routing_mask = routing_mask * (top_k_weights > 0).float().scatter_(1, top_k_indices, 1.0)
        
        return top_k_weights, top_k_indices, routing_mask.view(batch_size, seq_len, self.num_experts), dynamic_top_k


# MoE Expert - deputy expert
class DeputyExpert(nn.Module):
    def __init__(self, in_features, n_hidden, dropout, n_heads):
        super().__init__()
        self.gat = GAT(in_features=in_features, n_hidden=n_hidden, 
                      n_classes=in_features, dropout=dropout, n_heads=n_heads)
        
    def forward(self, x, adj):
        return self.gat(x, adj)


# Main GAT
class MainGAT(nn.Module):
    def __init__(self, in_features, n_hidden, dropout, n_heads):
        super().__init__()
        self.gat = GAT(in_features=in_features, n_hidden=n_hidden, 
                      n_classes=in_features, dropout=dropout, n_heads=n_heads)
        
    def forward(self, x, adj):
        return self.gat(x, adj)


# MoE Layer với Main GAT + Deputy Experts
class MoEGraphLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout_p=0.3, n_heads=6, num_experts=3, 
                 top_k=2, use_dynamic_topk=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_dynamic_topk = use_dynamic_topk
        
        # Main GAT xử lý toàn bộ adj
        self.main_gat = MainGAT(in_dim, hid_dim, dropout_p, n_heads)
        
        # Router: chọn Fixed hoặc Dynamic
        if use_dynamic_topk:
            self.router = DynamicMatrixRouter(in_dim, num_experts, max_top_k=top_k)
        else:
            self.router = FixedMatrixRouter(in_dim, num_experts, top_k=top_k)
        
        # 3 deputy experts: sentence, section, document
        self.deputy_experts = nn.ModuleList([
            DeputyExpert(in_dim, hid_dim, dropout_p, n_heads) for _ in range(num_experts)
        ])
        
        # Gating mechanism để blend main và deputy outputs
        self.blend_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        
        # Contribution loss coefficient
        self.contribution_loss_coef = 0.01
        self.target_main_contribution = args['target_main_contribution']  # Target: main đóng góp 50%
        
    def forward(self, feature, adj, doc_num, sect_num):
        batch_size, seq_len, dim = feature.shape
        
        # Main GAT xử lý toàn bộ graph
        main_output = self.main_gat(feature, adj)
        
        # Tạo adjacency matrices cho từng deputy expert
        sent_adj = adj.clone()
        sent_adj[:, :, -sect_num - doc_num:] = 0
        sect_adj = adj.clone()
        sect_adj[:, :, :-sect_num - doc_num] = sect_adj[:, :, -doc_num:] = 0
        doc_adj = adj.clone()
        doc_adj[:, :, :-doc_num] = 0
        
        adj_list = [sent_adj, sect_adj, doc_adj]
        
        # Get routing decisions
        if self.use_dynamic_topk:
            expert_weights, expert_indices, routing_mask, dynamic_top_k = self.router(feature)
        else:
            expert_weights, expert_indices, routing_mask = self.router(feature)
            dynamic_top_k = None
        
        # Initialize deputy output
        deputy_output = torch.zeros_like(feature)
        
        # Process through selected deputy experts
        for i in range(self.num_experts):
            # Mask cho deputy expert i
            expert_mask = routing_mask[:, :, i].unsqueeze(-1)  # [batch, seq_len, 1]
            
            if expert_mask.sum() > 0:
                # Apply deputy expert
                expert_input = feature * expert_mask
                expert_out = self.deputy_experts[i](expert_input, adj_list[i])
                
                # Weight by routing decision
                weighted_output = expert_out * expert_mask
                deputy_output += weighted_output
        
        # Blend main và deputy outputs
        blend_weight = self.blend_gate(feature)  # [batch, seq_len, dim]
        final_output = blend_weight * main_output + (1 - blend_weight) * deputy_output
        
        # Contribution loss - khuyến khích main và deputy đóng góp cân bằng
        main_contribution = blend_weight.mean()
        contribution_loss = torch.abs(main_contribution - self.target_main_contribution) * self.contribution_loss_coef
        
        return final_output, contribution_loss, main_contribution, dynamic_top_k
    
    def get_expert_stats(self, routing_mask):
        """Helper function để theo dõi việc sử dụng experts"""
        expert_usage = routing_mask.mean(dim=[0, 1])  # [num_experts]
        return expert_usage


class StepWiseGraphConvLayerMoE(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, dropout_p=0.3, act=nn.LeakyReLU(), 
                 nheads=6, iter=1, have_gate=False, num_experts=3, top_k=2, 
                 use_dynamic_topk=False):
        super().__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout_p)
        self.iter = iter
        self.use_dynamic_topk = use_dynamic_topk
        
        # MoE layers với Main GAT + Deputy Experts
        self.moe_layers = nn.ModuleList([
            MoEGraphLayer(in_dim, hid_dim, dropout_p, nheads, num_experts, 
                         top_k, use_dynamic_topk) 
            for _ in range(iter)
        ])
        
        # Giữ nguyên các layer khác
        self.ffn = MLP(in_dim, in_dim, hid_dim, dropout_p=dropout_p, layers=3)
        self.have_gate = have_gate
        if self.have_gate:
            self.gate = Gate(in_dim)
        
        # Store losses và stats
        self.contribution_losses = []
        self.main_contributions = []
        self.dynamic_topk_values = []
        
    def forward(self, feature, adj, docnum, secnum):
        feature_resi = feature
        self.contribution_losses = []
        self.main_contributions = []
        self.dynamic_topk_values = []
        
        # Apply MoE layers với residual connections
        for i in range(self.iter):
            feature_new, contrib_loss, main_contrib, dyn_topk = self.moe_layers[i](
                feature, adj, docnum, secnum
            )
            feature = feature_new + feature  # Residual connection
            self.contribution_losses.append(contrib_loss)
            self.main_contributions.append(main_contrib)
            if dyn_topk is not None:
                self.dynamic_topk_values.append(dyn_topk)
        
        # Apply FFN layers
        feature = self.ffn(feature) + feature_resi
        if self.have_gate:
            feature = self.gate(feature)
        
        return feature
    
    def get_contribution_loss(self):
        """Lấy tổng contribution loss"""
        return sum(self.contribution_losses) if self.contribution_losses else 0.0
    
    def get_main_contribution_ratio(self):
        """Lấy tỉ lệ đóng góp trung bình của main GAT"""
        if self.main_contributions:
            return sum(self.main_contributions) / len(self.main_contributions)
        return 0.0
    
    def get_topk_statistics(self):
        """Lấy thống kê về top-k (chỉ dùng cho dynamic mode)"""
        if not self.use_dynamic_topk or not self.dynamic_topk_values:
            return None
        
        all_topk = torch.cat([tk.float() for tk in self.dynamic_topk_values])
        return {
            'mean': all_topk.mean().item(),
            'std': all_topk.std().item(),
            'min': all_topk.min().item(),
            'max': all_topk.max().item()
        }


# ===== EXAMPLE =====
"""
# 1. Fixed Top-K (cố định)
model_fixed = StepWiseGraphConvLayerMoE(
    in_dim=768, 
    out_dim=768, 
    hid_dim=256,
    dropout_p=0.3,
    nheads=6,
    iter=2,
    have_gate=True,
    num_experts=3,
    top_k=2,                    # Top-k cố định = 2
    use_dynamic_topk=False      # Sử dụng Fixed Router
)

# 2. Dynamic Top-K (động)
model_dynamic = StepWiseGraphConvLayerMoE(
    in_dim=768, 
    out_dim=768, 
    hid_dim=256,
    dropout_p=0.3,
    nheads=6,
    iter=2,
    have_gate=True,
    num_experts=3,
    top_k=3,                    # Max top-k = 3 (model sẽ chọn 1-3)
    use_dynamic_topk=True       # Sử dụng Dynamic Router
)
"""