"""
Paper: Self-supervised Graph Learning for Recommendation
Author: Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie
Reference: https://github.com/wujcan/SGL-Torch
"""

__author__ = "Jiancan Wu"
__email__ = "wujcan@gmail.com"

__all__ = ["SGL"]

import torch
from torch.serialization import save
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import get_initializer
from util.common import Reduction
from data import PointwiseSamplerV2, PairwiseSamplerV2
import numpy as np
from time import time
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix, ensureDir
from util.pytorch import sp_mat_to_sp_tensor
from reckit import randint_choice
import os


def group_aware_infonce_full(logits: torch.Tensor,
                             anchor_user_ids: torch.Tensor,
                             user_group_tensor: torch.Tensor,
                             base_tau_vec: torch.Tensor,
                             kappa: float = 2.0,
                             reduction: str = "sum") -> torch.Tensor:
    """
    參數：
      logits: [B, num_users]
      anchor_user_ids: [B]，該 batch 的全域 user id（bat_users）
      user_group_tensor: [num_users]，0: Group B, 1: Group A
      base_tau_vec: [B]，每個 anchor 的基礎溫度（可全為 self.ssl_temp）
      kappa: 放大倍數（>1）
    """
    device = logits.device
    B, N = logits.size()

    # 基礎溫度展成 [B, N]，原本每個 anchor i 有一個 base_tau_vec[i]，現在展開成整個矩陣 tau[i, j]，初始時每一格都是同一個 τ（例如 0.2）。
    tau = base_tau_vec.unsqueeze(1).expand(B, N)  # [B, N]

    # 列（anchor）的群組、欄（所有 user）的群組
    anchor_groups = user_group_tensor[anchor_user_ids].to(device)  # [B]，第 i 個 anchor user 是 A(1) 還是 B(0)

    all_groups = user_group_tensor.to(device)                      # [N]，column j 的全域 user 是 A(1) 還是 B(0)

    # 只在 B-anchor × A-negative 的位置放大溫度
    mask_anchor_B = (anchor_groups == 0).unsqueeze(1).expand(B, N)  # anchor i ∈ Group B，找出哪些列是 B？

    mask_col_A = (all_groups == 1).unsqueeze(0).expand(B, N)        # candidate user j ∈ Group A，找出哪些欄是 A？

    mask_B_to_A = mask_anchor_B & mask_col_A                        # anchor 在 B，且 negative 在 A。

    if kappa != 1.0:
        tau = torch.where(mask_B_to_A, tau * kappa, tau) #只對 B→A 的 logit 放大溫度。

    scaled_logits = logits / tau

    # 依照每格 tau 縮放 logits；正樣本 label 是全域 user id
    labels = anchor_user_ids.to(device)
    return F.cross_entropy(scaled_logits, labels, reduction=reduction)


# === Group Contrastive Loss ===
def group_contrastive_loss(
    user_embs1: torch.Tensor,
    user_embs2: torch.Tensor,
    user_ids: torch.Tensor,
    user_group_tensor: torch.Tensor,
    margin: float = 0.5,
    neg_weight_B_anchor: float = 0.0,
    neg_weight_A_anchor: float = 1.0,
) -> torch.Tensor:
    """
    非對稱群組式對比損失：
    - 仍強化同群的 hardest positive（A–A、B–B）
    - 只有在 A 當 anchor 時，用權重 neg_weight_A_anchor 推遠 A–B
    - 在 B 當 anchor 時，對 A 的負樣本給權重 neg_weight_B_anchor（建議 0 或很小）
    參數：
        neg_weight_B_anchor: B 當 anchor 時，針對跨群負樣本的權重（0~1）
        neg_weight_A_anchor: A 當 anchor 時，針對跨群負樣本的權重（常用 1.0）
    """
    device = user_embs1.device
    groups = user_group_tensor.to(user_ids.device)[user_ids]  # [B]

    # 算 batch 內兩兩 user 的相似度矩陣，例如第 i 列就是以第 i 個 user 當 anchor，跟所有 user（另一 view）的相似度。
    sim_matrix = F.cosine_similarity(
        user_embs1.unsqueeze(1), user_embs2.unsqueeze(0), dim=2
    )  # [B, B]
    B = user_embs1.size(0)

    # 逐 anchor 蒐集 hardest pos/neg
    pos_list = []
    neg_list = []
    neg_w_list = []

    idxs = torch.arange(B, device=device)
    for i in range(B):
        same_mask = (groups == groups[i]) & (idxs != i)   # 同群（不含自己）
        diff_mask = (groups != groups[i])                 # 異群

        pos_sims = sim_matrix[i][same_mask] # 把這個 anchor i 對所有「同群人 j」的相似度撈出來
        if pos_sims.numel() > 0:
            hardest_pos = pos_sims.min()  # 拉近最難的正樣本，把這個變成 anchor i 的代表正樣本 → 讓整個同群都要至少像它那麼近
        else:
            # 無同群樣本時，可設為 0（或跳過）；這裡設成 0 以使 loss 可微
            hardest_pos = torch.tensor(0.0, device=device)

        neg_sims = sim_matrix[i][diff_mask]
        if neg_sims.numel() > 0:
            hardest_neg = neg_sims.max()  # 推遠最難的負樣本，把這個當作代表負樣本 → 讓所有異群至少比它還遠一些
        else:
            hardest_neg = torch.tensor(0.0, device=device)

        # 設定對稱性：A/B anchor 對跨群負樣本的權重
        if groups[i].item() == 1:  # 代表 Group A
            neg_w = neg_weight_A_anchor # A 當 anchor 的時候，有完整的 push B 的力道
        else:                      # Group B
            neg_w = neg_weight_B_anchor # B 當 anchor 的時候，幾乎不推 A

        pos_list.append(hardest_pos)
        neg_list.append(hardest_neg)
        neg_w_list.append(torch.tensor(neg_w, device=device))

    # 組合成 margin ranking loss
    pos = torch.stack(pos_list)                 # [B]
    neg = torch.stack(neg_list)                 # [B]
    neg_w = torch.stack(neg_w_list)             # [B]

    # margin ranking：margin + w*(neg - pos)
    # B 當 anchor 時 w~0 → 幾乎不推 A；A 當 anchor w=1 → 正常推 B
    loss_vec = F.relu(margin + neg_w * (neg - pos))
    return loss_vec.mean()



class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.dropout = nn.Dropout(0.1)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        # # weight initialization
        # self.reset_parameters()

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding).cuda()
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding).cuda()
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
        else:
            init = get_initializer(init_method)
            init(self.user_embeddings.weight)
            init(self.item_embeddings.weight)

    def forward(self, sub_graph1, sub_graph2, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)

        # Normalize embeddings learnt from sub-graph to construct SSL loss
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        neg_item_embs = F.embedding(neg_items, item_embeddings)
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        sup_pos_ratings = inner_product(user_embs, item_embs)       # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)   # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings              # [batch_size]

        pos_ratings_user = inner_product(user_embs1, user_embs2)    # [batch_size]
        pos_ratings_item = inner_product(item_embs1, item_embs2)    # [batch_size]
        tot_ratings_user = torch.matmul(user_embs1, 
                                        torch.transpose(user_embeddings2, 0, 1))        # [batch_size, num_users]
        tot_ratings_item = torch.matmul(item_embs1, 
                                        torch.transpose(item_embeddings2, 0, 1))        # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]                  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]                  # [batch_size, num_users]

        return sup_logits, ssl_logits_user, ssl_logits_item

    def _forward_gcn(self, norm_adj):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj)


class SGL(AbstractRecommender):
    def __init__(self, config):
        super(SGL, self).__init__(config)

        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        # General hyper-parameters
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.learner = config["learner"]
        self.lr = config['lr']
        self.param_init = config["param_init"]

        # Hyper-parameters for GCN
        self.n_layers = config['n_layers']

        # Hyper-parameters for SSL
        self.ssl_aug_type = config["aug_type"].lower()
        assert self.ssl_aug_type in ['nd', 'ed', 'rw']
        self.ssl_reg = config["ssl_reg"]
        self.ssl_ratio = config["ssl_ratio"]
        self.ssl_mode = config["ssl_mode"]
        self.ssl_temp = config["ssl_temp"]
        try:
            self.groupA_neg_temp_scale = config["groupA_neg_temp_scale"]
        except Exception:
            self.groupA_neg_temp_scale = 2.0  # 預設 κ


        # Hyper-parameters for Group Contrastive Loss
        self.alpha_values = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,5.5, 6.0, 7.0, 8.0]
        self.current_alpha_idx = 0
        self.alpha_results = {}
        self.alpha_search_epochs = 20
        self.alpha_losses = []

        # Other
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)  # 保留欄位但不使用評估
        self.best_alpha = 0.0
        self.best_loss = float('inf')

        self.model_str = 'layers_%d_reg_%.0e' % (self.n_layers, self.reg)
        self.model_str += '/ratio_%.1f_mode_%s_temp_%.2f_reg_%.0e' % (
            self.ssl_ratio, self.ssl_mode, self.ssl_temp, self.ssl_reg
        )
        self.pretrain_flag = config["pretrain_flag"]
        if self.pretrain_flag:
            self.epochs = 0
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % (
                self.dataset_name, self.model_name, self.model_str
            )
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % (
                self.dataset_name, self.model_name, self.n_layers
            )
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)

        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        self.num_ratings = self.dataset.num_train_ratings

        # 指定 Group A（示例，可換成你實際標註）
        group_a_ids = [50, 98, 118, 191, 260, 550, 735, 947, 1175, 1615]

        # self.user_group_tensor 形狀是 [num_users]

        # 值為 0 → Group B（沒買冷門）

        # 值為 1 → Group A（買過冷門）

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.user_group_tensor = torch.zeros(self.num_users, dtype=torch.long)
        self.user_group_tensor[group_a_ids] = 1
        self.user_group_tensor = self.user_group_tensor.to(self.device)  # 一次搬上裝置

        # 全域使用者的群組列向量（供 user-InfoNCE 的列/行遮罩用）
        self.col_group = self.user_group_tensor.clone().to(self.device)  # [num_users]，1=A, 0=B


        # 初始圖
        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)
        if self.pretrain_flag:
            self.lightgcn.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        else:
            self.lightgcn.reset_parameters(init_method=self.param_init)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    def _build_user_activity_from_train(self):
        """
        由訓練互動資料現算每位使用者的活躍度（互動次數）。
        回傳 torch.FloatTensor [num_users]，位於 self.device。
        """
        users_items = self.dataset.train_data.to_user_item_pairs()  # shape: [N, 2], [user, item]
        users_np = users_items[:, 0]
        counts = np.bincount(users_np, minlength=self.num_users).astype(np.float32)
        # 需要的話可改成 np.log1p(counts) 做壓縮
        return torch.from_numpy(counts).to(self.device)

    @timer
    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        if is_subgraph and self.ssl_ratio > 0:
            if aug_type == 'nd':
                # 轉整數，避免浮點長度問題
                _n_u = int(self.num_users * self.ssl_ratio)
                _n_i = int(self.num_items * self.ssl_ratio)
                drop_user_idx = randint_choice(self.num_users, size=_n_u, replace=False)
                drop_item_idx = randint_choice(self.num_items, size=_n_i, replace=False)
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(users_np, dtype=np.float32), (users_np, items_np)),
                    shape=(self.num_users, self.num_items)
                )
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix(
                    (ratings_keep, (user_np_keep, item_np_keep + self.num_users)),
                    shape=(n_nodes, n_nodes)
                )
            if aug_type in ['ed', 'rw']:
                keep_idx = randint_choice(len(users_np),
                                          size=int(len(users_np) * (1 - self.ssl_ratio)),
                                          replace=False)
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix(
                    (ratings, (user_np, item_np + self.num_users)),
                    shape=(n_nodes, n_nodes)
                )
        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix(
                (ratings, (users_np, items_np + self.num_users)),
                shape=(n_nodes, n_nodes)
            )

        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def train_model(self):
    
        # ===== 主訓練：含 alpha 搜尋 =====
        for epoch in range(1, self.epochs + 1):
            # ★ 每個 epoch 重新建立一次 data_iter，避免一次性 iterator 被耗盡
            data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1,
                                      batch_size=self.batch_size, shuffle=True)
            # ★ 在 epoch 開頭就鎖定當前 alpha（這個 epoch 都用它）
            current_alpha = self.alpha_values[self.current_alpha_idx]
            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            training_start_time = time()

            # 兩視角子圖
            if self.ssl_aug_type in ['nd', 'ed']:
                sub_graph1 = sp_mat_to_sp_tensor(self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)).to(self.device)
                sub_graph2 = sp_mat_to_sp_tensor(self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)).to(self.device)
            else:
                sub_graph1, sub_graph2 = [], []
                for _ in range(0, self.n_layers):
                    sub_graph1.append(sp_mat_to_sp_tensor(self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)).to(self.device))
                    sub_graph2.append(sp_mat_to_sp_tensor(self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)).to(self.device))

            self.lightgcn.train()


            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)

                sup_logits, ssl_logits_user, ssl_logits_item = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items
                )

                # BPR
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # L2
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # === Group-aware InfoNCE（群組化溫度，無個人化 τ_i） ===
                # 基礎溫度：全為 self.ssl_temp
                base_tau_users = torch.full(
                    (bat_users.size(0),), self.ssl_temp, device=ssl_logits_user.device
                )  # [B]

                user_infonce = group_aware_infonce_full(
                    logits=ssl_logits_user,                    # [B, num_users]
                    anchor_user_ids=bat_users,                 # [B] 全域 user id
                    user_group_tensor=self.user_group_tensor,  # [num_users]
                    base_tau_vec=base_tau_users,               # [B]
                    kappa=float(self.groupA_neg_temp_scale),   # κ
                    reduction="sum"
                )



                # Item 端暫用固定溫度
                item_logits = ssl_logits_item / self.ssl_temp
                item_labels = torch.arange(item_logits.size(0), device=item_logits.device)
                item_infonce = F.cross_entropy(item_logits, item_labels, reduction="sum")

                infonce_loss = user_infonce + item_infonce

                # 在 batch 迴圈內，算 group_loss 前面加上 no_grad（避免與主損失共用計算圖）
                with torch.no_grad():
                    _u_all_1, _ = self.lightgcn._forward_gcn(sub_graph1)
                    _u_all_2, _ = self.lightgcn._forward_gcn(sub_graph2)
                    _u_all_1 = F.normalize(_u_all_1, dim=1)
                    _u_all_2 = F.normalize(_u_all_2, dim=1)

                user_embs1 = F.embedding(bat_users, _u_all_1)
                user_embs2 = F.embedding(bat_users, _u_all_2)
                group_loss = group_contrastive_loss(
                    user_embs1, user_embs2, bat_users, self.user_group_tensor,
                    margin=0.5,
                    neg_weight_B_anchor=0.0,   # B 當 anchor 不推 A（完全關閉）
                    neg_weight_A_anchor=1.0,   # A 當 anchor 正常推 B
                )
           
                # loss 組合
                current_alpha = self.alpha_values[self.current_alpha_idx]
                loss = bpr_loss + self.ssl_reg * infonce_loss + self.reg * reg_loss + current_alpha * group_loss

                total_loss += loss.item()
                total_bpr_loss += bpr_loss.item()
                total_reg_loss += (self.reg * reg_loss).item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 紀錄訓練日誌（不做任何評估）
            current_loss = total_loss / self.num_ratings
            self.alpha_losses.append(current_loss)
            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f, alpha: %.2f, time: %.2fs]" % (
                epoch,
                current_loss,
                total_bpr_loss / self.num_ratings,
                (total_loss - total_bpr_loss - total_reg_loss) / self.num_ratings,
                total_reg_loss / self.num_ratings,
                current_alpha,
                time() - training_start_time,
            ))

            # alpha 搜尋
            if epoch % self.alpha_search_epochs == 0 and epoch > 0:
                avg_loss = float(np.mean(self.alpha_losses))
                self.alpha_results[current_alpha] = avg_loss
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.best_alpha = current_alpha
                self.current_alpha_idx += 1
                if self.current_alpha_idx < len(self.alpha_values):
                    self.alpha_losses = []
                else:
                    # alpha 搜尋完畢
                    self.logger.info("Alpha search done. Best alpha: %.2f (avg loss=%.4f)" %
                                     (self.best_alpha, self.best_loss))
                    break

        # ===== 以最佳 alpha 做 final retrain（仍不做評估）=====
        self.logger.info("=== Training with Best Alpha ===")
        self.logger.info("Using best alpha: %.2f (avg loss: %.4f)" % (self.best_alpha, self.best_loss))
        self.current_alpha_idx = self.alpha_values.index(self.best_alpha)

        # 重新建立一次 data_iter（避免一次性 iterator 用盡）
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1,
                                      batch_size=self.batch_size, shuffle=True)

        final_epochs = 10
        for epoch in range(final_epochs):
            total_loss = 0.0
            training_start_time = time()
            sum_tau, cnt_tau = 0.0, 0

            # 兩視角子圖
            if self.ssl_aug_type in ['nd', 'ed']:
                sub_graph1 = sp_mat_to_sp_tensor(self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)).to(self.device)
                sub_graph2 = sp_mat_to_sp_tensor(self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)).to(self.device)
            else:
                sub_graph1, sub_graph2 = [], []
                for _ in range(0, self.n_layers):
                    sub_graph1.append(sp_mat_to_sp_tensor(self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)).to(self.device))
                    sub_graph2.append(sp_mat_to_sp_tensor(self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)).to(self.device))

            self.lightgcn.train()


            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)

                sup_logits, ssl_logits_user, ssl_logits_item = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items
                )

                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # === Group-aware InfoNCE（群組化溫度，無個人化 τ_i） ===
                base_tau_users = torch.full(
                    (bat_users.size(0),), self.ssl_temp, device=ssl_logits_user.device
                )  # [B]

                user_infonce = group_aware_infonce_full(
                    logits=ssl_logits_user,
                    anchor_user_ids=bat_users,
                    user_group_tensor=self.user_group_tensor,
                    base_tau_vec=base_tau_users,
                    kappa=float(self.groupA_neg_temp_scale),
                    reduction="sum"
                )



                # Item 端固定溫度
                item_logits = ssl_logits_item / self.ssl_temp
                item_labels = torch.arange(item_logits.size(0), device=item_logits.device)
                item_infonce = F.cross_entropy(item_logits, item_labels, reduction="sum")

                infonce_loss = user_infonce + item_infonce

                # 在 batch 迴圈內，算 group_loss 前面加上 no_grad（避免與主損失共用計算圖）
                with torch.no_grad():
                    _u_all_1, _ = self.lightgcn._forward_gcn(sub_graph1)
                    _u_all_2, _ = self.lightgcn._forward_gcn(sub_graph2)
                    _u_all_1 = F.normalize(_u_all_1, dim=1)
                    _u_all_2 = F.normalize(_u_all_2, dim=1)

                user_embs1 = F.embedding(bat_users, _u_all_1)
                user_embs2 = F.embedding(bat_users, _u_all_2)
                group_loss = group_contrastive_loss(
                    user_embs1, user_embs2, bat_users, self.user_group_tensor,
                    margin=0.5,
                    neg_weight_B_anchor=0.0,   # B 當 anchor 不推 A
                    neg_weight_A_anchor=1.0,   # A 當 anchor 正常推 B
                )




                loss = bpr_loss + self.ssl_reg * infonce_loss + self.reg * reg_loss + self.best_alpha * group_loss

                total_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.info("[Final training %d/%d : loss : %.4f, alpha: %.2f, time: %.2fs]" % (
                epoch + 1, final_epochs, total_loss / self.num_ratings, self.best_alpha, time() - training_start_time
            ))

        # ===== 僅輸出「傳播後」最終向量，不做評估 =====
        try:
            out_dir = self.export_final_embeddings(out_dir=self.save_dir)
            self.logger.info(f"export_final_embeddings done: {out_dir}")
        except Exception as e:
            self.logger.warning(f"export_final_embeddings failed: {e}")

    # 不需要 evaluate_model（整個移除）

    def predict(self, users):
        # 可留可刪；若僅要 pretrain embeddings 給 BiGNAS，可不使用
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()

    def export_final_embeddings(self, out_dir=None):
        """
        匯出 GCN 傳播後的 user/item 最終向量：
        - user_embeddings_final.npy
        - item_embeddings_final.npy
        """
        self.lightgcn.eval()
        with torch.no_grad():
            user_final, item_final = self.lightgcn._forward_gcn(self.lightgcn.norm_adj)
        user_final = user_final.detach().cpu().numpy()
        item_final = item_final.detach().cpu().numpy()

        if out_dir is None:
            if self.save_dir is not None:
                out_dir = self.save_dir
            else:
                out_dir = self.config.data_dir + f"{self.dataset_name}/pretrain-embeddings/{self.model_name}/final/"
                ensureDir(out_dir)

        np.save(out_dir + 'user_embeddings_final.npy', user_final)
        np.save(out_dir + 'item_embeddings_final.npy', item_final)
        return out_dir