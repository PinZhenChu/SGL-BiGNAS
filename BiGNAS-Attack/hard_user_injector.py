import os
import math
import logging
import numpy as np
import torch
import torch.nn.functional as F


def _tensor2set(edge_index: torch.Tensor):
    return set(map(tuple, edge_index.t().tolist()))


def _apply_additions(edge_index: torch.Tensor, additions):
    s = _tensor2set(edge_index)
    for (u, i) in additions:
        s.add((int(u), int(i)))
    out = torch.tensor(list(s), dtype=torch.long).t()
    return out


class HardUserInjector:
    """
    方法A：Hard Users + 加邊（比例版）
    - target domain: 用 SGL user embedding 找 Hard Users（dist = 1 - max cos sim）
    - source domain: 以 GroupA 的熱門品為候選，從「所有可加邊」中抽 edge_ratio_source 比例
    - target domain: 對 Hard Users 連 cold item，抽 edge_ratio_target 比例
    """

    def __init__(self, top_ratio=0.10, log_dir="logs/hard_user"):
        assert 0 < top_ratio <= 1.0
        self.top_ratio = top_ratio
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    @staticmethod
    def _auto_find_cold_item(target_train_edge_index, num_target_items):
        item_ids, counts = target_train_edge_index[1].unique(return_counts=True)
        if item_ids.numel() == 0:
            return 0
        min_count = counts.min()
        candidates = item_ids[counts == min_count]
        return int(candidates.min().item())

    @staticmethod
    def _split_users_by_target_item(target_train_edge_index, cold_item_id, num_users):
        mask = (target_train_edge_index[1] == cold_item_id)
        ua = target_train_edge_index[0][mask].unique()
        groupA = set(ua.tolist())
        all_users = set(range(num_users))
        groupB = list(all_users - groupA)
        return list(groupA), groupB

    @staticmethod
    def _pick_hard_users(user_emb_target, groupA_users, groupB_users, top_ratio):
        if len(groupA_users) == 0 or len(groupB_users) == 0:
            return []
        A = torch.tensor(groupA_users, dtype=torch.long, device=user_emb_target.device)
        B = torch.tensor(groupB_users, dtype=torch.long, device=user_emb_target.device)
        uA = F.normalize(user_emb_target[A], dim=-1)
        uB = F.normalize(user_emb_target[B], dim=-1)
        sim = torch.matmul(uB, uA.t())              # [|B|, |A|]
        max_sim, _ = sim.max(dim=1)                 # 每個 B 與 A 的最大相似
        dist = 1.0 - max_sim                        # 距離大 → 更難
        k = max(1, math.ceil(len(groupB_users) * top_ratio))
        k = min(k, dist.numel())                    # 安全邊界
        top_idx = torch.topk(dist, k=k, largest=True).indices
        hard_users = [int(B[i]) for i in top_idx]
        return hard_users

    @staticmethod
    def _rank_source_items_by_groupA(source_train_edge_index, groupA_users, num_source_items):
        if len(groupA_users) == 0:
            return []
        mask = torch.isin(source_train_edge_index[0],
                          torch.tensor(groupA_users, dtype=torch.long, device=source_train_edge_index.device))
        items = source_train_edge_index[1][mask]
        if items.numel() == 0:
            return []
        unique_items, counts = items.unique(return_counts=True)
        order = torch.argsort(counts, descending=True)

        ranked_items = [int(i.item()) for i in unique_items[order]]

        # ✅ 保證 item id 落在合法範圍
        ranked_items = [i for i in ranked_items if 0 <= i < num_source_items]
        return ranked_items

    def run(
        self,
        split_result,
        user_emb_target: torch.Tensor,
        num_users: int,
        num_source_items: int,
        num_target_items: int,
        edge_ratio_source: float = 0.1,
        edge_ratio_target: float = 1.0,
        cold_item_id: int = -1,  # global id
    ):
        logging.info("[HardUser] 開始執行方法A（Hard Users + 加邊策略, global id版）...")

        # splits
        source_train_edge_index = split_result["source_train_edge_index"]
        target_train_edge_index = split_result["target_train_edge_index"]
        target_valid_edge_index = split_result.get("target_valid_edge_index", None)
        target_test_edge_index  = split_result.get("target_test_edge_index", None)

        # === Step 1: Cold item global id ===
        if cold_item_id < 0:
            raise ValueError("方案B下必須指定 cold_item_id 為 global id")
        cold_item_id_global = cold_item_id
        logging.info(f"[HardUser] Using global cold_item_id={cold_item_id_global}")

        # === Step 2: 轉 local id 來分組 ===
        cold_item_local = cold_item_id_global - (num_users + num_source_items)
        if cold_item_local < 0 or cold_item_local >= num_target_items:
            raise ValueError(f"cold_item_id_global={cold_item_id_global} 不在 target 範圍內!")

        # 只轉換 target item → local，不動 user
        target_train_local = target_train_edge_index.clone()
        target_train_local[1] -= (num_users + num_source_items)

        groupA, groupB = self._split_users_by_target_item(
            target_train_local,
            cold_item_local,
            num_users
        )
        logging.info(f"[HardUser] GroupA={len(groupA)} users, GroupB={len(groupB)} users")

        if len(groupA) == 0 or len(groupB) == 0:
            logging.warning("[HardUser] No valid GroupA or GroupB → 不產生 ΔE")
            empty = torch.empty((2, 0), dtype=torch.long)
            return {
                "hard_users": [],
                "cold_item_id": cold_item_id_global,
                "E_add_source": empty,
                "E_add_target": empty
            }

        # === Step 3: Hard Users ===
        hard_users = self._pick_hard_users(user_emb_target, groupA, groupB, self.top_ratio)
        logging.info(f"[HardUser] Picked {len(hard_users)} hard users (top_ratio={self.top_ratio})")

        # === Step 4: Target domain 假邊（用 global id 加邊） ===
        E_add_target = torch.empty((2, 0), dtype=torch.long)
        if edge_ratio_target > 0 and len(hard_users) > 0:
            rows = torch.tensor(hard_users, dtype=torch.long)
            cols = torch.full((len(hard_users),), cold_item_id_global, dtype=torch.long)
            E_target = torch.stack([rows, cols], dim=0)

            if edge_ratio_target < 1.0:
                keep = max(1, int(E_target.size(1) * edge_ratio_target))
                perm = torch.randperm(E_target.size(1))[:keep]
                E_target = E_target[:, perm]

            # 邊界檢查
            valid_min, valid_max = num_users + num_source_items, num_users + num_source_items + num_target_items - 1
            mask = (E_target[1] >= valid_min) & (E_target[1] <= valid_max)
            if mask.sum().item() < E_target.size(1):
                logging.warning(f"[HardUser] 移除 {E_target.size(1) - mask.sum().item()} 條越界 target 假邊")
            E_add_target = E_target[:, mask]

        # === Step 5: Source domain 假邊（比例抽樣） ===
        E_add_source = torch.empty((2, 0), dtype=torch.long)
        if edge_ratio_source > 0 and len(hard_users) > 0:
            ranked_items = self._rank_source_items_by_groupA(source_train_edge_index, groupA, num_source_items)
            if not ranked_items:
                ranked_items = list(range(num_source_items))

            exist_s = _tensor2set(source_train_edge_index)
            cand_pairs = []
            for u in hard_users:
                for i in ranked_items:
                    p = (int(u), int(i))
                    if p not in exist_s:
                        cand_pairs.append(p)

            if len(cand_pairs) > 0:
                keep = max(1, int(len(cand_pairs) * edge_ratio_source))
                idx = torch.randperm(len(cand_pairs))[:keep].tolist()
                picked = [cand_pairs[j] for j in idx]
                E_add_source = torch.tensor(picked, dtype=torch.long).t()

                # 邊界檢查
                mask = (E_add_source[1] >= 0) & (E_add_source[1] < num_source_items)
                if mask.sum().item() < E_add_source.size(1):
                    logging.warning(f"[HardUser] 移除 {E_add_source.size(1) - mask.sum().item()} 條越界 source 假邊")
                E_add_source = E_add_source[:, mask]

        # === Step 6: 存檔 & log ===
        np.save(os.path.join(self.log_dir, "E_add_source.npy"), E_add_source.cpu().numpy())
        np.save(os.path.join(self.log_dir, "E_add_target.npy"), E_add_target.cpu().numpy())
        logging.info(f"[HardUser] ✅ ΔE 已輸出: {self.log_dir}/E_add_source.npy, {self.log_dir}/E_add_target.npy")

        return {
            "hard_users": hard_users,
            "cold_item_id": cold_item_id_global,
            "E_add_source": E_add_source,
            "E_add_target": E_add_target,
        }
