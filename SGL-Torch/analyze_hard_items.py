import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd


def find_hard_items_and_export_verbose(
    model,
    groupA_ids,
    hard_user_ids,
    num_users,
    num_source_items,
    num_target_items,
    k_source,
    k_target,
    save_dir,
    preview_top_users
):
    """
    ✅ Hard User 自動挑 Hard Items（含 offset 修正 + 自動偵測）
    """

    os.makedirs(save_dir, exist_ok=True)
    device = model.device
    model.lightgcn.eval()

    # === Step 1. 匯出 embedding ===
    with torch.no_grad():
        uemb, iemb = model.lightgcn._forward_gcn(model.lightgcn.norm_adj)
        uemb = F.normalize(uemb, dim=1)
        iemb = F.normalize(iemb, dim=1)
    print("=" * 80)
    print(f"[1] Embedding ready: user={tuple(uemb.shape)}, item={tuple(iemb.shape)}")

    total_items = iemb.size(0)
    # 若沒指定，則自動平分為 source/target
    if num_source_items == 0 and num_target_items == 0:
        num_source_items = total_items // 2
        num_target_items = total_items - num_source_items
        print(f"⚠️ 自動推測 item 範圍: source={num_source_items}, target={num_target_items}")

    groupA = torch.tensor(groupA_ids, dtype=torch.long, device=device)
    hardU = torch.tensor(hard_user_ids, dtype=torch.long, device=device)

    # === Step 2. 計算 GroupA 對所有 item 的平均分數 ===
    with torch.no_grad():
        scores_A = model.lightgcn.predict(groupA)          # [|A|, num_items]
        mean_A = scores_A.mean(dim=0, keepdim=True)        # [1, num_items]
    print("=" * 80)
    print(f"[2] 計算 GroupA 平均分數完成: shape={mean_A.shape}")

    all_source_edges, all_target_edges = [], []
    preview_log = []

    for uid in hard_user_ids:
        with torch.no_grad():
            score_u = model.lightgcn.predict(torch.tensor([uid], device=device))  # [1, num_items]
            delta = mean_A - score_u
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            delta = torch.abs(delta).squeeze(0)

            # === Source domain ===
            delta_src = delta[:num_source_items]
            if delta_src.numel() > 0:
                vals_s, idx_s = torch.topk(delta_src, k=min(k_source, delta_src.numel()))
                for i, val in zip(idx_s.cpu().tolist(), vals_s.cpu().tolist()):
                    i_global = num_users + i
                    all_source_edges.append((uid, i_global))
                    if len(preview_log) < preview_top_users * (k_source + k_target):
                        preview_log.append((uid, i_global, score_u[0, i].item(), mean_A[0, i].item(), val))

            # === Target domain ===
            delta_tgt = delta[num_source_items:num_source_items + num_target_items]
            if delta_tgt.numel() > 0:
                vals_t, idx_t = torch.topk(delta_tgt, k=min(k_target, delta_tgt.numel()))
                for i, val in zip(idx_t.cpu().tolist(), vals_t.cpu().tolist()):
                    i_global = num_users + num_source_items + i
                    all_target_edges.append((uid, i_global))
                    if len(preview_log) < preview_top_users * (k_source + k_target):
                        preview_log.append((uid, i_global, score_u[0, i_global - num_users].item(), mean_A[0, i_global - num_users].item(), val))

    # === Step 3. 輸出假邊 ===
    def make_edge_tensor(edge_list):
        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t()

    E_add_source = make_edge_tensor(all_source_edges)
    E_add_target = make_edge_tensor(all_target_edges)
    np.save(os.path.join(save_dir, "E_add_source.npy"), E_add_source.cpu().numpy())
    np.save(os.path.join(save_dir, "E_add_target.npy"), E_add_target.cpu().numpy())

    print("=" * 80)
    print(f"[3] ✅ Hard Item 選取完畢 (每人加 {k_source}+{k_target} 條)")
    print(f"    Hard Users 數量: {len(hard_user_ids)}")
    print(f"    Source domain 假邊: {E_add_source.size(1)} 條")
    print(f"    Target domain 假邊: {E_add_target.size(1)} 條")

    # === Step 4. 預覽 top 用戶 ===
    print("=" * 80)
    print(f"[4] 🔍 Hard User 加邊預覽 (前 {preview_top_users} 位)")
    print(f"{'User':>6} | {'Item':>6} | {'HardUserScore':>13} | {'GroupA_Mean':>12} | {'|Δ|':>8}")
    print("-" * 60)
    for uid, iid, sc_u, sc_a, diff in preview_log[:preview_top_users * (k_source + k_target)]:
        print(f"{uid:>6d} | {iid:>6d} | {sc_u:>13.6f} | {sc_a:>12.6f} | {diff:>8.6f}")
    print("-" * 60)

    # === Step 5. 輸出 CSV 統計 ===
    src_df = pd.DataFrame(E_add_source.cpu().numpy().T, columns=["user_id", "item_id"])
    tgt_df = pd.DataFrame(E_add_target.cpu().numpy().T, columns=["user_id", "item_id"])
    src_df.to_csv(os.path.join(save_dir, "E_add_source.csv"), index=False)
    tgt_df.to_csv(os.path.join(save_dir, "E_add_target.csv"), index=False)

    print(f"[5] 輸出完成：{save_dir}/E_add_source.npy, E_add_target.npy, CSV版")
    print("=" * 80)

    return E_add_source, E_add_target, src_df, tgt_df
