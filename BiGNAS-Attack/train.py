import logging

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import numpy as np  
from auxilearn.optim import MetaOptimizer
from dataset import Dataset
from pytorchtools import EarlyStopping
from utils import link_split, load_model


def meta_optimizeation(
    target_meta_loader,
    replace_optimizer,
    model,
    args,
    criterion,
    replace_scheduler,
    source_edge_index,
    target_train_edge_index,   # ✅ 固定使用 train
):
    device = args.device
    for batch, (target_link, target_label) in enumerate(target_meta_loader):
        if batch < args.descent_step:
            target_link, target_label = target_link.to(device), target_label.to(device)

            replace_optimizer.zero_grad()
            out = model.meta_prediction(
                source_edge_index, target_train_edge_index, target_link
            ).squeeze()
            loss_target = criterion(out, target_label).mean()
            loss_target.backward()
            replace_optimizer.step()
        else:
            break
    replace_scheduler.step()

@torch.no_grad()
def evaluate(name, model, source_edge_index, target_edge_index, link, label):
    model.eval()

    out = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
    try:
        auc = roc_auc_score(label.tolist(), out.tolist())
    except:
        auc = 1.0
    logging.info(f"{name} AUC: {auc:4f}")

    model.train()
    return auc
def get_test_positive_dict(data):
    """
    根據 test link（data.target_test_link）建立 test set user 的正樣本字典。
    回傳: {user_id: [item1, item2, ...]}
    """
    test_user_item_dict = {}
    test_link = data.target_test_link.cpu()
    for u, i in zip(test_link[0], test_link[1]):
        u, i = u.item(), i.item()
        if u not in test_user_item_dict:
            test_user_item_dict[u] = []
        test_user_item_dict[u].append(i)
    return test_user_item_dict

def evaluate_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    top_k, num_candidates=99,
    device=None
):
    import random
    model.eval()
    hit_count = 0
    # all_target_items = set(range(data.num_target_items))
    # 改成 global 範圍
    all_target_items = set(range(
        data.num_users + data.num_source_items,
        data.num_users + data.num_source_items + data.num_target_items
    ))

    # ✅ 取得 test set 的 user -> positive items 對應關係
    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())  # 直接使用 test set 的 user
    print(f"✅ Test set user count: {len(sim_users)}")

    total_users = 0
    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    with torch.no_grad():
        for user_id in sim_users:
            pos_items = user_interactions.get(user_id, set())
            if len(pos_items) > 1:
                print(f"⚠️ Warning: User {user_id} has {len(pos_items)} positives in test set.")

            if len(pos_items) == 0:
                continue

            # ✅ 第一步：選擇一個正樣本
            pos_item = list(pos_items)[0]
            # print(f"\n=== [User {user_id}] ===")
            # print(f"👉 Positive item: {pos_item}")

            # ✅ 第二步：挑選負樣本（從非正樣本中隨機抽 num_candidates 個）
            negative_pool = list(all_target_items - set(pos_items))
            if len(negative_pool) < num_candidates:
                # print(f"❌ Negative pool too small for user {user_id}, skipping.")
                continue

            sampled_negatives = random.sample(negative_pool, num_candidates)
            # print(f"🎯 Sampled {num_candidates} negatives: {sampled_negatives[:10]}...")

            # ✅ 第三步：組成候選清單（正例 + 負例），並打亂
            candidate_items = sampled_negatives + [pos_item]
            random.shuffle(candidate_items)
            # print(f"🧮 Candidate items (shuffled): {candidate_items[:10]}...")

            # ✅ 第四步：轉成 tensor 並送入模型計算分數
            user_tensor = torch.tensor([user_id] * len(candidate_items), device=device)
            item_tensor = torch.tensor(candidate_items, device=device)
            link = torch.stack([user_tensor, item_tensor], dim=0)

            scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            top_k_indices = torch.topk(scores, k=top_k).indices.tolist()
            top_k_items = [candidate_items[i] for i in top_k_indices]

            # print(f"📈 Top-{top_k} prediction: {top_k_items}")
            # print(f"✔️ Hit? {'Yes ✅' if pos_item in top_k_items else 'No ❌'}")

            if pos_item in top_k_items:
                hit_count += 1
            total_users += 1

    hit_ratio = hit_count / total_users if total_users > 0 else 0.0
    logging.info(f"[HIT_RATIO@{top_k}] Users={total_users}, Hits={hit_count}, Hit Ratio={hit_ratio:.4f}")
    return hit_ratio

# 🔍 統計每個 cold item 在 test set 中出現的次數（有幾個 user 買過）
def count_cold_item_occurrences(data, cold_item_set):
    item_count = {item: 0 for item in cold_item_set}
    test_link = data.target_test_link.cpu().numpy()
    for u, i in zip(*test_link):
        if i in cold_item_set:
            item_count[i] += 1
    return item_count

# def find_cold_item_strict(data, target_train_edge_index, target_test_edge_index):
#     import numpy as np
#     from collections import Counter

#     train_edges = target_train_edge_index.cpu().numpy()
#     test_edges = target_test_edge_index.cpu().numpy()
#     overlap_users = set(data.raw_overlap_users.cpu().numpy())  # ⬅️ overlap user list

#     # Step 1: 統計 overlap user 在 test set 中點擊的 item 次數
#     test_user, test_item = test_edges
#     item_counter = Counter()

#     for u, i in zip(test_user, test_item):
#         if u in overlap_users:
#             item_counter[i] += 1

#     candidate_items = {i for i, cnt in item_counter.items() if cnt == 1}

#     train_items = set(train_edges[1])
#     test_items = set(test_item)

#     cold_items = [i for i in candidate_items if i not in train_items and i in test_items]

#     if not cold_items:
#         print("❌ 找不到符合條件的 cold item")
#         return None

#     selected = cold_items[0]
#     print(f"🧊 Found cold item: {selected}")
#     return selected

def evaluate_er_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    cold_item_set,
    top_k, num_candidates=99,
    device=None
):
    import random
    model.eval()

    # all_target_items = set(range(data.num_target_items))
    # 改成 global 範圍
    all_target_items = set(range(
        data.num_users + data.num_source_items,
        data.num_users + data.num_source_items + data.num_target_items
    ))
    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())

    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    total_users = 0
    cold_item_hit_count = 0
    cold_item_ranks = []  # ⬅️ 儲存 cold item 被排進去時的排名

    with torch.no_grad():
        for user_id in sim_users:
            # 建立候選池
            negative_pool = list(all_target_items - cold_item_set)
            if len(negative_pool) < num_candidates:
                continue

            sampled_items = random.sample(negative_pool, num_candidates)
            sampled_items += list(cold_item_set)
            sampled_items = list(set(sampled_items))
            random.shuffle(sampled_items)

            user_tensor = torch.tensor([user_id] * len(sampled_items), device=device)
            item_tensor = torch.tensor(sampled_items, device=device)
            link = torch.stack([user_tensor, item_tensor], dim=0)

            scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            scores_list = scores.tolist()

            # 印出每個 item 的分數
            # print(f"\n=== [User {user_id}] ===")
            # for item, score in zip(sampled_items, scores_list):
            #     tag = "🧊 COLD" if item in cold_item_set else ""
            #     print(f"Item {item:4d} | Score: {score:.4f} {tag}")

            # 計算排序
            item_score_pairs = list(zip(sampled_items, scores_list))
            item_score_pairs.sort(key=lambda x: x[1], reverse=True)
            sorted_items = [item for item, _ in item_score_pairs]

            # 印出 cold item 的排名
            for cold_item in cold_item_set:
                if cold_item in sorted_items:
                    rank = sorted_items.index(cold_item) + 1
                    # print(f"🔍 Cold item {cold_item} ranked #{rank} / {len(sorted_items)}")

            top_k_items = sorted_items[:top_k]


            # ⬇️ 統計命中與排名
            cold_hits = [item for item in top_k_items if item in cold_item_set]
            if cold_hits:
                cold_item_hit_count += 1
                for cold_item in cold_hits:
                    rank = top_k_items.index(cold_item) + 1  # 1-based rank
                    cold_item_ranks.append(rank)

            total_users += 1

    er_ratio = cold_item_hit_count / total_users if total_users > 0 else 0.0
    avg_rank = sum(cold_item_ranks) / len(cold_item_ranks) if cold_item_ranks else -1
    median_rank = (
        sorted(cold_item_ranks)[len(cold_item_ranks) // 2] if cold_item_ranks else -1
    )

    logging.info(f"[ER@{top_k}] Users={total_users}, Cold Item Hits={cold_item_hit_count}, ER Ratio={er_ratio:.4f}")
    # logging.info(f"[ER@{top_k}] Cold item avg rank: {avg_rank:.2f}, median rank: {median_rank}")

    return er_ratio


def evaluate_multiple_topk(model, data, source_edge_index, target_edge_index, cold_item_set, device):
    topk_list = [10, 15, 20, 25, 30]
    print("\n📊 Evaluation for multiple top-K values:")
    for k in topk_list:
        hr = evaluate_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            top_k=k,
            num_candidates=99,
            device=device
        )

        er = evaluate_er_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            cold_item_set=cold_item_set,
            top_k=k,
            num_candidates=99,
            device=device
        )

# === 保留這個（原本完整版本，能分辨 target / source）===
def check_all_edges(edge_index, num_users, num_source_items, num_target_items, name, is_target=False):
    if edge_index is None or edge_index.numel() == 0:
        logging.info(f"[{name}] empty (skip check)")
        return

    u, v = edge_index
    u_min, u_max = u.min().item(), u.max().item()
    v_min, v_max = v.min().item(), v.max().item()

    if is_target:
        valid_min = num_users + num_source_items
        valid_max = num_users + num_source_items + num_target_items - 1
    else:
        valid_min = num_users
        valid_max = num_users + num_source_items - 1

    logging.info(
        f"[{name}] users {u_min}~{u_max} (limit {num_users-1}), "
        f"items {v_min}~{v_max}, valid [{valid_min}~{valid_max}]"
    )

    if u_min < 0 or u_max >= num_users:
        raise ValueError(f"[{name}] user id 越界: {u_min} ~ {u_max}, 應該在 [0, {num_users-1}]")
    if v_min < valid_min or v_max > valid_max:
        raise ValueError(f"[{name}] item id 越界: {v_min} ~ {v_max}, 應該在 [{valid_min}, {valid_max}]")

# === 新增一個給「全局 num_nodes」用的 ===
def check_all_edges_global(edge_index, num_nodes, name):
    if edge_index is None or edge_index.numel() == 0:
        logging.info(f"[{name}] empty (skip check)")
        return
    if edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes:
        raise ValueError(
            f"❌ {name}: index 越界 (min={edge_index.min().item()}, max={edge_index.max().item()}, num_nodes={num_nodes})"
        )
    u, v = edge_index
    logging.info(
        f"[{name}] OK: u[{u.min().item()}~{u.max().item()}], v[{v.min().item()}~{v.max().item()}], num_nodes={num_nodes}"
    )

def train(model, perceptor, data, args, split_result):
    device = args.device
    data = data.to(device)
    model = model.to(device)
    perceptor = perceptor.to(device)

    # ✅ edge_index 只用 split_result (global id)
    source_edge_index       = split_result["source_train_edge_index"].to(device)
    target_train_edge_index = split_result["target_train_edge_index"].to(device)
    target_valid_edge_index = split_result["target_valid_edge_index"].to(device)
    target_test_edge_index  = split_result["target_test_edge_index"].to(device)

    # ✅ link & label 還是從 link_split 拿，但 edge_index 忽略
    (
        _,
        source_label,
        source_link,
        _,
        target_train_label,
        target_train_link,
        target_valid_link,
        target_valid_label,
        target_test_link,
        target_test_label,
        _,
    ) = link_split(data)

    # ✅ 全局節點數
    num_nodes = args.num_users + args.num_source_items + args.num_target_items
    check_all_edges_global(source_edge_index, num_nodes, "source_edge_index")
    check_all_edges_global(target_train_edge_index, num_nodes, "target_train_edge_index")
    check_all_edges_global(target_valid_edge_index, num_nodes, "target_valid_edge_index")
    check_all_edges_global(target_test_edge_index, num_nodes, "target_test_edge_index")


    data.target_test_link = target_test_link
    source_set_size = source_link.shape[1]
    train_set_size = target_train_link.shape[1]
    val_set_size = target_valid_link.shape[1]
    test_set_size = target_test_link.shape[1]
    logging.info(f"Train set size: {train_set_size}")
    logging.info(f"Valid set size: {val_set_size}")
    logging.info(f"Test set size: {test_set_size}")

    target_train_set = Dataset(
        target_train_link.to("cpu"),
        target_train_label.to("cpu"),
    )
    target_train_loader = DataLoader(
        target_train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=target_train_set.collate_fn,
    )

    source_batch_size = int(args.batch_size * train_set_size / source_set_size)
    source_train_set = Dataset(source_link.to("cpu"), source_label.to("cpu"))
    source_train_loader = DataLoader(
        source_train_set,
        batch_size=source_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=source_train_set.collate_fn,
    )

    target_meta_loader = DataLoader(
        target_train_set,
        batch_size=args.meta_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=target_train_set.collate_fn,
    )
    target_meta_iter = iter(target_meta_loader)
    source_meta_batch_size = int(
        args.meta_batch_size * train_set_size / source_set_size
    )
    source_meta_loader = DataLoader(
        source_train_set,
        batch_size=source_meta_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=source_train_set.collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    perceptor_optimizer = torch.optim.Adam(
        perceptor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    meta_optimizer = MetaOptimizer(
        meta_optimizer=perceptor_optimizer,
        hpo_lr=args.hpo_lr,
        truncate_iter=3,
        max_grad_norm=10,
    )

    model_param = [
        param for name, param in model.named_parameters() if "preds" not in name
    ]
    replace_param = [
        param for name, param in model.named_parameters() if name.startswith("replace")
    ]
    replace_optimizer = torch.optim.Adam(replace_param, lr=args.lr)
    replace_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        replace_optimizer, T_max=args.T_max
    )

    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=args.model_path,
        trace_func=logging.info,
    )

    criterion = nn.BCELoss(reduction="none")
    iteration = 0
    for epoch in range(args.epochs):
        for (source_link, source_label), (target_link, target_label) in zip(
            source_train_loader, target_train_loader
        ):
            torch.cuda.empty_cache()
            source_link = source_link.to(device)
            source_label = source_label.to(device)
            target_link = target_link.to(device)
            target_label = target_label.to(device)
            weight_source = perceptor(source_link[1], source_edge_index, model)

            optimizer.zero_grad()
            source_out = model(
                source_edge_index, target_train_edge_index, source_link, is_source=True
            ).squeeze()
            target_out = model(
                source_edge_index, target_train_edge_index, target_link, is_source=False
            ).squeeze()
            source_loss = (
                criterion(source_out, source_label).reshape(-1, 1) * weight_source
            ).sum()
            target_loss = criterion(target_out, target_label).mean()
            loss = source_loss + target_loss if args.use_meta else target_loss
            loss.backward()
            optimizer.step()

            iteration += 1
            if (
                args.use_source
                and args.use_meta
                and iteration % args.meta_interval == 0
            ):
                logging.info(f"Entering meta optimization, iteration: {iteration}")
                meta_optimizeation(
                    target_meta_loader,
                    replace_optimizer,
                    model,
                    args,
                    criterion,
                    replace_scheduler,
                    source_edge_index,
                    target_train_edge_index,
                )

                try:
                    target_meta_link, target_meta_label = next(target_meta_iter)
                except StopIteration:
                    target_meta_iter = iter(target_meta_loader)
                    target_meta_link, target_meta_label = next(target_meta_iter)

                target_meta_link, target_meta_label = (
                    target_meta_link.to(device),
                    target_meta_label.to(device),
                )
                optimizer.zero_grad()
                target_out = model(
                    source_edge_index,
                    target_train_edge_index,
                    target_meta_link,
                    is_source=False,
                ).squeeze()
                meta_loss = criterion(target_out, target_meta_label).mean()

                for (source_link, source_label), (target_link, target_label) in zip(
                    source_meta_loader, target_meta_loader
                ):
                    source_link, source_label = source_link.to(device), source_label.to(
                        device
                    )
                    target_link, target_label = target_link.to(device), target_label.to(
                        device
                    )
                    weight_source = perceptor(source_link[1], source_edge_index, model)

                    optimizer.zero_grad()
                    source_out = model(
                        source_edge_index,
                        target_train_edge_index,
                        source_link,
                        is_source=True,
                    ).squeeze()
                    target_out = model(
                        source_edge_index,
                        target_train_edge_index,
                        target_link,
                        is_source=False,
                    ).squeeze()
                    source_loss = (
                        criterion(source_out, source_label).reshape(-1, 1)
                        * weight_source
                    ).sum()
                    target_loss = criterion(target_out, target_label).mean()
                    meta_train_loss = (
                        source_loss + target_loss if args.use_meta else target_loss
                    )
                    break

                torch.cuda.empty_cache()
                meta_optimizer.step(
                    train_loss=meta_train_loss,
                    val_loss=meta_loss,
                    aux_params=list(perceptor.parameters()),
                    parameters=model_param,
                    return_grads=True,
                    entropy=None,
                )
        train_auc = evaluate(
            "Train",
            model,
            source_edge_index,
            target_train_edge_index,
            target_train_link,
            target_train_label,
        )
        val_auc = evaluate(
            "Valid",
            model,
            source_edge_index,
            target_train_edge_index,
            target_valid_link,
            target_valid_label,
        )

        logging.info(
            f"[Epoch: {epoch}]Train Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Valid AUC: {val_auc:.4f}"
        )
        wandb.log(
            {
                "loss": loss,
                "train_auc": train_auc,
                "val_auc": val_auc
            },
            step=epoch,
        )

        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

        lr_scheduler.step()

    model = load_model(args).to(device)
    evaluate_hit_ratio(
        model=model,
        data=data,
        source_edge_index=source_edge_index,
        target_edge_index=target_train_edge_index,  # ✅ 正確傳入測試集 edge_index
        top_k=args.top_k,
        num_candidates=99,
        device=device,
    )
    # cold_item_id = find_cold_item_strict(data, target_train_edge_index, target_test_edge_index)
    if args.cold_item_id >= 0:
        cold_item_id = args.cold_item_id
        target_min = args.num_users + args.num_source_items
        target_max = args.num_users + args.num_source_items + args.num_target_items - 1
        assert target_min <= cold_item_id <= target_max, \
            f"❌ cold_item_id={cold_item_id} 不在 target 範圍 [{target_min}, {target_max}]"
        logging.info(f"[ColdItem] 使用指定 cold_item_id={cold_item_id}")
    else:
        # 保險起見，還是留一個 fallback（自動選）
        cold_item_id = None
        logging.info("[ColdItem] 未指定，跳過 ER 評估")

    if cold_item_id is not None:
        evaluate_er_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_train_edge_index,
            cold_item_set={cold_item_id},
            top_k=args.top_k,
            num_candidates=99,
            device=device,
        )


    # logging.info(f"Hit Ratio (no injection): {pre_hit_ratio:.4f}")
    test_auc = evaluate(
        "Test",
        model,
        source_edge_index,
        target_train_edge_index,
        target_test_link,
        target_test_label,
    )
    logging.info(f"Test AUC: {test_auc:.4f}")
    wandb.log({"Test AUC": test_auc})
    evaluate_multiple_topk(
        model=model,
        data=data,
        source_edge_index=source_edge_index,
        target_edge_index=target_train_edge_index,
        cold_item_set={cold_item_id},   # 注意這邊是 set，不是 cold_item_id=
        device=device
    )
        # === 存下 source_item_embedding ===
    # source_emb = model.source_item_embedding.weight.detach().cpu().numpy()
    # np.save("source_item_embedding.npy", source_emb)
    # np.savetxt("source_item_embedding.csv", source_emb, delimiter=",")
    # logging.info(f"✅ Saved source_item_embedding: shape={source_emb.shape}")