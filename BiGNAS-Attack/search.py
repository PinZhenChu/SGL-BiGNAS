import argparse
import logging
import os
import time

import wandb
import numpy as np
import torch

from hard_user_injector import HardUserInjector
from utils import set_logging, set_seed
from dataset import CrossDomain
from model import Model, Perceptor
from train import train

def debug_cold_item_counts(split_result, cold_item_id):
    """
    Debug: 計算冷門商品在 train/valid/test 出現次數
    split_result: link_split() 的回傳結果
        - 包含 target_train_edge_index, target_valid_edge_index, target_test_edge_index
    cold_item_id: int, target domain 冷門商品 index
    """
    # === train ===
    train_edges = split_result['target_train_edge_index']
    train_count = (train_edges[1] == cold_item_id).sum().item()

    # === valid ===
    valid_edges = split_result['target_valid_edge_index']
    valid_count = (valid_edges[1] == cold_item_id).sum().item()

    # === test ===
    test_edges = split_result['target_test_edge_index']
    test_count = (test_edges[1] == cold_item_id).sum().item()

    print(f"[DEBUG] cold_item_id={cold_item_id}")
    print(f"  train 出現次數: {train_count}")
    print(f"  valid 出現次數: {valid_count}")
    print(f"  test  出現次數: {test_count}")

    return train_count, valid_count, test_count


def search(args):
    args.search = True

    wandb.init(project="BiGNAS", config=args)
    set_seed(args.seed)
    set_logging()

    logging.info(f"args: {args}")

    # === 載入資料 ===
    dataset = CrossDomain(
        root=args.root,
        categories=args.categories,
        target=args.target,
        use_source=args.use_source,
    )
    data = dataset[0]

    # === 基本統計 ===
    args.num_users = data.num_users
    args.num_source_items = data.num_source_items
    args.num_target_items = data.num_target_items
    logging.info(f"data: {data}")

    # === 模型存檔路徑 ===
    DATE_FORMAT = "%Y-%m-%d_%H:%M:%S"
    args.model_path = os.path.join(
        args.model_dir,
        f'{time.strftime(DATE_FORMAT, time.localtime())}_{"_".join(args.categories)}.pt',
    )

    # === split_result: BiGNAS 用的標準輸入格式 ===
    split_result = {
        "source_train_edge_index": data.source_link,
        "target_train_edge_index": data.target_train_edge_index,
        "target_valid_edge_index": data.target_valid_edge_index,
        "target_test_edge_index": data.target_test_edge_index,
    }
    # === 將 split_result 裡的邊輸出到檔案 ===
    import numpy as np
    os.makedirs("logs/split_edges", exist_ok=True)

    def save_edge_index(name, edge_index):
        npy_path = f"logs/split_edges/{name}.npy"
        csv_path = f"logs/split_edges/{name}.csv"
        # 存 npy
        np.save(npy_path, edge_index.cpu().numpy())
        # 存 csv（兩欄：user,item）
        np.savetxt(csv_path, edge_index.cpu().numpy().T, fmt="%d", delimiter=",")
        logging.info(f"[Search] 已輸出 {name}: {edge_index.shape}, npy={npy_path}, csv={csv_path}")

    save_edge_index("source_train_edge_index", data.source_link)
    save_edge_index("target_train_edge_index", data.target_train_edge_index)
    save_edge_index("target_valid_edge_index", data.target_valid_edge_index)
    save_edge_index("target_test_edge_index",  data.target_test_edge_index)


    # === HardUser 加邊策略 ===
    if args.use_hard_user_augment:
        logging.info("[HardUser] 開始執行方法A（Hard Users + 加邊策略）...")

        injector = HardUserInjector(
            top_ratio=args.hard_top_ratio,
            log_dir="logs/hard_user",
        )

        # 讀 SGL 輸出的 target user embedding
        # user_emb_target_path = os.path.join(args.sgl_dir_target, "user_embeddings_final.npy")
        user_emb_target_path = os.path.join(args.sgl_dir_target, "user_embeddings_final.npy")
        if not os.path.exists(user_emb_target_path):
            raise FileNotFoundError(f"[HardUser] 找不到 SGL user embedding：{user_emb_target_path}")

        user_emb_target = torch.tensor(np.load(user_emb_target_path), dtype=torch.float)

        # 執行加邊，得到 ΔE
        summary = injector.run(
            split_result=split_result,
            user_emb_target=user_emb_target,
            num_users=args.num_users,
            num_source_items=args.num_source_items,
            num_target_items=args.num_target_items,
            cold_item_id=args.cold_item_id,
            edge_ratio_source=args.edge_ratio_source,   # 新增：控制 source domain 假邊比例
            edge_ratio_target=args.edge_ratio_target,   # 新增：控制 target domain 假邊比例
        )

        logging.info(
            f"[HardUser] hard_users={len(summary['hard_users'])}, "
            f"cold_item_id={summary['cold_item_id']}, "
            f"E_add_source={summary['E_add_source'].shape[1]}, "
            f"E_add_target={summary['E_add_target'].shape[1]}"
        )
        # 假設 split_result 已經從 link_split(data) 得到
        cold_item_id = args.cold_item_id
        debug_cold_item_counts(split_result, cold_item_id)

        def check_edge_index(edge_index, num_users, num_source_items, name):
            if edge_index is None or edge_index.numel() == 0:
                logging.info(f"[{name}] empty (skip check)")
                return

            u, v = edge_index
            min_u, max_u = u.min().item(), u.max().item()
            min_v, max_v = v.min().item(), v.max().item()
            logging.info(
                f"[{name}] users {min_u}~{max_u} (limit {num_users-1}), "
                f"items {min_v}~{max_v}, valid user∈[0,{num_users-1}], item∈[0,{num_source_items-1}]"
            )

        check_edge_index(summary["E_add_source"], args.num_users, args.num_source_items, "E_add_source")
        check_edge_index(summary["E_add_target"], args.num_users, args.num_target_items, "E_add_target")

        # merge ΔE 回 split_result
        if summary["E_add_source"].numel() > 0:
            split_result["source_train_edge_index"] = torch.cat(
                [split_result["source_train_edge_index"], summary["E_add_source"]], dim=1
            )
        if summary["E_add_target"].numel() > 0:
            split_result["target_train_edge_index"] = torch.cat(
                [split_result["target_train_edge_index"], summary["E_add_target"]], dim=1
            )

    # === 建立 BiGNAS 模型並訓練 ===
    model = Model(args)
    perceptor = Perceptor(args)
    logging.info(f"model: {model}")

    train(model, perceptor, data, args, split_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device & mode settings
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--search", default=False, action="store_true")
    parser.add_argument("--use-meta", default=False, action="store_true")
    parser.add_argument("--use-source", default=False, action="store_true")

    # dataset settings
    parser.add_argument("--categories", type=str, nargs="+", default=["CD", "Kitchen"])
    parser.add_argument("--target", type=str, default="Kitchen")
    parser.add_argument("--root", type=str, default="data/")

    # model settings
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--model-dir", type=str, default="./save/")

    # supernet settings
    parser.add_argument("--space", type=str, nargs="+",
                        default=["gcn", "gatv2", "sage", "lightgcn", "linear"])
    parser.add_argument("--warm-up", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--entropy", type=float, default=0.0)

    # training settings
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eta-min", type=float, default=0.001)
    parser.add_argument("--T-max", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=15,
                        help="Top-K for hit ratio evaluation")

    # meta settings
    parser.add_argument("--meta-interval", type=int, default=50)
    parser.add_argument("--meta-num-layers", type=int, default=2)
    parser.add_argument("--meta-hidden-dim", type=int, default=32)
    parser.add_argument("--meta-batch-size", type=int, default=512)
    parser.add_argument("--conv-lr", type=float, default=1)
    parser.add_argument("--hpo-lr", type=float, default=0.01)
    parser.add_argument("--descent-step", type=int, default=10)
    parser.add_argument("--meta-op", type=str, default="gat")

    # CL超參數
    parser.add_argument('--ssl_aug_type', type=str, default='edge', choices=['edge', 'node'])
    parser.add_argument('--edge_drop_rate', type=float, default=0.2)
    parser.add_argument('--node_drop_rate', type=float, default=0.2)
    parser.add_argument('--ssl_reg', type=float, default=0.1)
    parser.add_argument('--reg', type=float, default=1e-4)
    parser.add_argument('--nce_temp', type=float, default=0.2)
    parser.add_argument('--hard_ratio', type=float, default=0.1)
    parser.add_argument('--hard_mine_interval', type=int, default=1)
    parser.add_argument('--inject_source', action='store_true')
    parser.add_argument('--inject_target', action='store_true')
    parser.add_argument('--neg_samples', type=int, default=1)

    # HardUser 參數
    parser.add_argument("--use-hard-user-augment", action="store_true",
                        help="啟用 Hard User 加邊（方法A）")
    parser.add_argument("--hard-top-ratio", type=float, default=0.10,
                        help="Hard Users 的比例（Group B 中距離最大前x%）")
    parser.add_argument("--cold-item-id", type=int, default=-1,
                        help="指定 target 冷門 item；<0 自動由 train split 找最冷")
    parser.add_argument("--edge-ratio-source", type=float, default=0.1,
                        help="Source domain 要加邊的比例 (0~1)")
    parser.add_argument("--edge-ratio-target", type=float, default=1.0,
                        help="Target domain 要加邊的比例 (0~1)")

    # 讀 target domain 的 SGL user embedding（只需 user）
    parser.add_argument("--sgl-dir-target", type=str,
        default="/mnt/sda1/sherry/SGL-BiGNAS/SGL-Torch/dataset/amazon/pretrain-embeddings/SGL/n_layers=3",
        help="target domain 的 SGL 輸出資料夾，內含 user_embeddings_final.npy")

    args = parser.parse_args()
    search(args)
