# sgl2bignas_adapter.py
import logging
from typing import Optional

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"expected np.ndarray or torch.Tensor, got {type(x)}")


def load_sgl_final_embeddings(sgl_dir: str):
    """
    載入 SGL 訓練後的最終（傳播後）embedding，並做 L2 normalize。
    回傳 numpy，避免後續混用 dtype/device。
    """
    sgl_dir = sgl_dir if sgl_dir.endswith("/") else sgl_dir + "/"
    user_file = sgl_dir + "user_embeddings_final.npy"
    item_file = sgl_dir + "item_embeddings_final.npy"

    try:
        E_s_u = np.load(user_file)
        E_s_i = np.load(item_file)

        # L2 normalize（單位向量）
        E_s_u = E_s_u / (np.linalg.norm(E_s_u, axis=1, keepdims=True) + 1e-12)
        E_s_i = E_s_i / (np.linalg.norm(E_s_i, axis=1, keepdims=True) + 1e-12)

        logging.info(
            f"[SGL Adapter] Loaded embeddings: "
            f"users={E_s_u.shape}, items={E_s_i.shape} from {sgl_dir}"
        )
        return E_s_u, E_s_i
    except FileNotFoundError as e:
        logging.error(f"[SGL Adapter] Embeddings not found in {sgl_dir}")
        logging.error(
            "Please ensure SGL training completed and export_final_embeddings() was called"
        )
        raise e


def _maybe_remap_by_index(
    emb_mat_np: np.ndarray,
    index_map: Optional[np.ndarray],
    take_k: Optional[int] = None,
) -> np.ndarray:
    """
    依照 index_map 重排；若未提供 index_map，則以切片前 k 筆方式對齊。
    """
    if index_map is not None:
        assert index_map.ndim == 1, "index_map must be 1D"
        assert index_map.max() < emb_mat_np.shape[0], "index_map out of range"
        return emb_mat_np[index_map]
    if take_k is None:
        return emb_mat_np
    return emb_mat_np[:take_k]


def _project_if_needed(E_np: np.ndarray, d_target: int, device: torch.device) -> np.ndarray:
    """
    若 SGL 維度 != BiGNAS 維度，臨時用一個 Linear 做投影（不掛在 model 上）。
    回傳 numpy。
    """
    d_src = E_np.shape[1]
    if d_src == d_target:
        return E_np
    proj = torch.nn.Linear(d_src, d_target, bias=False).to(device)
    with torch.no_grad():
        E_t = torch.from_numpy(E_np).to(device=device, dtype=torch.float32)
        E_proj = proj(E_t).cpu().numpy()
    logging.info(f"[SGL Adapter] Projected embeddings from dim {d_src} -> {d_target}")
    return E_proj


def init_bignas_source_from_sgl(
    model: torch.nn.Module,
    data,
    E_s_u_np: np.ndarray,
    E_s_i_np: np.ndarray,
    device: torch.device,
    freeze_steps: int = 1000,
    user_index_map: Optional[np.ndarray] = None,
    source_item_index_map: Optional[np.ndarray] = None,
):
    """
    用 SGL 的 user/item 最終向量初始化 BiGNAS 的 user / source_item embedding。
    """
    # 1) 對齊維度
    d_target = int(model.user_embedding.embedding_dim)
    E_s_u_np = _project_if_needed(E_s_u_np, d_target, device)
    E_s_i_np = _project_if_needed(E_s_i_np, d_target, device)

    # 2) 依照 index_map 或切片對齊尺寸
    n_users = int(data.num_users)
    n_src_items = int(data.num_source_items)
    E_u_use = _maybe_remap_by_index(E_s_u_np, user_index_map, take_k=n_users)
    E_i_use = _maybe_remap_by_index(E_s_i_np, source_item_index_map, take_k=n_src_items)

    if E_u_use.shape[0] < n_users:
        logging.warning(
            f"[SGL Adapter] Users truncated: {E_u_use.shape[0]} < {n_users}"
        )
    if E_i_use.shape[0] < n_src_items:
        logging.warning(
            f"[SGL Adapter] Items truncated: {E_i_use.shape[0]} < {n_src_items}"
        )

    # 3) 覆寫 BiGNAS 權重
    with torch.no_grad():
        u_tensor = torch.from_numpy(E_u_use).to(device=model.user_embedding.weight.device,
                                               dtype=model.user_embedding.weight.dtype)
        model.user_embedding.weight[: u_tensor.size(0)].copy_(u_tensor)

        i_tensor = torch.from_numpy(E_i_use).to(device=model.source_item_embedding.weight.device,
                                               dtype=model.source_item_embedding.weight.dtype)
        model.source_item_embedding.weight[: i_tensor.size(0)].copy_(i_tensor)

    # 4) 註冊錨點
    model.register_buffer(
        "_sgl_user_anchor",
        torch.from_numpy(E_u_use).to(device=device, dtype=torch.float32),
        persistent=False,
    )

    # 5) 凍結來源 item
    model._sgl_freeze_counter = int(freeze_steps)
    for p in model.source_item_embedding.parameters():
        p.requires_grad = False

    logging.info(
        f"[SGL Adapter] Init done: users={E_u_use.shape}, items={E_i_use.shape}, "
        f"freeze_src_steps={freeze_steps}"
    )
    return model


def step_unfreeze_if_ready(model: torch.nn.Module):
    """每個 iteration 呼叫一次，解凍來源 item"""
    if hasattr(model, "_sgl_freeze_counter") and model._sgl_freeze_counter is not None:
        if model._sgl_freeze_counter > 0:
            model._sgl_freeze_counter -= 1
        if model._sgl_freeze_counter == 0:
            for p in model.source_item_embedding.parameters():
                p.requires_grad = True
            logging.info("[SGL Adapter] Unfroze source embeddings")
            model._sgl_freeze_counter = None


def load_user_alignment(csv_path: str):
    """
    載入跨域使用者對應表（如果有的話）
    CSV: source_uid,target_uid
    """
    try:
        pairs = np.loadtxt(csv_path, delimiter=",", dtype=np.int64, skiprows=1)
        src_u = torch.from_numpy(pairs[:, 0]).long()
        tgt_u = torch.from_numpy(pairs[:, 1]).long()
        logging.info(f"[SGL Adapter] Loaded {len(src_u)} user alignments from {csv_path}")
        return src_u, tgt_u
    except Exception as e:
        logging.warning(f"[SGL Adapter] Failed to load user alignment: {e}")
        return None, None
