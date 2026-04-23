"""
utils.py
--------
Các hàm tiện ích cho HPNC:
    - Tải dữ liệu đồ thị (Cora, CiteSeer, PubMed, ACM, DBLP)
"""

import os
import torch
from torch_geometric.datasets import Planetoid, AttributedGraphDataset
from torch_geometric.data import Data


# ──────────────────────────────────────────────
# 1. Tải dữ liệu
# ──────────────────────────────────────────────

def load_dataset(name: str, data_dir: str = './data') -> Data:
    """
    Tải dataset đồ thị theo tên. Hỗ trợ: Cora, CiteSeer, PubMed, ACM, DBLP.

    Parameters
    ----------
    name : str
        Tên dataset ('Cora', 'CiteSeer', 'PubMed', 'ACM', 'DBLP').
    data_dir : str
        Thư mục lưu dữ liệu.

    Returns
    -------
    torch_geometric.data.Data
        Object chứa x (node features), edge_index, y (labels).
    """
    os.makedirs(data_dir, exist_ok=True)
    name_lower = name.lower()
    if name_lower in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=data_dir, name=name, transform=NormalizeFeatures())
        data = dataset[0]
    elif name_lower in ['acm', 'dblp']:
        dataset = AttributedGraphDataset(root=data_dir, name=name, transform=NormalizeFeatures())
        data = dataset[0]
    else:
        raise ValueError(
            f"Dataset '{name}' không được hỗ trợ. Chọn: Cora, CiteSeer, PubMed, ACM, DBLP."
        )
    return data
