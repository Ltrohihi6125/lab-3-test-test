"""
utils.py
--------
Các hàm tiện ích cho HPNC:
    - Tải dữ liệu đồ thị (Cora, CiteSeer, PubMed, ACM, DBLP)
"""

import os
import os.path as osp
from typing import Callable, List, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.io import read_txt_array

class SDCNDataset(InMemoryDataset):
    """
    Dataset class tự động tải và xử lý dữ liệu ACM, DBLP 
    từ GitHub repository của bài báo SDCN.
    """
    url = 'https://github.com/bdy9527/SDCN/raw/master'

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        
        # Thêm weights_only=False vào dòng này để PyTorch 2.6+ cho phép load object Data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        # Tên các file mong đợi sau khi tải về thư mục raw/
        return[f'{self.name}.txt', f'{self.name}_label.txt', f'{self.name}_graph.txt']

    @property
    def processed_file_names(self) -> str:
        # Tên file tensor sau khi process xong
        return 'data.pt'

    def download(self):
        """Tải dữ liệu trực tiếp từ GitHub"""
        # Node features và Labels nằm ở thư mục /data/
        download_url(f'{self.url}/data/{self.name}.txt', self.raw_dir)
        download_url(f'{self.url}/data/{self.name}_label.txt', self.raw_dir)
        # Edges nằm ở thư mục /graph/
        download_url(f'{self.url}/graph/{self.name}_graph.txt', self.raw_dir)

    def process(self):
        """Đọc file txt đã tải, chuyển thành Tensor và lưu vào thư mục processed/"""
        # 1. Đọc node features
        x = read_txt_array(self.raw_paths[0], dtype=torch.float)
        
        # 2. Đọc labels
        y = read_txt_array(self.raw_paths[1], dtype=torch.long)
        if y.dim() > 1:
            y = y.squeeze() # Chuyển về 1D array
        if y.min() > 0:
            y = y - y.min() # Đảm bảo label bắt đầu từ 0
            
        # 3. Đọc edges (ma trận kề)
        edge_index = read_txt_array(self.raw_paths[2], dtype=torch.long).t().contiguous()

        # 4. Gói gọn vào object Data
        data = Data(x=x, edge_index=edge_index, y=y)
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
            
        # 5. Lưu xuống file data.pt
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.upper()}()'


def load_dataset(name: str, data_dir: str = './data') -> Data:
    """
    Tải dataset đồ thị theo tên. Hỗ trợ: Cora, CiteSeer, PubMed, ACM, DBLP.

    Parameters
    ----------
    name : str
        Tên dataset.
    data_dir : str
        Thư mục lưu dữ liệu.

    Returns
    -------
    Data
        PyTorch Geometric Data.
    """
    os.makedirs(data_dir, exist_ok=True)
    name_lower = name.lower()
    
    if name_lower in['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=data_dir, name=name, transform=NormalizeFeatures())
        data = dataset[0]
        
    elif name_lower in['acm', 'dblp']:
        # Đặt thư mục root cho dataset này, ví dụ: ./data/ACM
        root_dir = osp.join(data_dir, name.upper())
        dataset = SDCNDataset(root=root_dir, name=name_lower, transform=NormalizeFeatures())
        data = dataset[0]
        
    else:
        raise ValueError(f"Dataset '{name}' không được hỗ trợ. Chọn: Cora, CiteSeer, PubMed, ACM, DBLP.")
        
    return data