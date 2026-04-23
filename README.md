# HPNC – Hyperspherical Prototype Node Clustering

Cài đặt lại từ đầu dựa trên mô tả trong bài báo:  
**"Hyperspherical Prototype Node Clustering"** – Transactions on Machine Learning Research (01/2024).

## Cấu trúc thư mục

```
Group_ID/
├── README.md
├── requirements.txt
├── paper/
│   └── paper.pdf
├── notebooks/
│   ├── 01_main_experiments.ipynb   # Tái hiện kết quả chính
│   ├── 02_ablation_study.ipynb     # Ablation study
│   └── 03_new_dataset.ipynb        # Thực nghiệm tập dữ liệu mới
├── src/
│   ├── model.py      # Cài đặt chính HPNC (GNN encoder, prototype, rotation)
│   ├── utils.py      # Tiện ích: load data, tính loss, v.v.
│   └── metrics.py    # Các độ đo: ACC, NMI, ARI
├── data/
│   └── (datasets tải tự động qua PyG)
└── docs/
    └── Report.pdf
```

## Hướng dẫn chạy

```bash
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Chạy notebook chính
jupyter notebook notebooks/01_main_experiments.ipynb
```

## Datasets

Cora, CiteSeer, PubMed, ACM, DBLP – tải tự động qua `torch_geometric.datasets`.

## Lưu ý

- Toàn bộ code được cài đặt **từ đầu** dựa trên mô tả trong bài báo.
- Không sao chép mã nguồn công khai của tác giả.
- `random_state` cố định = 42 để đảm bảo reproducibility.
