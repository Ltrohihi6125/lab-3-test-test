"""
model.py
--------
Cài đặt lại Hyperspherical Prototype Node Clustering (HPNC) từ đầu
dựa trên mô tả trong bài báo:

    "Hyperspherical Prototype Node Clustering"
    Transactions on Machine Learning Research (01/2024)

Các thành phần chính:
    1. GATEncoder        – Encoder dùng Graph Attention Network (GAT)
    2. GATDecoder        – Decoder tái tạo node features
    3. HypersphericalPrototypes – Prototype được khởi tạo và cố định trên hypersphere
    4. HPNC_IM           – Scheme 1: dùng Information Maximization loss
    5. HPNC_DEC          – Scheme 2: dùng DEC loss

Không sao chép mã nguồn công khai của tác giả.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ──────────────────────────────────────────────────────────────────────
# 1. GAT Encoder (Section 3.2.2)
# ──────────────────────────────────────────────────────────────────────

class GATEncoder(nn.Module):
    """
    Encoder sử dụng 2 lớp Graph Attention Network (GAT).

    Theo bài báo (Appendix A.3):
        - 2 lớp GAT, mỗi lớp có 4 attention heads, mỗi head 128 chiều
        - Dropout 0.1 cho attention coefficients
        - Dropout 0.2 giữa các lớp
        - Activation: PReLU
        - Output dimension: m (embedding dimension)

    Parameters
    ----------
    in_channels : int
        Số chiều của node features đầu vào.
    hidden_channels : int
        Số chiều ẩn mỗi attention head (mặc định = 128).
    out_channels : int
        Số chiều output embedding (m).
    heads : int
        Số attention heads (mặc định = 4).
    dropout_attn : float
        Dropout rate cho attention coefficients (mặc định = 0.1).
    dropout_feat : float
        Dropout rate giữa các lớp GAT (mặc định = 0.2).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 256, heads: int = 4,
                 dropout_attn: float = 0.1, dropout_feat: float = 0.2):
        super().__init__()

        # Lớp GAT 1: in_channels -> hidden_channels * heads
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout_attn,
            concat=True  # concatenate heads => output: hidden_channels * heads
        )

        # Lớp GAT 2: hidden_channels * heads -> out_channels (single head)
        self.gat2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            dropout=dropout_attn,
            concat=False  # BẮT BUỘC không concatenate để output ra đúng 512 chiều
        )

        self.prelu1 = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout_feat)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của encoder.

        Parameters
        ----------
        x : torch.Tensor, shape (n, in_channels)
            Node feature matrix (có thể bị mask).
        edge_index : torch.Tensor, shape (2, E)
            Edge indices.

        Returns
        -------
        torch.Tensor, shape (n, out_channels)
            Node embeddings Z.
        """
        # GAT layer 1
        h = self.gat1(x, edge_index)
        h = self.prelu1(h)
        h = self.dropout(h)

        # GAT layer 2
        z = self.gat2(h, edge_index)

        return z  # shape: (n, out_channels)


# ──────────────────────────────────────────────────────────────────────
# 2. GAT Decoder (Section 3.2.2)
# ──────────────────────────────────────────────────────────────────────

class GATDecoder(nn.Module):
    """
    Decoder dùng 1 lớp GAT để tái tạo node features từ embeddings.

    Theo bài báo: "A decoder is a single GAT layer without non-linear activation."

    Parameters
    ----------
    in_channels : int
        Số chiều embedding đầu vào (m).
    out_channels : int
        Số chiều output = số chiều feature gốc (d).
    heads : int
        Số attention heads.
    dropout_attn : float
        Dropout rate cho attention.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 1, dropout_attn: float = 0.1):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout_attn,
            concat=False
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Tái tạo node features.

        Parameters
        ----------
        z : torch.Tensor, shape (n, m)
            Node embeddings.
        edge_index : torch.Tensor, shape (2, E)
            Edge indices.

        Returns
        -------
        torch.Tensor, shape (n, d)
            Node features được tái tạo.
        """
        return self.gat(z, edge_index)  # Không có non-linear activation


# ──────────────────────────────────────────────────────────────────────
# 3. Hyperspherical Prototypes (Section 3.2.1)
# ──────────────────────────────────────────────────────────────────────

class HypersphericalPrototypes(nn.Module):
    """
    Khởi tạo và huấn luyện cluster prototypes trên unit-hypersphere.

    Theo bài báo (Section 3.2.1):
        - Prototype được khởi tạo ngẫu nhiên
        - Được tối ưu hóa để phân tán đều trên hypersphere bằng cách
          minimize cosine similarity lớn nhất giữa các cặp prototype (Eq. 2)
        - Sau khi pretrain, prototype được CỐ ĐỊNH (không update nữa)

    Parameters
    ----------
    num_clusters : int
        Số lượng cụm (c).
    embed_dim : int
        Số chiều embedding (m).
    """

    def __init__(self, num_clusters: int, embed_dim: int):
        super().__init__()
        self.num_clusters = num_clusters
        self.embed_dim = embed_dim

        # Prototype vectors (unnormalized, sẽ được normalize khi dùng)
        self.mu = nn.Parameter(torch.randn(num_clusters, embed_dim))

    def pretrain(self, num_epochs: int = 3000, lr: float = 0.01,
                 device: torch.device = torch.device('cpu'),
                 verbose: bool = False) -> None:
        """
        Pretrain prototype bằng gradient descent để phân tán đều trên hypersphere.

        Minimize L_HP = (1/c) * sum_i max_{j≠i} cosine_sim(mu_i, mu_j)  [Eq. 2]

        Sau khi hoàn thành, prototype được đóng băng (requires_grad = False).

        Parameters
        ----------
        num_epochs : int
            Số epoch pretrain (mặc định 3000 như bài báo).
        lr : float
            Learning rate.
        device : torch.device
            Thiết bị tính toán.
        verbose : bool
            In loss mỗi 500 epoch nếu True.
        """
        self.to(device)
        optimizer = torch.optim.Adam([self.mu], lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self._prototype_loss()
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 500 == 0:
                print(f"  Prototype pretrain epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")

        # Cố định prototype sau khi pretrain
        self.mu.requires_grad_(False)

        if verbose:
            print("  Prototype pretrain hoàn thành. Prototype đã được cố định.")

    def _prototype_loss(self) -> torch.Tensor:
        """
        Tính hyperspherical prototype loss (Eq. 2).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        mu_norm = F.normalize(self.mu, p=2, dim=1)  # (c, m)
        sim = mu_norm @ mu_norm.t()  # (c, c)

        # Loại diagonal
        mask = torch.eye(self.num_clusters, dtype=torch.bool, device=self.mu.device)
        sim = sim.masked_fill(mask, float('-inf'))

        max_sim = sim.max(dim=1).values  # (c,)
        return max_sim.mean()

    @property
    def normalized_prototypes(self) -> torch.Tensor:
        """
        Trả về L2-normalized prototype vectors.

        Returns
        -------
        torch.Tensor, shape (c, m)
            Prototype đã normalize.
        """
        return F.normalize(self.mu, p=2, dim=1)


# ──────────────────────────────────────────────────────────────────────
# 4. Masked Feature Reconstruction Helper (Section 3.2.2 – GraphMAE)
# ──────────────────────────────────────────────────────────────────────

class MaskedGraphAutoencoder(nn.Module):
    """
    Masked Graph Autoencoder theo GraphMAE (Hou et al., 2022) được tích hợp trong HPNC.

    Thực hiện:
        1. Randomly mask một subset node features bằng mask token trainable
        2. Encode bằng GATEncoder để ra embeddings Z
        3. Re-mask embeddings của masked nodes bằng zeros trước khi decode
        4. Decode bằng GATDecoder để tái tạo features

    Parameters
    ----------
    in_channels : int
        Số chiều node features (d).
    hidden_channels : int
        Số chiều ẩn mỗi head trong GAT.
    out_channels : int
        Số chiều embedding (m).
    heads : int
        Số attention heads.
    mask_ratio : float
        Tỷ lệ node bị mask (mặc định 0.5 = 50% như bài báo).
    remask_ratio : float
        Tỷ lệ masked nodes được thay bằng random features thay vì mask token
        (mặc định 0.15 = 15% theo "random-substitution" trong bài báo).
    dropout_attn : float
        Dropout attention.
    dropout_feat : float
        Dropout giữa các layer.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 256, heads: int = 4,
                 mask_ratio: float = 0.5, remask_ratio: float = 0.15,
                 dropout_attn: float = 0.1, dropout_feat: float = 0.2):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.remask_ratio = remask_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Learnable mask token (shared across all masked nodes)
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels))

        # Encoder và Decoder
        self.encoder = GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout_attn=dropout_attn,
            dropout_feat=dropout_feat
        )

        self.decoder = GATDecoder(
            in_channels=out_channels,
            out_channels=in_channels,
            heads=1,
            dropout_attn=dropout_attn
        )

    def _apply_mask(self, x: torch.Tensor) -> tuple:
        """
        Áp dụng masking strategy lên node features.

        Parameters
        ----------
        x : torch.Tensor, shape (n, d)
            Node features gốc.

        Returns
        -------
        tuple:
            - x_masked: features sau khi mask (n, d)
            - mask_indices: boolean mask chỉ các node bị mask (n,)
        """
        n = x.size(0)
        num_mask = max(1, int(n * self.mask_ratio))

        # Random shuffle để chọn masked nodes
        perm = torch.randperm(n, device=x.device)
        mask_idx = perm[:num_mask]

        x_masked = x.clone()

        # "Random-substitution": 15% masked nodes dùng random feature thay mask token
        num_random = max(0, int(num_mask * self.remask_ratio))
        random_idx = mask_idx[:num_random]
        token_idx = mask_idx[num_random:]

        # Thay bằng features ngẫu nhiên từ tập dữ liệu
        if num_random > 0:
            rand_perm = torch.randperm(n, device=x.device)[:num_random]
            x_masked[random_idx] = x[rand_perm]

        # Thay bằng mask token (trainable)
        x_masked[token_idx] = self.mask_token.expand(token_idx.size(0), -1)

        # Tạo boolean mask
        bool_mask = torch.zeros(n, dtype=torch.bool, device=x.device)
        bool_mask[mask_idx] = True

        return x_masked, bool_mask

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor,
               training: bool = True) -> tuple:
        """
        Encode node features thành embeddings.

        Parameters
        ----------
        x : torch.Tensor, shape (n, d)
        edge_index : torch.Tensor, shape (2, E)
        training : bool
            Nếu True, áp dụng masking. Nếu False (inference), không mask.

        Returns
        -------
        tuple:
            - z: embeddings (n, m)
            - mask: boolean mask (n,) – None khi inference
        """
        if training:
            x_masked, mask = self._apply_mask(x)
            z = self.encoder(x_masked, edge_index)
        else:
            z = self.encoder(x, edge_index)
            mask = None
        return z, mask

    def decode_features(self, z: torch.Tensor, edge_index: torch.Tensor,
                        mask: torch.Tensor = None) -> torch.Tensor:
        """
        Decode embeddings để tái tạo node features.
        Áp dụng re-mask (zero out embeddings của masked nodes) trước khi decode.

        Parameters
        ----------
        z : torch.Tensor, shape (n, m)
        edge_index : torch.Tensor, shape (2, E)
        mask : torch.Tensor or None
            Boolean mask chỉ masked nodes.

        Returns
        -------
        torch.Tensor, shape (n, d)
            Reconstructed features.
        """
        z_remask = z.clone()
        if mask is not None:
            z_remask[mask] = 0.0  # Re-mask strategy: zero out masked node embeddings

        x_hat = self.decoder(z_remask, edge_index)
        return x_hat

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                training: bool = True) -> tuple:
        """
        Full forward pass: encode + decode.

        Parameters
        ----------
        x : torch.Tensor, shape (n, d)
        edge_index : torch.Tensor, shape (2, E)
        training : bool

        Returns
        -------
        tuple:
            - z: embeddings (n, m)
            - x_hat: reconstructed features (n, d)
            - mask: boolean mask (n,)
        """
        z, mask = self.encode(x, edge_index, training=training)
        x_hat = self.decode_features(z, edge_index, mask)
        return z, x_hat, mask


# ──────────────────────────────────────────────────────────────────────
# 5. Rotated Clustering Affinity (Section 3.2.3)
# ──────────────────────────────────────────────────────────────────────

class RotatedClusteringAffinity(nn.Module):
    """
    Tính soft clustering labels Q bằng cosine similarity giữa node embeddings
    và prototype được xoay bởi một rotation matrix R (Eq. 6).

    Q_{i,j} = softmax(z_i^T R mu_j)   subject to R^T R = I

    Rotation matrix R cho phép prototype xoay trên hypersphere để dễ match với
    semantic centroids của node embeddings, mà không làm thay đổi pairwise distances.

    Parameters
    ----------
    embed_dim : int
        Số chiều embedding (m).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # Learnable rotation matrix với orthogonal constraint R^T R = I.
        # Theo bài báo (Section 3.2.3):
        # "We initialize R as an identity matrix and enforce the orthogonal
        #  constraint with torch.nn.utils.parametrizations.orthogonal."
        #
        # torch.nn.utils.parametrizations.orthogonal(module) áp dụng
        # orthogonal constraint lên weight của module và trả về module đã
        # được parametrize. Ta lưu nó như submodule bình thường.
        _R_base = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(_R_base.weight)
        self._R_linear = torch.nn.utils.parametrizations.orthogonal(_R_base)

    def get_rotation_matrix(self) -> torch.Tensor:
        """
        Lấy rotation matrix R hiện tại (đã enforce orthogonal, R^T R = I).

        Returns
        -------
        torch.Tensor, shape (m, m)
        """
        return self._R_linear.weight

    def forward(self, z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Tính soft clustering assignments Q (Eq. 6).

        Parameters
        ----------
        z : torch.Tensor, shape (n, m)
            Node embeddings (chưa normalize).
        prototypes : torch.Tensor, shape (c, m)
            L2-normalized prototype vectors (cố định sau pretrain).

        Returns
        -------
        torch.Tensor, shape (n, c)
            Soft cluster assignment matrix Q.
        """
        # L2-normalize node embeddings
        z_norm = F.normalize(z, p=2, dim=1)  # (n, m)

        # Rotate prototypes: R mu_j
        R = self.get_rotation_matrix()  # (m, m)
        rotated_proto = prototypes @ R.t()  # (c, m) @ (m, m) => (c, m)

        # Inner product: z̃_i^T (R μ̃_j) — đúng theo Eq. 6
        logits = z_norm @ rotated_proto.t()  # (n, c)

        # Softmax theo Eq. 6
        Q = F.softmax(logits, dim=1)  # (n, c)
        return Q


# ──────────────────────────────────────────────────────────────────────
# 6. HPNC – Scheme 1: Information Maximization (Section 3.3)
# ──────────────────────────────────────────────────────────────────────

class HPNC_IM(nn.Module):
    """
    HPNC với Information Maximization (IM) loss (Scheme 1).

    Full objective (Eq. 10):
        L_HPNC-IM = L_fea + alpha * L_edge - beta * L_bal + gamma * L_ent

    Tất cả huấn luyện end-to-end từ đầu (không cần pretrain encoder riêng).

    Parameters
    ----------
    in_channels : int
        Số chiều node features (d).
    hidden_channels : int
        Số chiều ẩn mỗi GAT head (mặc định 128).
    embed_dim : int
        Số chiều embedding/prototype (m).
    num_clusters : int
        Số cụm (c).
    heads : int
        Số attention heads (mặc định 4).
    mask_ratio : float
        Tỷ lệ node bị mask.
    remask_ratio : float
        Tỷ lệ random-substitution trong masked nodes.
    dropout_attn : float
        Dropout attention.
    dropout_feat : float
        Dropout giữa layers.
    alpha : float
        Hệ số cho edge reconstruction loss.
    beta : float
        Hệ số cho L_bal (maximize marginal entropy).
    gamma : float
        Hệ số cho L_ent (minimize conditional entropy).
    gamma_cos : float
        Exponent trong scaled cosine error (>=1, mặc định 2).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 embed_dim: int = 256, num_clusters: int = 7,
                 heads: int = 4, mask_ratio: float = 0.5, remask_ratio: float = 0.15,
                 dropout_attn: float = 0.1, dropout_feat: float = 0.2,
                 alpha: float = 0.01, beta: float = 0.01, gamma: float = 0.01,
                 gamma_cos: float = 2.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gamma_cos = gamma_cos
        self.num_clusters = num_clusters

        # Masked Graph Autoencoder
        self.mgae = MaskedGraphAutoencoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=embed_dim,
            heads=heads,
            mask_ratio=mask_ratio,
            remask_ratio=remask_ratio,
            dropout_attn=dropout_attn,
            dropout_feat=dropout_feat
        )

        # Hyperspherical Prototypes (sẽ được pretrain riêng)
        self.prototypes = HypersphericalPrototypes(
            num_clusters=num_clusters,
            embed_dim=embed_dim
        )

        # Rotated Clustering Affinity
        self.affinity = RotatedClusteringAffinity(embed_dim=embed_dim)

    def pretrain_prototypes(self, num_epochs: int = 3000, lr: float = 0.01,
                            device: torch.device = torch.device('cpu'),
                            verbose: bool = True) -> None:
        """Pretrain prototype bằng gradient descent."""
        self.prototypes.pretrain(num_epochs=num_epochs, lr=lr,
                                 device=device, verbose=verbose)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                neg_edge_index: torch.Tensor = None,
                training: bool = True) -> dict:
        """
        Forward pass của HPNC-IM.

        Parameters
        ----------
        x : torch.Tensor, shape (n, d)
        edge_index : torch.Tensor, shape (2, E)
        neg_edge_index : torch.Tensor or None, shape (2, E_neg)
        training : bool

        Returns
        -------
        dict với các keys:
            'z': node embeddings (n, m)
            'z_norm': L2-normalized embeddings (n, m)
            'Q': soft assignments (n, c)
            'x_hat': reconstructed features (n, d)
            'mask': boolean mask (n,)
            'loss': tổng loss
            'loss_fea', 'loss_edge', 'loss_bal', 'loss_ent': thành phần loss
        """
        # 1. Encode + Decode
        z, x_hat, mask = self.mgae(x, edge_index, training=training)

        # 2. L2-normalize embeddings để đưa lên hypersphere
        z_norm = F.normalize(z, p=2, dim=1)

        # 3. Tính soft clustering assignments Q (Eq. 6)
        proto = self.prototypes.normalized_prototypes.detach()  # (c, m), cố định
        Q = self.affinity(z, proto)  # (n, c)

        if not training:
            return {'z': z, 'z_norm': z_norm, 'Q': Q,
                    'x_hat': x_hat, 'mask': mask}

        # 4. Tính losses
        # L_fea: Scaled cosine error trên masked nodes (Eq. 3)
        loss_fea = self._cosine_feature_loss(x, x_hat, mask)

        # L_edge: Edge reconstruction BCE (Eq. 5)
        if neg_edge_index is not None:
            loss_edge = self._edge_loss(z, edge_index, neg_edge_index)
        else:
            loss_edge = torch.tensor(0.0, device=x.device)

        # L_IM: Information Maximization (Eq. 7-9)
        L_bal, L_ent = self._im_loss(Q)

        # Full objective (Eq. 10): L = L_fea + alpha*L_edge - beta*L_bal + gamma*L_ent
        loss = loss_fea + self.alpha * loss_edge - self.beta * L_bal + self.gamma * L_ent

        return {
            'z': z, 'z_norm': z_norm, 'Q': Q,
            'x_hat': x_hat, 'mask': mask,
            'loss': loss,
            'loss_fea': loss_fea.item(),
            'loss_edge': loss_edge.item(),
            'loss_bal': L_bal.item(),
            'loss_ent': L_ent.item()
        }

    def _cosine_feature_loss(self, x, x_hat, mask, gamma=None):
        if gamma is None:
            gamma = self.gamma_cos
        if mask is None or mask.sum() == 0:
            return torch.tensor(0.0, device=x.device)
        cos_sim = F.cosine_similarity(x[mask], x_hat[mask], dim=1)
        return (1.0 - cos_sim).pow(gamma).mean()

    def _edge_loss(self, z, edge_index, neg_edge_index):
        z_norm = F.normalize(z, p=2, dim=1)
        u, v = edge_index[0], edge_index[1]
        pos_score = (z_norm[u] * z_norm[v]).sum(dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()

        u_n, v_n = neg_edge_index[0], neg_edge_index[1]
        neg_score = (z_norm[u_n] * z_norm[v_n]).sum(dim=1)
        neg_loss = -torch.log(1.0 - torch.sigmoid(neg_score) + 1e-8).mean()

        return pos_loss + neg_loss

    def _im_loss(self, Q):
        eps = 1e-8
        # L_bal = H(mean_Q) => maximize
        q_mean = Q.mean(dim=0)
        L_bal = -(q_mean * torch.log(q_mean + eps)).sum()
        # L_ent = mean H(Q_i) => minimize
        L_ent = -(Q * torch.log(Q + eps)).sum(dim=1).mean()
        return L_bal, L_ent

    @torch.no_grad()
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Dự đoán nhãn cụm bằng arg max của Q (không cần K-means).

        Parameters
        ----------
        x, edge_index: inputs

        Returns
        -------
        torch.Tensor, shape (n,)
            Cluster labels.
        """
        self.eval()
        out = self.forward(x, edge_index, training=False)
        return out['Q'].argmax(dim=1)


# ──────────────────────────────────────────────────────────────────────
# 7. HPNC – Scheme 2: DEC Loss (Section 3.4)
# ──────────────────────────────────────────────────────────────────────

class HPNC_DEC(nn.Module):
    """
    HPNC với DEC loss (Scheme 2).

    Full objective (Eq. 14):
        L_HPNC-DEC = L_fea + alpha * L_edge - beta * L_bal + gamma * L_DEC

    DEC sharpen Q bằng cách minimize KL(P||Q) với P là auxiliary distribution.
    L_bal ngăn empty clusters.

    Parameters
    ----------
    Giống HPNC_IM, với alpha/beta/gamma tương ứng.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 embed_dim: int = 256, num_clusters: int = 7,
                 heads: int = 4, mask_ratio: float = 0.5, remask_ratio: float = 0.15,
                 dropout_attn: float = 0.1, dropout_feat: float = 0.2,
                 alpha: float = 0.01, beta: float = 0.01, gamma: float = 0.01,
                 gamma_cos: float = 2.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gamma_cos = gamma_cos
        self.num_clusters = num_clusters

        self.mgae = MaskedGraphAutoencoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=embed_dim,
            heads=heads,
            mask_ratio=mask_ratio,
            remask_ratio=remask_ratio,
            dropout_attn=dropout_attn,
            dropout_feat=dropout_feat
        )

        self.prototypes = HypersphericalPrototypes(
            num_clusters=num_clusters,
            embed_dim=embed_dim
        )

        self.affinity = RotatedClusteringAffinity(embed_dim=embed_dim)

    def pretrain_prototypes(self, num_epochs: int = 3000, lr: float = 0.01,
                            device: torch.device = torch.device('cpu'),
                            verbose: bool = True) -> None:
        """Pretrain prototype."""
        self.prototypes.pretrain(num_epochs=num_epochs, lr=lr,
                                 device=device, verbose=verbose)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                neg_edge_index: torch.Tensor = None,
                training: bool = True) -> dict:
        """Forward pass của HPNC-DEC."""
        z, x_hat, mask = self.mgae(x, edge_index, training=training)
        z_norm = F.normalize(z, p=2, dim=1)

        proto = self.prototypes.normalized_prototypes.detach()
        Q = self.affinity(z, proto)

        if not training:
            return {'z': z, 'z_norm': z_norm, 'Q': Q,
                    'x_hat': x_hat, 'mask': mask}

        loss_fea = self._cosine_feature_loss(x, x_hat, mask)

        if neg_edge_index is not None:
            loss_edge = self._edge_loss(z, edge_index, neg_edge_index)
        else:
            loss_edge = torch.tensor(0.0, device=x.device)

        L_bal, L_dec = self._dec_loss(Q)

        # Eq. 14: L = L_fea + alpha*L_edge - beta*L_bal + gamma*L_DEC
        loss = loss_fea + self.alpha * loss_edge - self.beta * L_bal + self.gamma * L_dec

        return {
            'z': z, 'z_norm': z_norm, 'Q': Q,
            'x_hat': x_hat, 'mask': mask,
            'loss': loss,
            'loss_fea': loss_fea.item(),
            'loss_edge': loss_edge.item(),
            'loss_bal': L_bal.item(),
            'loss_dec': L_dec.item()
        }

    def _cosine_feature_loss(self, x, x_hat, mask, gamma=None):
        if gamma is None:
            gamma = self.gamma_cos
        if mask is None or mask.sum() == 0:
            return torch.tensor(0.0, device=x.device)
        cos_sim = F.cosine_similarity(x[mask], x_hat[mask], dim=1)
        return (1.0 - cos_sim).pow(gamma).mean()

    def _edge_loss(self, z, edge_index, neg_edge_index):
        z_norm = F.normalize(z, p=2, dim=1)
        u, v = edge_index[0], edge_index[1]
        pos_score = (z_norm[u] * z_norm[v]).sum(dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()
        u_n, v_n = neg_edge_index[0], neg_edge_index[1]
        neg_score = (z_norm[u_n] * z_norm[v_n]).sum(dim=1)
        neg_loss = -torch.log(1.0 - torch.sigmoid(neg_score) + 1e-8).mean()
        return pos_loss + neg_loss

    def _dec_loss(self, Q):
        eps = 1e-8
        q_mean = Q.mean(dim=0)
        L_bal = -(q_mean * torch.log(q_mean + eps)).sum()

        Q2 = Q.pow(2)
        freq = Q.sum(dim=0, keepdim=True) + eps
        P = Q2 / freq
        P = P / (P.sum(dim=1, keepdim=True) + eps)
        
        # [QUAN TRỌNG] Ngắt computational graph của P
        P = P.detach() 
        
        L_dec = (P * torch.log(P / (Q + eps) + eps)).sum(dim=1).mean()
        return L_bal, L_dec

    @torch.no_grad()
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Dự đoán nhãn cụm."""
        self.eval()
        out = self.forward(x, edge_index, training=False)
        return out['Q'].argmax(dim=1)
