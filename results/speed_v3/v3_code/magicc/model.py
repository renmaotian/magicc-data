"""
MAGICC Multi-Branch Fusion Neural Network.

Architecture:
  K-mer Branch (~9,249 -> 256):
    Dense(4096) -> BN -> SiLU -> Drop(0.4)
    Dense(1024) -> BN -> SiLU -> Drop(0.2)
    Dense(256)  -> BN -> SiLU

  Assembly Branch (26 -> 64):
    Dense(128) -> BN -> SiLU -> Drop(0.2)
    Dense(64)  -> BN -> SiLU

  Fusion (320 -> 2):
    Concat(256 + 64 = 320)
    Dense(128) -> BN -> SiLU -> Drop(0.1)
    Dense(64)  -> SiLU
    Dense(2)   -> Completeness: Sigmoid*50+50 [50,100], Contamination: Sigmoid*100 [0,100]
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class KmerBranch(nn.Module):
    """K-mer feature processing branch: ~9,249 -> 256 dimensions."""

    def __init__(self, n_kmer_features: int = 9249, dropout1: float = 0.3,
                 dropout2: float = 0.2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_kmer_features, 4096),
            nn.BatchNorm1d(4096),
            nn.SiLU(),
            nn.Dropout(dropout1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(dropout2),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class AssemblyBranch(nn.Module):
    """Assembly statistics processing branch: 26 -> 64 dimensions."""

    def __init__(self, n_assembly_features: int = 26, dropout1: float = 0.2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_assembly_features, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class FusionHead(nn.Module):
    """Fusion and output head: 320 -> 2 dimensions.

    Output activation per task:
    - Completeness: Sigmoid * 50 + 50 -> [50, 100] (uses full sigmoid range)
    - Contamination: Sigmoid * 100 -> [0, 100]
    """

    def __init__(self, fusion_dim: int = 320, dropout1: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout1),
        )
        self.pre_output = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.output_linear = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fusion(x)
        x = self.pre_output(x)
        raw = self.output_linear(x)
        activated = self.sigmoid(raw)
        # Completeness: Sigmoid * 50 + 50 -> [50, 100]
        comp = activated[:, 0:1] * 50.0 + 50.0
        # Contamination: Sigmoid * 100 -> [0, 100]
        cont = activated[:, 1:2] * 100.0
        return torch.cat([comp, cont], dim=1)


class MAGICCModel(nn.Module):
    """
    Multi-Branch Fusion Neural Network for genome quality prediction.

    Predicts completeness (0-100%) and contamination (0-100%) from
    k-mer count features and assembly statistics.

    Parameters
    ----------
    n_kmer_features : int
        Number of k-mer input features (default: 9249).
    n_assembly_features : int
        Number of assembly statistics input features (default: 26).
    kmer_dropout1 : float
        Dropout rate for k-mer branch layer 1.
    kmer_dropout2 : float
        Dropout rate for k-mer branch layer 2.
    assembly_dropout1 : float
        Dropout rate for assembly branch layer 1.
    fusion_dropout1 : float
        Dropout rate for fusion layer 1.
    use_gradient_checkpointing : bool
        Whether to use gradient checkpointing to save GPU memory.
    """

    def __init__(
        self,
        n_kmer_features: int = 9249,
        n_assembly_features: int = 26,
        kmer_dropout1: float = 0.4,
        kmer_dropout2: float = 0.2,
        assembly_dropout1: float = 0.2,
        fusion_dropout1: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.n_kmer_features = n_kmer_features
        self.n_assembly_features = n_assembly_features
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.kmer_branch = KmerBranch(
            n_kmer_features=n_kmer_features,
            dropout1=kmer_dropout1,
            dropout2=kmer_dropout2,
        )
        self.assembly_branch = AssemblyBranch(
            n_assembly_features=n_assembly_features,
            dropout1=assembly_dropout1,
        )
        self.fusion_head = FusionHead(
            fusion_dim=256 + 64,  # kmer_embed + assembly_embed
            dropout1=fusion_dropout1,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal for SiLU activation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _checkpoint_kmer(self, x: torch.Tensor) -> torch.Tensor:
        """Run k-mer branch with optional gradient checkpointing."""
        if self.use_gradient_checkpointing and self.training:
            # checkpoint requires inputs to have requires_grad
            x = x.requires_grad_(True)
            return checkpoint(self.kmer_branch, x, use_reentrant=False)
        return self.kmer_branch(x)

    def _checkpoint_assembly(self, x: torch.Tensor) -> torch.Tensor:
        """Run assembly branch with optional gradient checkpointing."""
        if self.use_gradient_checkpointing and self.training:
            x = x.requires_grad_(True)
            return checkpoint(self.assembly_branch, x, use_reentrant=False)
        return self.assembly_branch(x)

    def forward(
        self,
        kmer_features: torch.Tensor,
        assembly_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        kmer_features : torch.Tensor
            K-mer count features, shape (batch_size, n_kmer_features).
        assembly_features : torch.Tensor
            Assembly statistics features, shape (batch_size, n_assembly_features).

        Returns
        -------
        torch.Tensor
            Predictions [completeness, contamination], shape (batch_size, 2).
            Values bounded in [0, 100].
        """
        # Process through specialized branches
        kmer_embed = self._checkpoint_kmer(kmer_features)
        assembly_embed = self._checkpoint_assembly(assembly_features)

        # Concatenate embeddings and produce output
        fused = torch.cat([kmer_embed, assembly_embed], dim=1)
        output = self.fusion_head(fused)

        return output

    def count_parameters(self) -> dict:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'kmer_branch': sum(p.numel() for p in self.kmer_branch.parameters()),
            'assembly_branch': sum(p.numel() for p in self.assembly_branch.parameters()),
            'fusion_head': sum(p.numel() for p in self.fusion_head.parameters()),
        }


def build_model(
    n_kmer_features: int = 9249,
    n_assembly_features: int = 26,
    use_gradient_checkpointing: bool = True,
    device: str = 'cuda',
) -> MAGICCModel:
    """
    Build and return the MAGICC model.

    Parameters
    ----------
    n_kmer_features : int
        Number of k-mer input features.
    n_assembly_features : int
        Number of assembly statistics input features.
    use_gradient_checkpointing : bool
        Whether to use gradient checkpointing.
    device : str
        Device to place model on.

    Returns
    -------
    MAGICCModel
        The initialized model on the specified device.
    """
    model = MAGICCModel(
        n_kmer_features=n_kmer_features,
        n_assembly_features=n_assembly_features,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    model = model.to(device)
    return model


# ============================================================================
# V3: Attention-Based Architecture
# ============================================================================


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for feature reweighting.

    Global average pool -> Dense(reduction) -> ReLU -> Dense(channels) -> Sigmoid
    Element-wise multiply to reweight features.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    reduction : int
        Reduction ratio for the bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 32)
        self.squeeze = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels)
        scale = self.squeeze(x)
        return x * scale


class KmerBranchV3(nn.Module):
    """K-mer branch with SE attention for feature reweighting.

    Architecture:
        Dense(4096) -> BN -> SiLU -> Drop -> SE(4096) ->
        Dense(1024) -> BN -> SiLU -> Drop -> SE(1024) ->
        Dense(256)  -> BN -> SiLU
    """

    def __init__(self, n_kmer_features: int = 9249, dropout1: float = 0.4,
                 dropout2: float = 0.2, se_reduction: int = 16):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_kmer_features, 4096),
            nn.BatchNorm1d(4096),
            nn.SiLU(),
            nn.Dropout(dropout1),
        )
        self.se1 = SEBlock(4096, reduction=se_reduction)
        self.layer2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(dropout2),
        )
        self.se2 = SEBlock(1024, reduction=se_reduction)
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion: assembly embedding queries k-mer embedding.

    Assembly features act as query, k-mer features provide key/value.
    This allows structural genome properties (size, GC, fragmentation) to
    dynamically select which k-mer patterns are most informative.

    The k-mer embedding (256d) is reshaped into n_groups tokens, each of
    dim (256 // n_groups). Multi-head attention is applied, and the output
    is pooled back to a fixed-size vector.

    Parameters
    ----------
    kmer_dim : int
        Dimension of k-mer embedding (default 256).
    assembly_dim : int
        Dimension of assembly embedding (default 64).
    n_heads : int
        Number of attention heads.
    n_groups : int
        Number of groups to split the k-mer embedding into tokens.
    dropout : float
        Dropout for attention weights.
    """

    def __init__(self, kmer_dim: int = 256, assembly_dim: int = 64,
                 n_heads: int = 4, n_groups: int = 16, dropout: float = 0.1):
        super().__init__()
        self.n_groups = n_groups
        self.token_dim = kmer_dim // n_groups  # 256/16 = 16
        self.n_heads = n_heads

        # Project assembly embedding to query dimension matching token_dim
        self.query_proj = nn.Linear(assembly_dim, self.token_dim)
        # Key and value projections from k-mer tokens
        self.key_proj = nn.Linear(self.token_dim, self.token_dim)
        self.value_proj = nn.Linear(self.token_dim, self.token_dim)

        self.scale = (self.token_dim // n_heads) ** -0.5

        # Multi-head attention via splitting token_dim into heads
        assert self.token_dim % n_heads == 0, \
            f"token_dim ({self.token_dim}) must be divisible by n_heads ({n_heads})"
        self.head_dim = self.token_dim // n_heads

        # Output projection after attention
        self.out_proj = nn.Linear(self.token_dim, self.token_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # LayerNorm for stability
        self.norm_q = nn.LayerNorm(self.token_dim)
        self.norm_kv = nn.LayerNorm(self.token_dim)

        # Final projection to get attended kmer representation
        self.pool_proj = nn.Sequential(
            nn.Linear(n_groups * self.token_dim, kmer_dim),
            nn.SiLU(),
        )

    def forward(self, kmer_embed: torch.Tensor, assembly_embed: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        kmer_embed : (batch, kmer_dim)  e.g. (B, 256)
        assembly_embed : (batch, assembly_dim)  e.g. (B, 64)

        Returns
        -------
        attended_kmer : (batch, kmer_dim)  e.g. (B, 256)
        """
        B = kmer_embed.shape[0]

        # Reshape k-mer embedding into tokens: (B, n_groups, token_dim)
        kv_tokens = kmer_embed.view(B, self.n_groups, self.token_dim)
        kv_tokens = self.norm_kv(kv_tokens)

        # Assembly -> single query token: (B, 1, token_dim)
        q = self.query_proj(assembly_embed).unsqueeze(1)
        q = self.norm_q(q)

        # Project to key/value
        k = self.key_proj(kv_tokens)   # (B, n_groups, token_dim)
        v = self.value_proj(kv_tokens)  # (B, n_groups, token_dim)

        # Multi-head attention: reshape for heads
        # q: (B, n_heads, 1, head_dim)
        # k: (B, n_heads, n_groups, head_dim)
        # v: (B, n_heads, n_groups, head_dim)
        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, self.n_groups, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, self.n_groups, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, n_heads, 1, n_groups)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted values: (B, n_heads, 1, head_dim)
        out = torch.matmul(attn, v)

        # Reshape: (B, 1, token_dim)
        out = out.transpose(1, 2).contiguous().view(B, 1, self.token_dim)
        out = self.out_proj(out)  # (B, 1, token_dim)

        # Broadcast attended query back to all tokens and combine with original
        # Gated residual: use attention output to reweight original tokens
        gate = torch.sigmoid(out)  # (B, 1, token_dim)
        gated_tokens = kv_tokens * gate  # (B, n_groups, token_dim) broadcast

        # Pool back to kmer_dim
        attended = gated_tokens.view(B, -1)  # (B, n_groups * token_dim)
        attended = self.pool_proj(attended)   # (B, kmer_dim)

        return attended


class FusionHeadV3(nn.Module):
    """Fusion head with cross-attention between k-mer and assembly branches.

    Instead of simple concatenation, uses cross-attention to let assembly
    features guide which k-mer patterns are important, then concatenates
    the attended k-mer representation with the assembly embedding.

    Output activation per task:
    - Completeness: Sigmoid * 50 + 50 -> [50, 100]
    - Contamination: Sigmoid * 100 -> [0, 100]
    """

    def __init__(self, kmer_dim: int = 256, assembly_dim: int = 64,
                 n_heads: int = 4, n_groups: int = 16,
                 attn_dropout: float = 0.1, fusion_dropout: float = 0.1):
        super().__init__()

        self.cross_attention = CrossAttentionFusion(
            kmer_dim=kmer_dim,
            assembly_dim=assembly_dim,
            n_heads=n_heads,
            n_groups=n_groups,
            dropout=attn_dropout,
        )

        # Residual connection for k-mer embedding
        self.kmer_gate = nn.Sequential(
            nn.Linear(kmer_dim * 2, kmer_dim),
            nn.Sigmoid(),
        )

        fusion_input_dim = kmer_dim + assembly_dim  # 256 + 64 = 320

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(fusion_dropout),
        )
        self.pre_output = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.output_linear = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, kmer_embed: torch.Tensor,
                assembly_embed: torch.Tensor) -> torch.Tensor:
        # Cross-attention: assembly queries k-mer
        attended_kmer = self.cross_attention(kmer_embed, assembly_embed)

        # Gated residual: blend original and attended k-mer embeddings
        gate = self.kmer_gate(torch.cat([kmer_embed, attended_kmer], dim=1))
        kmer_fused = gate * attended_kmer + (1 - gate) * kmer_embed

        # Concatenate with assembly and predict
        fused = torch.cat([kmer_fused, assembly_embed], dim=1)
        x = self.fusion(fused)
        x = self.pre_output(x)
        raw = self.output_linear(x)
        activated = self.sigmoid(raw)
        # Completeness: Sigmoid * 50 + 50 -> [50, 100]
        comp = activated[:, 0:1] * 50.0 + 50.0
        # Contamination: Sigmoid * 100 -> [0, 100]
        cont = activated[:, 1:2] * 100.0
        return torch.cat([comp, cont], dim=1)


class MAGICCModelV3(nn.Module):
    """
    V3: Attention-Based Multi-Branch Fusion Neural Network.

    Improvements over V2:
    - SE (Squeeze-and-Excitation) attention in k-mer branch for adaptive
      feature reweighting after each dense layer
    - Cross-attention at fusion stage: assembly embedding queries k-mer
      embedding to extract context-dependent k-mer patterns
    - Gated residual connections for stable training

    Architecture:
      K-mer Branch with SE Attention (~9,249 -> 256):
        Dense(4096) -> BN -> SiLU -> Drop(0.4) -> SE(4096)
        Dense(1024) -> BN -> SiLU -> Drop(0.2) -> SE(1024)
        Dense(256)  -> BN -> SiLU

      Assembly Branch (26 -> 64):
        Dense(128) -> BN -> SiLU -> Drop(0.2)
        Dense(64)  -> BN -> SiLU

      Cross-Attention Fusion (256+64 -> 2):
        CrossAttention(query=assembly_64, key/value=kmer_256)
        GatedResidual(kmer_original, kmer_attended) -> kmer_fused_256
        Concat(kmer_fused_256 + assembly_64 = 320)
        Dense(128) -> BN -> SiLU -> Drop(0.1)
        Dense(64) -> SiLU
        Dense(2) -> Completeness: Sigmoid*50+50, Contamination: Sigmoid*100

    Parameters
    ----------
    n_kmer_features : int
        Number of k-mer input features (default: 9249).
    n_assembly_features : int
        Number of assembly statistics input features (default: 26).
    kmer_dropout1 : float
        Dropout rate for k-mer branch layer 1.
    kmer_dropout2 : float
        Dropout rate for k-mer branch layer 2.
    se_reduction : int
        SE block reduction ratio.
    assembly_dropout1 : float
        Dropout rate for assembly branch layer 1.
    n_attn_heads : int
        Number of cross-attention heads.
    n_attn_groups : int
        Number of groups for k-mer token splitting in cross-attention.
    attn_dropout : float
        Dropout for attention weights.
    fusion_dropout : float
        Dropout for fusion layers.
    use_gradient_checkpointing : bool
        Whether to use gradient checkpointing to save GPU memory.
    """

    def __init__(
        self,
        n_kmer_features: int = 9249,
        n_assembly_features: int = 26,
        kmer_dropout1: float = 0.4,
        kmer_dropout2: float = 0.2,
        se_reduction: int = 16,
        assembly_dropout1: float = 0.2,
        n_attn_heads: int = 4,
        n_attn_groups: int = 16,
        attn_dropout: float = 0.1,
        fusion_dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.n_kmer_features = n_kmer_features
        self.n_assembly_features = n_assembly_features
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.kmer_branch = KmerBranchV3(
            n_kmer_features=n_kmer_features,
            dropout1=kmer_dropout1,
            dropout2=kmer_dropout2,
            se_reduction=se_reduction,
        )
        self.assembly_branch = AssemblyBranch(
            n_assembly_features=n_assembly_features,
            dropout1=assembly_dropout1,
        )
        self.fusion_head = FusionHeadV3(
            kmer_dim=256,
            assembly_dim=64,
            n_heads=n_attn_heads,
            n_groups=n_attn_groups,
            attn_dropout=attn_dropout,
            fusion_dropout=fusion_dropout,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal for SiLU/ReLU layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _checkpoint_kmer(self, x: torch.Tensor) -> torch.Tensor:
        """Run k-mer branch with optional gradient checkpointing."""
        if self.use_gradient_checkpointing and self.training:
            x = x.requires_grad_(True)
            return checkpoint(self.kmer_branch, x, use_reentrant=False)
        return self.kmer_branch(x)

    def _checkpoint_assembly(self, x: torch.Tensor) -> torch.Tensor:
        """Run assembly branch with optional gradient checkpointing."""
        if self.use_gradient_checkpointing and self.training:
            x = x.requires_grad_(True)
            return checkpoint(self.assembly_branch, x, use_reentrant=False)
        return self.assembly_branch(x)

    def forward(
        self,
        kmer_features: torch.Tensor,
        assembly_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        kmer_features : torch.Tensor
            K-mer count features, shape (batch_size, n_kmer_features).
        assembly_features : torch.Tensor
            Assembly statistics features, shape (batch_size, n_assembly_features).

        Returns
        -------
        torch.Tensor
            Predictions [completeness, contamination], shape (batch_size, 2).
        """
        kmer_embed = self._checkpoint_kmer(kmer_features)
        assembly_embed = self._checkpoint_assembly(assembly_features)

        # Cross-attention fusion (not simple concatenation)
        output = self.fusion_head(kmer_embed, assembly_embed)

        return output

    def count_parameters(self) -> dict:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'kmer_branch': sum(p.numel() for p in self.kmer_branch.parameters()),
            'assembly_branch': sum(p.numel() for p in self.assembly_branch.parameters()),
            'fusion_head': sum(p.numel() for p in self.fusion_head.parameters()),
        }


def build_model_v3(
    n_kmer_features: int = 9249,
    n_assembly_features: int = 26,
    use_gradient_checkpointing: bool = True,
    device: str = 'cuda',
    **kwargs,
) -> MAGICCModelV3:
    """
    Build and return the MAGICC V3 attention model.

    Parameters
    ----------
    n_kmer_features : int
        Number of k-mer input features.
    n_assembly_features : int
        Number of assembly statistics input features.
    use_gradient_checkpointing : bool
        Whether to use gradient checkpointing.
    device : str
        Device to place model on.
    **kwargs
        Additional keyword arguments passed to MAGICCModelV3.

    Returns
    -------
    MAGICCModelV3
        The initialized model on the specified device.
    """
    model = MAGICCModelV3(
        n_kmer_features=n_kmer_features,
        n_assembly_features=n_assembly_features,
        use_gradient_checkpointing=use_gradient_checkpointing,
        **kwargs,
    )
    model = model.to(device)
    return model
