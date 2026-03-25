"""
networks/multi_distortion_encoder.py
=====================================
Multi-Distortion Encoder for Re-IQA + ARNIQA fusion.

WHAT THIS FILE DOES:
--------------------
This replaces Re-IQA's single noise-aware ResNet-50 encoder with a
more powerful system that has:

    1. A shared ResNet-50 backbone  (runs ONCE per image)
    2. Four specialist heads        (one per distortion type)
    3. ARNIQA-style manifold loss   (per head, in training)
    4. A MoE gating network         (blends the four heads)

ARCHITECTURE FLOW:
------------------
    image
      |
      v
  [Shared Backbone]  <-- ResNet-50 layers 1-4, outputs 2048-dim vector
      |
   /  |  |  \
  v   v  v   v
 [G] [B] [J] [W]   <-- 4 specialist heads (each: Linear->ReLU->Linear->L2norm)
  |   |  |   |         G=Gaussian, B=Blur, J=JPEG, W=Weather/Haze
  v   v  v   v
 f1  f2  f3  f4    <-- 4 x 128-dim unit-norm embeddings

      image
        |
        v
  [Gate Network]   <-- lightweight ResNet-18, outputs 4 softmax weights
        |
        v
    [w1,w2,w3,w4]

FUSION:
-------
    fused = w1*f1 + w2*f2 + w3*f3 + w4*f4   (weighted sum, then L2 norm)

OUTPUT:
-------
    128-dim unit-norm fused noise embedding
    (same shape as original Re-IQA noise encoder output)

HOW IT CONNECTS TO RE-IQA:
---------------------------
    - Original Re-IQA:  concat(noise_embed, content_embed) -> regressor
    - Modified Re-IQA:  concat(fused_embed, content_embed) -> regressor
    - Content encoder and regressor are UNCHANGED.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ─────────────────────────────────────────────────────────────
#  SECTION 1: Specialist Head
#  One of these exists per distortion type.
#  Takes the 2048-dim backbone output and projects it into a
#  128-dim space organised around that specific distortion.
# ─────────────────────────────────────────────────────────────

class DistortionHead(nn.Module):
    """
    A two-layer MLP projection head for one distortion type.

    Each head learns its own 128-dim embedding space organised as
    an ARNIQA-style distortion manifold — meaning distance in this
    space corresponds to distortion severity for this specific type.

    Args:
        in_dim  : input dimension from backbone (default 2048 for ResNet-50)
        hidden_dim : intermediate dimension (default 512)
        out_dim : output embedding dimension (default 128)

    Input:  (B, in_dim)  — shared backbone features
    Output: (B, out_dim) — L2-normalised distortion embedding
    """

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: backbone features of shape (B, in_dim)
        Returns:
            L2-normalised embedding of shape (B, out_dim)
        """
        return F.normalize(self.mlp(x), dim=-1)


# ─────────────────────────────────────────────────────────────
#  SECTION 2: Gating Network
#  Looks at the RAW IMAGE (not the backbone features) and
#  decides how much weight to give each specialist head.
#  Uses a lightweight ResNet-18 so it doesn't add much compute.
# ─────────────────────────────────────────────────────────────

class GatingNetwork(nn.Module):
    """
    Lightweight image classifier that outputs soft mixture weights.

    Takes the raw input image and learns to detect which distortion
    types are present and in what proportion. Outputs 4 weights that
    sum to 1.0 via softmax — these become the blend coefficients for
    the four specialist head embeddings.

    WHY SEPARATE FROM BACKBONE:
        The backbone is trained to be distortion-aware but not to
        classify distortion type explicitly. The gating network has
        a dedicated, independent objective: identify distortion type.
        Keeping them separate prevents interference.

    Args:
        num_heads: number of specialist heads (default 4)

    Input:  (B, 3, H, W)  — raw image
    Output: (B, num_heads) — softmax weights summing to 1
    """

    def __init__(self, num_heads: int = 4):
        super().__init__()
        # ResNet-18 is fast and sufficient for distortion-type classification
        resnet = models.resnet18(weights=None)
        # Remove the final FC layer — keep everything up to global avg pool
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(512, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw image tensor of shape (B, 3, H, W)
        Returns:
            soft weights of shape (B, num_heads), sum to 1 per sample
        """
        feat = self.features(x)                    # (B, 512, 1, 1)
        feat = feat.squeeze(-1).squeeze(-1)        # (B, 512)
        logits = self.classifier(feat)             # (B, num_heads)
        return F.softmax(logits, dim=-1)           # (B, num_heads)


# ─────────────────────────────────────────────────────────────
#  SECTION 3: Multi-Distortion Encoder (main class)
#  This is the DROP-IN REPLACEMENT for Re-IQA's single
#  noise-aware encoder. Same input, same output shape.
# ─────────────────────────────────────────────────────────────

class MultiDistortionEncoder(nn.Module):
    """
    Multi-Distortion Encoder: drop-in replacement for Re-IQA's
    single quality-aware (noise-aware) encoder.

    Combines:
        - Shared ResNet-50 backbone (same as original Re-IQA)
        - 4 specialist heads with ARNIQA-style manifolds
        - MoE gating network for intelligent blending

    DISTORTION TYPES HANDLED:
        'gaussian' — sensor noise, AWGN, random pixel corruption
        'blur'     — motion blur, defocus, Gaussian blur
        'jpeg'     — blocking artifacts, compression noise
        'weather'  — haze, fog, rain-induced degradation

    OUTPUT:
        A single 128-dim L2-normalised embedding that is a
        distortion-weighted blend of all specialist embeddings.
        This is the SAME SHAPE as the original Re-IQA quality
        encoder output, so nothing downstream needs to change.

    Args:
        backbone_dim : output dim of ResNet-50 backbone (2048)
        hidden_dim   : hidden dim inside each head (512)
        embed_dim    : final embedding dimension per head (128)

    Example:
        encoder = MultiDistortionEncoder()
        img = torch.randn(4, 3, 224, 224)
        embedding = encoder(img)          # (4, 128)
        embedding, weights = encoder.forward_with_weights(img)
        # weights shape: (4, 4) — tells you which distortion dominated
    """

    # Ordered list — index order matters for gating weights
    DISTORTION_TYPES = ['gaussian', 'blur', 'jpeg', 'weather']

    def __init__(
        self,
        backbone_dim: int = 2048,
        hidden_dim: int = 512,
        embed_dim: int = 128,
    ):
        super().__init__()

        # ── Shared backbone ──────────────────────────────────────
        # Same ResNet-50 as original Re-IQA, strip the FC head.
        # Pretrained on ImageNet to give a strong starting point.
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Output: (B, 2048, 1, 1) -> after squeeze: (B, 2048)

        # ── Specialist heads ─────────────────────────────────────
        # One head per distortion type. Stored in a ModuleDict so
        # PyTorch tracks all parameters correctly.
        self.heads = nn.ModuleDict({
            name: DistortionHead(backbone_dim, hidden_dim, embed_dim)
            for name in self.DISTORTION_TYPES
        })

        # ── Gating network ───────────────────────────────────────
        self.gate = GatingNetwork(num_heads=len(self.DISTORTION_TYPES))

        self.embed_dim = embed_dim

    def _get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run shared backbone once.
        Input:  (B, 3, H, W)
        Output: (B, 2048)
        """
        feat = self.backbone(x)                    # (B, 2048, 1, 1)
        return feat.squeeze(-1).squeeze(-1)        # (B, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass. Returns fused embedding only.

        Use this during MoCo training and feature extraction,
        where you just need the embedding vector.

        Args:
            x: image tensor (B, 3, H, W)
        Returns:
            fused L2-normalised embedding (B, embed_dim)
        """
        fused, _ = self.forward_with_weights(x)
        return fused

    def forward_with_weights(self, x: torch.Tensor):
        """
        Full forward pass returning both embedding and gate weights.

        Use this when:
          - Computing the gating entropy regularisation loss
          - Inspecting which distortion type dominates an image
          - Debugging / visualisation

        Args:
            x: image tensor (B, 3, H, W)
        Returns:
            fused   : (B, embed_dim) — L2-normalised fused embedding
            weights : (B, num_heads) — softmax gating weights
                      columns = [w_gaussian, w_blur, w_jpeg, w_weather]
        """
        # Step 1: shared backbone features (runs ONCE)
        feat = self._get_backbone_features(x)      # (B, 2048)

        # Step 2: each head produces its own embedding
        head_embeds = torch.stack(
            [self.heads[name](feat) for name in self.DISTORTION_TYPES],
            dim=1
        )  # (B, num_heads, embed_dim)

        # Step 3: gating weights from raw image
        weights = self.gate(x)                     # (B, num_heads)

        # Step 4: weighted sum fusion
        # weights.unsqueeze(-1) : (B, num_heads, 1)
        # broadcast multiply    : (B, num_heads, embed_dim)
        # sum over heads        : (B, embed_dim)
        fused = (weights.unsqueeze(-1) * head_embeds).sum(dim=1)

        # Step 5: L2 normalise (keeps embedding on unit hypersphere,
        # which is required for InfoNCE / contrastive losses)
        fused = F.normalize(fused, dim=-1)

        return fused, weights

    def get_individual_embeddings(self, x: torch.Tensor) -> dict:
        """
        Returns each head's embedding separately (for analysis/debugging).

        Args:
            x: image tensor (B, 3, H, W)
        Returns:
            dict mapping distortion name -> (B, embed_dim) tensor
            e.g. {'gaussian': tensor, 'blur': tensor, ...}
        """
        feat = self._get_backbone_features(x)
        return {
            name: self.heads[name](feat)
            for name in self.DISTORTION_TYPES
        }
