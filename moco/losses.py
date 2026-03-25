"""
moco/losses.py
===============
All training losses for the Multi-Distortion Encoder system.

WHAT THIS FILE DOES:
--------------------
Three loss functions work together during Phase 1 training:

    1. InfoNCE Loss (contrastive)
       ─ Same as original Re-IQA's MoCo-v2 loss.
       ─ Pulls two augmented views of the same image together.
       ─ Pushes views from different images apart.
       ─ Applied once per specialist head.

    2. Manifold Triplet Loss (ARNIQA-style)
       ─ Teaches each head about severity ordering.
       ─ anchor (clean) should be close to positive (mild distortion)
       ─ anchor should be far from negative (heavy distortion)
       ─ This is what creates the "clean → degraded" manifold geometry.

    3. Gating Entropy Loss (regularisation)
       ─ Prevents the gating network from always picking one head.
       ─ Encourages balanced utilisation of all specialists.
       ─ Without this, the gate tends to collapse to one head quickly.

TOTAL LOSS:
-----------
    L_total = L_infonce + λ_triplet * L_triplet + λ_gate * L_gate

    Recommended starting weights:
        λ_triplet = 0.5
        λ_gate    = 0.1  (small — it's a regulariser, not a main objective)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  SECTION 1: InfoNCE Loss
#  This is the standard MoCo-v2 contrastive loss.
#  Re-IQA uses this exactly as-is. We apply it per head.
# ─────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss, identical to MoCo-v2.

    Given a batch of (query, key) pairs where each query matches
    exactly one key (its augmented counterpart), this loss maximises
    agreement between matching pairs while minimising agreement with
    all other pairs in the queue.

    INTUITION:
        Imagine 256 images in a batch. For image #42, we have a query
        embedding q42 and a key embedding k42. InfoNCE asks the network
        to identify which of the 65,536 queue entries is k42. Getting
        it right requires the embeddings to encode meaningful similarity.

    Args:
        temperature: softmax temperature (lower = sharper, default 0.2)
                     Re-IQA uses 0.2 following MoCo-v2 convention.

    Inputs:
        q   : query embeddings (B, D) — from encoder_q
        k   : key embeddings (B, D)   — from momentum encoder_k
        queue: negative queue (D, K)  — maintained by MoCo builder

    Returns:
        scalar loss value
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        queue: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q     : (B, D) query embeddings, L2 normalised
            k     : (B, D) key embeddings,   L2 normalised
            queue : (D, K) negative sample queue, L2 normalised
        Returns:
            scalar InfoNCE loss
        """
        B = q.shape[0]

        # Positive logits: dot product between each query and its key
        # Shape: (B, 1)
        l_pos = torch.einsum('bd,bd->b', q, k).unsqueeze(-1)

        # Negative logits: dot product between each query and all queue entries
        # Shape: (B, K)
        l_neg = torch.einsum('bd,dk->bk', q, queue)

        # Concatenate: column 0 is the positive, rest are negatives
        logits = torch.cat([l_pos, l_neg], dim=1)  # (B, K+1)
        logits /= self.temperature

        # Labels: the positive is always at index 0
        labels = torch.zeros(B, dtype=torch.long, device=q.device)

        return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────
#  SECTION 2: Manifold Triplet Loss
#  This is the ARNIQA contribution to our training.
#  It teaches each head that distance = severity difference.
# ─────────────────────────────────────────────────────────────

class ManifoldTripletLoss(nn.Module):
    """
    Triplet loss for learning the distortion severity manifold.

    This is what turns a generic distortion-aware head into an
    ARNIQA-style manifold encoder. After sufficient training:

        d(anchor, positive) < d(anchor, negative)

    meaning mildly distorted images are closer in embedding space
    to clean images than heavily distorted ones are.

    WHAT THIS CREATES:
        For the Gaussian head, imagine the embedding space. Clean
        images cluster in one region. As you add more Gaussian noise,
        images move smoothly away from that cluster. The path they
        trace is the "Gaussian noise manifold." Its geometry is:
            clean → mild noise → moderate noise → heavy noise
        Each step is an equal-ish distance in embedding space.

    Args:
        margin: minimum required gap between d(a,p) and d(a,n).
                Default 0.5 — large enough to force clear separation.

    Inputs:
        anchor   : (B, D) clean/mild embeddings
        positive : (B, D) mild distortion embeddings
        negative : (B, D) heavy distortion embeddings

    Returns:
        scalar loss = mean(max(0, d(a,p) - d(a,n) + margin))
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor   : (B, D) L2-normalised anchor embeddings
            positive : (B, D) L2-normalised mild-distortion embeddings
            negative : (B, D) L2-normalised heavy-distortion embeddings
        Returns:
            scalar triplet loss
        """
        # Euclidean distances (works on unit sphere = equivalent to cosine)
        d_pos = F.pairwise_distance(anchor, positive, p=2)  # (B,)
        d_neg = F.pairwise_distance(anchor, negative, p=2)  # (B,)

        # Hinge loss: only penalise when d_pos >= d_neg - margin
        losses = F.relu(d_pos - d_neg + self.margin)        # (B,)

        return losses.mean()


# ─────────────────────────────────────────────────────────────
#  SECTION 3: Gating Entropy Regularisation Loss
#  Without this, the gating network quickly learns to always
#  activate the "easiest" head and ignore the rest.
# ─────────────────────────────────────────────────────────────

class GatingEntropyLoss(nn.Module):
    """
    Regularisation loss that encourages the gating network to
    distribute weight across all specialist heads.

    WHY THIS IS NEEDED:
        During early training, one head (usually the first one to
        converge) will start producing better embeddings. The gating
        network notices this and starts weighting it at ~1.0 and the
        rest at ~0.0. This "winner-take-all" collapse prevents the
        other heads from getting useful gradients, so they never
        specialise, which in turn never gives the gate a reason to
        use them. It's a self-reinforcing collapse.

    HOW IT WORKS:
        Shannon entropy of the gate weights = -Σ w_i * log(w_i)
        Maximum entropy = log(num_heads) ≈ 1.386 for 4 heads
        (achieved when all weights are equal = 0.25 each)
        We MAXIMISE entropy → equivalent to MINIMISING -entropy.

    Args:
        num_heads: number of specialist heads (default 4)

    Input:
        weights: (B, num_heads) softmax gate weights
    Returns:
        scalar loss — minimise this to encourage balanced gate usage
    """

    def __init__(self, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weights: (B, num_heads) softmax weights from GatingNetwork
        Returns:
            scalar: negative mean entropy (minimise to maximise entropy)
        """
        # Clamp to avoid log(0) numerical issues
        w = weights.clamp(min=1e-8)

        # Entropy per sample: -Σ w_i * log(w_i)
        entropy = -(w * w.log()).sum(dim=-1)        # (B,)

        # We want to MAXIMISE entropy → return NEGATIVE mean entropy
        # (standard optimisers minimise, so flipping sign achieves maximisation)
        return -entropy.mean()


# ─────────────────────────────────────────────────────────────
#  SECTION 4: Combined Loss
#  Wraps all three losses with configurable weights.
#  Use this in main_contrast.py — single call, clean interface.
# ─────────────────────────────────────────────────────────────

class MultiDistortionLoss(nn.Module):
    """
    Combined loss for training the Multi-Distortion Encoder.

    Combines:
        L_total = L_infonce
                + lambda_triplet * L_triplet
                + lambda_gate    * L_gate

    Args:
        temperature    : InfoNCE temperature (default 0.2)
        triplet_margin : triplet loss margin (default 0.5)
        lambda_triplet : weight for triplet loss (default 0.5)
        lambda_gate    : weight for gating entropy loss (default 0.1)
        num_heads      : number of specialist heads (default 4)

    Usage:
        criterion = MultiDistortionLoss()

        # In training loop:
        loss, breakdown = criterion(
            q=query_embed,
            k=key_embed,
            queue=moco_queue,
            anchor=anchor_embed,
            positive=positive_embed,
            negative=negative_embed,
            gate_weights=gate_w,
        )
        loss.backward()
    """

    def __init__(
        self,
        temperature: float = 0.2,
        triplet_margin: float = 0.5,
        lambda_triplet: float = 0.5,
        lambda_gate: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.infonce  = InfoNCELoss(temperature)
        self.triplet  = ManifoldTripletLoss(triplet_margin)
        self.gate_reg = GatingEntropyLoss(num_heads)

        self.lambda_triplet = lambda_triplet
        self.lambda_gate    = lambda_gate

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        queue: torch.Tensor,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        gate_weights: torch.Tensor,
    ) -> tuple:
        """
        Args:
            q, k, queue     : InfoNCE inputs (see InfoNCELoss)
            anchor, positive, negative : triplet inputs (see ManifoldTripletLoss)
            gate_weights    : (B, num_heads) from GatingNetwork
        Returns:
            total_loss : scalar, backpropagate this
            breakdown  : dict with individual loss values (for logging)
        """
        l_infonce = self.infonce(q, k, queue)
        l_triplet = self.triplet(anchor, positive, negative)
        l_gate    = self.gate_reg(gate_weights)

        total = (
            l_infonce
            + self.lambda_triplet * l_triplet
            + self.lambda_gate    * l_gate
        )

        breakdown = {
            'loss_infonce' : l_infonce.item(),
            'loss_triplet' : l_triplet.item(),
            'loss_gate'    : l_gate.item(),
            'loss_total'   : total.item(),
        }

        return total, breakdown
