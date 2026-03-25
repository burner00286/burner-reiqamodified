"""
moco/builder_mde.py
====================
Modified MoCo builder using MultiDistortionEncoder.

WHAT THIS FILE DOES:
--------------------
Re-IQA's original moco/builder.py wraps a single ResNet-50 in the
MoCo momentum-contrast framework (query encoder + momentum encoder
+ queue of negatives).

This file does the SAME THING but swaps the ResNet-50 for our
MultiDistortionEncoder. Everything else — momentum update, queue,
training loop interface — is identical to the original.

WHAT IS MoCo:
-------------
MoCo (Momentum Contrast) maintains:
    encoder_q  : the "query" encoder, updated by gradient
    encoder_k  : the "key" encoder, updated by MOMENTUM
                 (slow-moving average of encoder_q weights)
    queue      : a large bank of recent key embeddings used
                 as negatives in the InfoNCE loss

The momentum encoder produces more stable keys than if we just
ran the same image twice through encoder_q. This stability is
crucial for the quality of the contrastive signal.

CHANGES FROM ORIGINAL builder.py:
-----------------------------------
    Line ~30: import MultiDistortionEncoder
    Line ~50: encoder_q = MultiDistortionEncoder(...)
    Line ~55: encoder_k = MultiDistortionEncoder(...)
    EVERYTHING ELSE IS IDENTICAL.
"""

import torch
import torch.nn as nn
from networks.multi_distortion_encoder import MultiDistortionEncoder


class MoCo_MDE(nn.Module):
    """
    MoCo wrapper around MultiDistortionEncoder.

    Drop-in replacement for Re-IQA's original MoCo builder.
    The training loop in main_contrast.py sees the same interface.

    Args:
        embed_dim  : embedding dimension (default 128, same as Re-IQA)
        K          : queue size — number of negative keys (default 65536)
        m          : momentum for key encoder update (default 0.999)
        T          : InfoNCE temperature (default 0.2)
        backbone_dim: backbone output dim (2048 for ResNet-50)

    Key attributes:
        encoder_q  : query encoder (MultiDistortionEncoder), trained by grad
        encoder_k  : key encoder   (MultiDistortionEncoder), trained by momentum
        queue      : (embed_dim, K) tensor of recent negative embeddings
    """

    def __init__(
        self,
        embed_dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.2,
        backbone_dim: int = 2048,
    ):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # ── Encoders ─────────────────────────────────────────────
        # Both use the same MultiDistortionEncoder architecture.
        # encoder_k starts as an exact copy of encoder_q.
        self.encoder_q = MultiDistortionEncoder(
            backbone_dim=backbone_dim,
            embed_dim=embed_dim,
        )
        self.encoder_k = MultiDistortionEncoder(
            backbone_dim=backbone_dim,
            embed_dim=embed_dim,
        )

        # ── Initialise encoder_k as copy of encoder_q ────────────
        # Copy weights, then freeze — encoder_k is NEVER updated
        # by gradients, only by momentum (see _momentum_update).
        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # ── Queue of negative embeddings ─────────────────────────
        # Shape: (embed_dim, K) — each column is one embedding.
        # Registered as a buffer so it moves to GPU with .to(device)
        # but is NOT a model parameter (no gradients).
        self.register_buffer('queue', torch.randn(embed_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # Pointer tracks where to write the next batch of keys
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    # ─────────────────────────────────────────────────────────
    #  Momentum update
    #  Called once per training step BEFORE the forward pass.
    #  Slowly moves encoder_k weights toward encoder_q.
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Update key encoder by exponential moving average:
            θ_k = m * θ_k + (1 - m) * θ_q

        With m=0.999, each update moves θ_k only 0.1% toward θ_q.
        This makes the key encoder change very slowly, which
        provides stable and consistent negatives for the queue.
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data = (
                param_k.data * self.m
                + param_q.data * (1.0 - self.m)
            )

    # ─────────────────────────────────────────────────────────
    #  Queue operations
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """
        Add new keys to the queue, overwriting the oldest entries.

        The queue is a circular buffer of size K. Each call writes
        a new batch of keys at the current pointer position and
        advances the pointer.

        Args:
            keys: (B, embed_dim) new key embeddings to add
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Overwrite queue at pointer position
        # keys.T is (embed_dim, B)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    # ─────────────────────────────────────────────────────────
    #  Forward pass
    # ─────────────────────────────────────────────────────────

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor):
        """
        MoCo forward pass for one training step.

        Args:
            im_q: (B, 3, H, W) query images   — augmented view 1
            im_k: (B, 3, H, W) key images     — augmented view 2

        Returns:
            q            : (B, embed_dim) query embeddings
            k            : (B, embed_dim) key embeddings (detached)
            queue_copy   : (embed_dim, K) current queue (for InfoNCE)
            gate_weights : (B, num_heads) gating weights from encoder_q
                           (used for gating entropy regularisation loss)
        """
        # ── Query embeddings from encoder_q (with gradients) ─────
        q, gate_weights = self.encoder_q.forward_with_weights(im_q)
        # q: (B, embed_dim), already L2-normalised

        # ── Key embeddings from encoder_k (no gradients) ─────────
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            # k: (B, embed_dim), already L2-normalised

        # ── Snapshot queue BEFORE updating it ────────────────────
        # The loss uses the queue state from BEFORE this batch's keys
        # are added (prevents the same sample being both query and key)
        queue_copy = self.queue.clone().detach()

        # ── Update queue ──────────────────────────────────────────
        self._dequeue_and_enqueue(k)

        return q, k, queue_copy, gate_weights
