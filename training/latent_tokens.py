"""
Latent Tokens — Minimal Viable Latent (L) Component

Inserts K learnable continuous tokens into the input sequence.
These tokens:
  - Have no vocabulary mapping (pure computation space)
  - Participate in self-attention with real tokens
  - Are included in STR trajectory computation (critical for S+L interaction)
  - Do NOT contribute to language modeling loss

Design:
  input:  [BOS] Q1 Q2 ... Qn [LAT1] [LAT2] ... [LATK] A1 A2 ... Am [EOS]
                               ^^^^^^^^^^^^^^^^^^^^^^^^^
                               learnable embeddings

  The latent tokens provide K extra "thinking steps" between question and answer.
  When combined with STR, the trajectory through latent positions is constrained
  to be geometrically coherent — this is the hypothesized S+L interaction.

Compatible with:
  - GPT-2 (124M) — full fine-tuning
  - Llama-3.2-3B — LoRA fine-tuning (latent params always trainable)

Usage:
    from latent_tokens import LatentTokenInjector

    injector = LatentTokenInjector(K=8, hidden_dim=768)
    input_embeds, attention_mask, latent_mask = injector(
        input_ids, attention_mask, model.get_input_embeddings(), split_pos=question_len
    )
    outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentTokenInjector(nn.Module):
    """
    Injects K learnable continuous tokens into the input sequence.

    The tokens are inserted at a specified split position (typically after
    the question tokens, before the answer tokens). This creates an
    explicit "thinking" region in the sequence.

    Args:
        K: number of latent tokens to insert
        hidden_dim: model hidden dimension (e.g., 768 for GPT-2, 3072 for Llama-3B)
        init_scale: initialization scale for latent embeddings
        init_strategy: 'normal' (random) or 'mean' (initialized from mean of real embeddings)
    """

    def __init__(self, K: int, hidden_dim: int, init_scale: float = 0.02,
                 init_strategy: str = "normal"):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.init_strategy = init_strategy

        # Learnable latent embeddings — always float32 for numerical stability
        self.latent_embeddings = nn.Parameter(
            torch.randn(K, hidden_dim, dtype=torch.float32) * init_scale
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                embed_layer: nn.Embedding, split_positions: torch.Tensor = None,
                default_split_ratio: float = 0.5):
        """
        Inject latent tokens into the input sequence.

        Args:
            input_ids: (B, T) input token IDs
            attention_mask: (B, T) binary attention mask
            embed_layer: model's input embedding layer
            split_positions: (B,) per-sample split position (int).
                If None, uses default_split_ratio * seq_len.
            default_split_ratio: fallback split ratio when split_positions is None

        Returns:
            inputs_embeds: (B, T+K, D) tensor with latent tokens inserted
            new_attention_mask: (B, T+K) attention mask
            latent_mask: (B, T+K) binary mask where 1 = latent token position
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Get real token embeddings
        real_embeds = embed_layer(input_ids)  # (B, T, D)
        D = real_embeds.shape[-1]

        # Broadcast latent embeddings to batch, cast to model dtype
        latent = self.latent_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        latent = latent.to(dtype=real_embeds.dtype, device=device)

        # Determine split positions
        if split_positions is None:
            # Default: split at ratio of actual sequence length
            seq_lens = attention_mask.sum(dim=-1)  # (B,)
            split_positions = (seq_lens * default_split_ratio).long()

        # Build output sequences
        new_T = T + self.K
        inputs_embeds = torch.zeros(B, new_T, D, dtype=real_embeds.dtype, device=device)
        new_attention_mask = torch.zeros(B, new_T, dtype=attention_mask.dtype, device=device)
        latent_mask = torch.zeros(B, new_T, dtype=torch.bool, device=device)

        for b in range(B):
            sp = split_positions[b].item()
            sp = max(1, min(sp, T - 1))  # clamp to valid range

            # [prefix] + [latent] + [suffix]
            prefix = real_embeds[b, :sp]        # (sp, D)
            suffix = real_embeds[b, sp:]         # (T-sp, D)

            inputs_embeds[b, :sp] = prefix
            inputs_embeds[b, sp:sp + self.K] = latent[b]
            inputs_embeds[b, sp + self.K:] = suffix

            # Attention mask: latent tokens always attend
            prefix_mask = attention_mask[b, :sp]
            suffix_mask = attention_mask[b, sp:]
            new_attention_mask[b, :sp] = prefix_mask
            new_attention_mask[b, sp:sp + self.K] = 1  # latent tokens always active
            new_attention_mask[b, sp + self.K:] = suffix_mask

            # Latent mask
            latent_mask[b, sp:sp + self.K] = True

        return inputs_embeds, new_attention_mask, latent_mask

    def init_from_embeddings(self, embed_layer: nn.Embedding, tokenizer=None):
        """
        Optional: Initialize latent embeddings from the mean of real token embeddings.
        Call this once before training if init_strategy='mean'.
        """
        with torch.no_grad():
            weights = embed_layer.weight.data.float()
            mean_emb = weights.mean(dim=0)
            std_emb = weights.std(dim=0)
            for k in range(self.K):
                # Each latent token = mean + small perturbation
                self.latent_embeddings.data[k] = mean_emb + 0.1 * std_emb * torch.randn_like(mean_emb)


def compute_lm_loss_with_latent(logits: torch.Tensor, input_ids: torch.Tensor,
                                 latent_mask: torch.Tensor,
                                 attention_mask: torch.Tensor = None):
    """
    Compute language modeling loss, EXCLUDING latent token positions.

    labels[t] = the actual token at augmented position t.
    Latent positions get -100 (ignored). The standard shift
    (shift_logits = logits[:,:-1], shift_labels = labels[:,1:])
    handles next-token alignment automatically.

    Augmented sequence layout:
      pos:    0   1  ... sp-1  sp ... sp+K-1  sp+K ... sp+K+S-1
      token:  Q0  Q1 ... Qn    L0 ... LK-1    A0   ... Am
      label:  Q0  Q1 ... Qn   -100.. -100     A0   ... Am

    After shift: logits[t] is trained to predict labels[t+1].
    """
    B = logits.shape[0]
    T_orig = input_ids.shape[1]
    T_aug = logits.shape[1]
    K = T_aug - T_orig
    device = logits.device

    labels = torch.full((B, T_aug), -100, dtype=torch.long, device=device)

    for b in range(B):
        latent_positions = latent_mask[b].nonzero(as_tuple=True)[0]
        if len(latent_positions) == 0:
            # No latent tokens — standard LM labels
            labels[b, :T_orig] = input_ids[b]
        else:
            split_pos = latent_positions[0].item()

            # Prefix: labels = actual tokens at positions 0..sp-1
            labels[b, :split_pos] = input_ids[b, :split_pos]

            # Latent positions sp..sp+K-1: stay -100

            # Suffix: labels = actual tokens at positions sp+K..sp+K+(T-sp)-1
            suffix_len = T_orig - split_pos
            labels[b, split_pos + K:split_pos + K + suffix_len] = \
                input_ids[b, split_pos:split_pos + suffix_len]

    # Standard next-token shift (logits[t] → labels[t+1])
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def extract_hidden_for_str(hidden_states: torch.Tensor,
                            attention_mask: torch.Tensor,
                            latent_mask: torch.Tensor,
                            include_latent: bool = True):
    """
    Extract hidden states for STR computation.

    This is the critical junction for S+L interaction:
    - include_latent=True: STR is computed over the FULL trajectory
      including latent tokens. This means STR constrains the latent
      computation to be geometrically coherent with real tokens.
    - include_latent=False: STR only on real tokens (baseline comparison)

    Args:
        hidden_states: (B, T+K, D) from model output
        attention_mask: (B, T+K) augmented attention mask
        latent_mask: (B, T+K) latent token positions
        include_latent: whether to include latent tokens in STR trajectory

    Returns:
        list of (T_i, D) tensors, one per batch element
    """
    B = hidden_states.shape[0]
    trajectories = []

    for b in range(B):
        if include_latent:
            # Full trajectory: all attended positions
            mask = attention_mask[b].bool()
        else:
            # Real tokens only: attended AND not latent
            mask = attention_mask[b].bool() & ~latent_mask[b]

        h = hidden_states[b][mask]  # (T_valid, D)
        if h.shape[0] >= 2:
            trajectories.append(h)

    return trajectories


def find_question_end(input_ids: torch.Tensor, tokenizer, marker: str = "\nAnswer:"):
    """
    Find the position of the question-answer boundary for each sample.

    Args:
        input_ids: (B, T) token IDs
        tokenizer: tokenizer with decode method
        marker: text marker for the split point

    Returns:
        split_positions: (B,) tensor of split positions
    """
    B = input_ids.shape[0]
    # Encode the marker
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    marker_len = len(marker_ids)

    split_positions = []
    for b in range(B):
        ids = input_ids[b].tolist()
        found = False
        for i in range(len(ids) - marker_len + 1):
            if ids[i:i + marker_len] == marker_ids:
                split_positions.append(i + marker_len)
                found = True
                break
        if not found:
            # Fallback: split at 50%
            seq_len = (input_ids[b] != tokenizer.pad_token_id).sum().item()
            split_positions.append(seq_len // 2)

    return torch.tensor(split_positions, dtype=torch.long)
