# =============================================================================
# REVIEW NOTES (temporary, remove once the model is finished)
#
# Markers:
#   [KEEP]      stays as-is
#   [MODIFY]    needs the change described
#   [DELETE?]   proposed for deletion, pending your decision
#   [OPTIONAL]  possible improvement, not required for the experiment
# =============================================================================

import torch
import torch.nn as nn

from .block import TransformerBlock
from .config import Config

# positional encoding is left out
# [KEEP] Reasons for leaving it out (NoPE = "no positional encoding"):
#   - The experiment measures RAM and time per training step, not output quality.
#     A positional encoding would add the same small, linear-in-L cost to every
#     configuration (learned table: max_seq_len * D parameters + one (b, L, D)
#     addition; RoPE: a rotation of Q and K per layer). It does not change the
#     O(L^2) vs. O(L * W) difference in attention.
#   - The model still works without it: with a causal mask, decoder models learn
#     position implicitly (Haviv et al. 2022; Kazemnejad et al. 2023).
#   - Caveat: in local (sliding-window) layers this implicit signal is weak, since
#     every token sees the same number of keys. If losses of the 4:1 model and the
#     all-global baseline are ever compared, part of a gap may come from the
#     missing positional encoding, not from sliding-window attention itself.
#     A learned position embedding (lecture / GPT-2 style) could then be added
#     behind a config flag (~5 lines).
# [MODIFY] Move this explanation into the class docstring when finalizing.
# [MODIFY] Add a class docstring and comments in the style of attention.py / block.py:
#          embedding -> N blocks -> final norm -> lm_head, shapes, memory notes.
class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # [KEEP] Stored on the model; scripts/generate.py reads model.config.max_seq_len
        self.config = config
        # [KEEP] Token embedding: vocab_size x D table (24.6M parameters with the
        #        defaults V = 32000, D = 768). Output (b, L, D), linear in L.
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        # [KEEP] N transformer blocks; each decides itself whether it is global
        #        (block.is_global). Note (config, not this file): num_layers=12 gives
        #        LLLLGLLLLGLL; 10 or 15 layers give a clean 4:1 pattern.
        self.blocks = nn.ModuleList(
            TransformerBlock(config, layer_idx) for layer_idx in range(config.num_layers)
        )
        # [KEEP] Final LayerNorm: needed with pre-norm blocks, since the residual
        #        stream x itself is never normalized inside the blocks.
        self.final_norm = nn.LayerNorm(config.hidden_size)
        # [KEEP] Language modeling head: D -> vocab_size logits (next-token scores).
        #        Memory: the (b, L, vocab_size) logits are large, e.g. ~262 MB for
        #        b=1, L=2048, V=32000 (about as much as one global layer's attention
        #        weights with 12 heads), plus same-sized tensors in cross-entropy.
        #        Identical in all configurations, but it can hide the attention
        #        differences in total RAM -> vocab_size is a decision for the
        #        training setup (dataset/tokenizer), not for this file.
        # [OPTIONAL] Weight tying: `self.lm_head.weight = self.embed.weight` (as in
        #            GPT-2) saves vocab_size * D parameters. Same in all configurations,
        #            so irrelevant for the comparison.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # [OPTIONAL] Weight initialization: PyTorch defaults are used. GPT-2 style
        #            init (normal(0, 0.02), scaled residual projections) trains more
        #            stably in deep models. Irrelevant for RAM/time measurements.

    # [OPTIONAL] Helper for the report, e.g. `def num_parameters(self) -> int`
    #            (sum of p.numel() over self.parameters()), to state model sizes.

    # [OPTIONAL] Accept `targets` and return the loss as well (nanoGPT style), so
    #            train.py doesn't need to reshape logits for cross-entropy itself.
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [MODIFY] Comment the shapes: input_ids (b, L) token IDs -> logits (b, L, V).
        # [OPTIONAL] Check L <= config.max_seq_len. Without a positional embedding
        #            nothing in the model enforces it; max_seq_len is currently only
        #            used by generate.py for cropping.
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)
