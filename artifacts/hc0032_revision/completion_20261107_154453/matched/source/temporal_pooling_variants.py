"""Masked global/local ECAPA pooling variants.

The module wraps an already-loaded SpeechBrain ECAPA-TDNN.  Its convolutional
and attentive pooling weights are reused; only the post-MFA summary combination
is new.  ``original`` delegates to the untouched pretrained model.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class GlobalLocalPoolingECAPA(nn.Module):
    """Add masked overlapping local summaries after ECAPA's MFA layer."""

    def __init__(
        self,
        embedding_model: nn.Module,
        *,
        variant: str = "original",
        frame_shift_seconds: float = 0.01,
        local_window_seconds: float = 0.5,
        local_overlap: float = 0.5,
        gate_bias: float = -8.0,
    ) -> None:
        super().__init__()
        if variant not in {"original", "global_local", "global_local_duration"}:
            raise ValueError(f"Unknown pooling variant: {variant}")
        if not 0.0 <= local_overlap < 1.0:
            raise ValueError("local_overlap must be in [0, 1)")
        self.embedding_model = embedding_model
        self.variant = variant
        self.frame_shift_seconds = float(frame_shift_seconds)
        self.local_window_frames = max(
            1, int(round(local_window_seconds / frame_shift_seconds))
        )
        self.local_hop_frames = max(
            1, int(round(self.local_window_frames * (1.0 - local_overlap)))
        )
        summary_dim = int(embedding_model.mfa.conv.conv.out_channels)
        self.summary_dim = summary_dim * 2
        if variant != "original":
            gate_inputs = self.summary_dim * 2
            if variant == "global_local_duration":
                gate_inputs += 1
            self.gate = nn.Linear(gate_inputs, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, gate_bias)

    def _frame_features(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        frame_features = features.transpose(1, 2)
        block_outputs = []
        for block in self.embedding_model.blocks:
            try:
                frame_features = block(frame_features, lengths=lengths)
            except TypeError:
                frame_features = block(frame_features)
            block_outputs.append(frame_features)
        frame_features = torch.cat(block_outputs[1:], dim=1)
        return self.embedding_model.mfa(frame_features)

    @staticmethod
    def _valid_frames(lengths: torch.Tensor, frame_count: int) -> torch.Tensor:
        return (lengths.clamp(0.0, 1.0) * frame_count).round().long().clamp(
            min=1, max=frame_count
        )

    def _summary(self, frame_features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.embedding_model.asp(frame_features, lengths=lengths).squeeze(-1)

    def _local_summary(
        self, frame_features: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, frame_count = frame_features.shape
        valid_frames = self._valid_frames(lengths, frame_count)
        max_start = int(valid_frames.max().item())
        starts = list(range(0, max_start, self.local_hop_frames)) or [0]
        local_summaries = []
        local_presence = []
        for start in starts:
            stop = min(frame_count, start + self.local_window_frames)
            region = frame_features[:, :, start:stop]
            region_frames = (valid_frames - start).clamp(min=1, max=stop - start)
            region_lengths = region_frames.to(frame_features.dtype) / (stop - start)
            local_summaries.append(self._summary(region, region_lengths))
            local_presence.append((valid_frames > start).to(frame_features.dtype))
        summaries = torch.stack(local_summaries, dim=1)
        presence = torch.stack(local_presence, dim=1).unsqueeze(-1)
        return (summaries * presence).sum(dim=1) / presence.sum(dim=1).clamp_min(1.0)

    def _project_summary(self, summary: torch.Tensor) -> torch.Tensor:
        summary = self.embedding_model.asp_bn(summary.unsqueeze(-1))
        summary = self.embedding_model.fc(summary)
        return summary.transpose(1, 2).squeeze(1)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        disable_new_contribution: bool = False,
    ) -> torch.Tensor:
        if self.variant == "original" or disable_new_contribution:
            return self.embedding_model(features, lengths)
        if lengths is None:
            lengths = torch.ones(features.shape[0], device=features.device)
        frame_features = self._frame_features(features, lengths)
        global_summary = self._summary(frame_features, lengths)
        local_summary = self._local_summary(frame_features, lengths)
        gate_inputs = [global_summary, local_summary]
        if self.variant == "global_local_duration":
            valid_duration = lengths.clamp(0.0, 1.0).unsqueeze(-1)
            gate_inputs.append(valid_duration)
        gate = torch.sigmoid(self.gate(torch.cat(gate_inputs, dim=-1)))
        combined_summary = global_summary + gate * (local_summary - global_summary)
        return self._project_summary(combined_summary)


def load_state_dict_report(
    module: nn.Module, state_dict: dict[str, Any], *, strict: bool = False
) -> dict[str, list[str]]:
    """Load weights while making incompatibilities explicit."""
    result = module.load_state_dict(state_dict, strict=False)
    report = {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
    if strict and any(report.values()):
        raise RuntimeError(f"Checkpoint incompatibility: {report}")
    print(f"checkpoint_load_report={report}", flush=True)
    return report


def frame_shift_from_fbank(compute_features: nn.Module) -> float:
    """Recover Fbank frame shift from SpeechBrain's configured extractor."""
    for name in ("win hop", "hop_length", "frame_shift"):
        value = getattr(compute_features, name.replace(" ", "_"), None)
        if value is not None:
            return float(value) / 1000.0 if float(value) > 1.0 else float(value)
    return 0.01