"""Compact joint H&E/protein/RNA family; RNA states are regional mixtures.

No target measurements enter this module. Broad types group molecular states;
state IDs and type strings never become learned identity embeddings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .schema import Prediction, identifiers, image_backbone_identity


@dataclass(frozen=True)
class JointModelConfig:
    feature_dim: int
    state_dim: int
    gene_ids: tuple[str, ...]
    supervised_gene_ids: tuple[str, ...]
    normalization_gene_ids: tuple[str, ...]
    state_gene_ids: tuple[str, ...] = ()
    marker_ids: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ("H", "R", "P")
    hidden_dim: int = 32
    residual_rank: int = 2
    residual_bound: float = 0.5
    assignment_mode: str = "learned"
    transfer_only: bool = False
    image_encoder_manifest_sha256: str | None = None

    def __post_init__(self):
        for name in (
            "gene_ids",
            "supervised_gene_ids",
            "normalization_gene_ids",
            "state_gene_ids",
            "marker_ids",
            "modalities",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if set(self.modalities) not in (
            {"H"},
            {"H", "R"},
            {"H", "P"},
            {"R", "P"},
            {"H", "R", "P"},
        ) or len(set(self.modalities)) != len(self.modalities):
            raise ValueError("joint modalities must identify one of H, HR, HP, RP, HRP")
        image_backbone_identity(self.image_encoder_manifest_sha256, present="H" in self.modalities)
        for name in ("gene_ids", "supervised_gene_ids", "normalization_gene_ids"):
            identifiers(getattr(self, name), name)
        if not set(self.supervised_gene_ids) <= set(self.gene_ids):
            raise ValueError("supervised genes must belong to the output panel")
        if len(self.normalization_gene_ids) < 2 or not set(self.normalization_gene_ids) <= set(
            self.gene_ids
        ):
            raise ValueError("normalization needs at least two named output genes")
        for name, active in (
            ("feature_dim", "H" in self.modalities),
            ("state_dim", "R" in self.modalities),
        ):
            size = getattr(self, name)
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or (size < 1 if active else size != 0)
            ):
                raise ValueError(f"{name} must be positive when used, zero when absent")
        for name in ("hidden_dim", "residual_rank"):
            size = getattr(self, name)
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.residual_bound) or self.residual_bound <= 0:
            raise ValueError("residual_bound must be finite and positive")
        if "R" in self.modalities:
            identifiers(self.state_gene_ids, "state_gene_ids")
            if len(self.state_gene_ids) != self.state_dim:
                raise ValueError("state_gene_ids must name every molecular feature")
        elif self.state_gene_ids or self.transfer_only or self.assignment_mode != "learned":
            raise ValueError("RNA-free models cannot use RNA axes, transfer, or fixed mixtures")
        if "R" not in self.modalities and not set(self.normalization_gene_ids) <= set(
            self.supervised_gene_ids
        ):
            raise ValueError("direct prediction requires supervised normalization genes")
        if "P" in self.modalities:
            identifiers(self.marker_ids, "marker_ids")
        elif self.marker_ids:
            raise ValueError("protein-free models cannot carry a marker feature axis")
        if self.assignment_mode not in {"learned", "constant_reference"}:
            raise ValueError("unknown joint assignment mode")
        if not isinstance(self.transfer_only, bool):
            raise ValueError("transfer_only must be boolean")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping):
        return cls(**dict(values))


class JointModel(nn.Module):
    """Separate encoders, joint fusion, type/state matching, bounded correction."""

    family = "joint"

    def __init__(self, config: JointModelConfig):
        super().__init__()
        self.config = config
        width, markers = config.hidden_dim, len(config.marker_ids)
        self.image_encoder = (
            nn.Sequential(nn.Linear(config.feature_dim, width), nn.Tanh())
            if "H" in config.modalities
            else None
        )
        self.protein_encoder = (
            nn.Sequential(nn.Linear(4 * markers, width), nn.Tanh())
            if "P" in config.modalities
            else None
        )
        self.missing_protein = nn.Parameter(torch.zeros(width)) if markers else None
        self.fusion = nn.Sequential(nn.Linear(3 * width + 2 * markers, width), nn.Tanh())
        self.state_encoder = (
            nn.Sequential(nn.Linear(config.state_dim, width), nn.Tanh())
            if "R" in config.modalities
            else None
        )
        self.type_query = nn.Linear(width, width, bias=False) if self.state_encoder else None
        self.state_query = nn.Linear(width, width, bias=False) if self.state_encoder else None
        self.direct_head = (
            nn.Linear(width, len(config.gene_ids)) if not self.state_encoder else None
        )
        self.correction = nn.Linear(3 * width, config.residual_rank) if self.state_encoder else None
        if self.state_encoder:
            self.correction_gene = nn.Parameter(
                torch.empty(len(config.gene_ids), config.residual_rank)
            )
            nn.init.normal_(self.correction_gene, std=0.02)
        else:
            self.register_parameter("correction_gene", None)
        self.register_buffer(
            "supervised_mask",
            torch.tensor([gene in config.supervised_gene_ids for gene in config.gene_ids]),
        )
        if config.assignment_mode == "constant_reference":
            self.type_query.requires_grad_(False)
            self.state_query.requires_grad_(False)
        if config.transfer_only:
            self.correction.requires_grad_(False)
            self.correction_gene.requires_grad_(False)

    def _tensor(self, value):
        parameter = next(self.parameters())
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
        return torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)

    def _validate_inputs(self, inputs):
        for code, name in (("H", "image_features"), ("R", "reference"), ("P", "protein_features")):
            value = getattr(inputs, name)
            if code not in self.config.modalities and value is not None:
                raise ValueError(f"unused modality {code} must be explicitly absent")
            if code in self.config.modalities and code != "P" and value is None:
                raise ValueError(f"required modality {code} is missing")
        if inputs.image_encoder_manifest_sha256 != self.config.image_encoder_manifest_sha256:
            raise ValueError("image encoder manifest identity differs from joint model bundle")
        if (
            inputs.image_features is not None
            and inputs.image_features.shape[1] != self.config.feature_dim
        ):
            raise ValueError("image feature dimension differs from joint model")

    def _forward(self, inputs, *, correction_enabled=True):
        self._validate_inputs(inputs)
        config, n = self.config, len(inputs.cell_ids)
        zeros = self._tensor(np.zeros((n, config.hidden_dim), dtype=np.float32))
        h = self.image_encoder(self._tensor(inputs.image_features)) if self.image_encoder else zeros
        p = zeros
        coverage = zeros[:, :0]
        if self.protein_encoder is not None:
            m = len(config.marker_ids)
            protein = inputs.protein_features
            if protein is None:
                observed = np.zeros((n, m), dtype=bool)
                mean = q90 = counts = np.zeros((n, m))
            else:
                if tuple(protein.region_ids) != tuple(inputs.cell_ids):
                    raise ValueError("protein features must follow the declared region axis")
                lookup = {name: j for j, name in enumerate(protein.marker_ids)}
                if set(lookup) != set(config.marker_ids):
                    raise ValueError("protein feature marker panel differs from joint model")
                columns = [lookup[name] for name in config.marker_ids]
                observed = np.asarray(protein.observed)[:, columns]
                mean = np.where(observed, np.asarray(protein.mean)[:, columns], 0.0)
                q90 = np.where(observed, np.asarray(protein.q90)[:, columns], 0.0)
                counts = np.where(observed, np.asarray(protein.valid_counts)[:, columns], 0.0)
            mask, count = self._tensor(observed), self._tensor(np.log1p(counts))
            coverage = torch.cat((mask, count), dim=1)
            p = self.protein_encoder(
                torch.cat((self._tensor(mean), self._tensor(q90), coverage), 1)
            )
            available = mask.any(1, keepdim=True)
            p = torch.where(available, p, self.missing_protein.expand(n, -1))
        z = self.fusion(torch.cat((h, p, h * p, coverage), dim=1))
        diagnostics = {"image_embedding": h, "protein_embedding": p, "fused_embedding": z}
        if self.state_encoder is None:
            transfer = zeros.new_zeros((n, len(config.gene_ids)))
            delta = torch.zeros_like(transfer)
            expression = F.softplus(self.direct_head(z)) * self.supervised_mask
            q = zeros.new_zeros((n, 0))
            assignment = zeros.new_ones((n, 1))
            state_ids = ("unsupported",)
        else:
            reference = inputs.reference
            gene_lookup = {name: i for i, name in enumerate(reference.gene_ids)}
            state_lookup = {name: i for i, name in enumerate(reference.embedding_gene_ids)}
            if not set(config.gene_ids) <= set(gene_lookup) or not set(
                config.state_gene_ids
            ) <= set(state_lookup):
                raise ValueError(
                    "reference output or embedding gene panel differs from joint model"
                )
            r = self.state_encoder(
                self._tensor(reference.embedding)[
                    :, [state_lookup[g] for g in config.state_gene_ids]
                ]
            )
            proportions = self._tensor(reference.proportions)
            proportions = proportions / proportions.sum()
            type_ids = tuple(sorted(set(reference.state_types)))
            groups = [
                torch.tensor([t == name for t in reference.state_types], device=r.device)
                for name in type_ids
            ]
            type_features = []
            for group in groups:
                weights = proportions[group]
                # Zero-prior types still have molecular descriptors and can be assigned.
                weights = (
                    weights / weights.sum()
                    if weights.sum() > 0
                    else torch.ones_like(weights) / len(weights)
                )
                type_features.append((r[group] * weights[:, None]).sum(0))
            rt = torch.stack(type_features)
            if config.assignment_mode == "constant_reference":
                q = proportions.expand(n, -1)
                pi = torch.stack([q[:, group].sum(1) for group in groups], dim=1)
            else:
                pi = (self.type_query(z) @ rt.T / config.hidden_dim**0.5).softmax(1)
                logits = self.state_query(z) @ r.T / config.hidden_dim**0.5
                columns = [None] * len(reference.state_ids)
                for index, group in enumerate(groups):
                    within = logits[:, group].softmax(1) * pi[:, index : index + 1]
                    for local, column in enumerate(torch.where(group)[0].tolist()):
                        columns[column] = within[:, local]
                q = torch.stack(columns, dim=1)
            b = self._tensor(reference.expression)[:, [gene_lookup[g] for g in config.gene_ids]]
            transfer = q @ b
            u = q @ r
            summary = (proportions @ r).expand(n, -1)
            if config.transfer_only or not correction_enabled:
                delta = torch.zeros_like(transfer)
                expression = transfer
            else:
                factors = self.correction(torch.cat((z, u, summary), dim=1))
                delta = config.residual_bound * torch.tanh(factors @ self.correction_gene.T)
                delta = delta * self.supervised_mask
                expression = torch.where(
                    self.supervised_mask, (transfer + 1e-8) * delta.exp(), transfer
                )
            assignment = torch.cat((q.new_zeros((n, 1)), q), 1)
            state_ids = ("unsupported", *reference.state_ids)
            diagnostics.update(
                type_ids=type_ids,
                type_mixture=pi,
                state_embedding=r,
                assigned_rna_embedding=u,
                specimen_rna_embedding=summary,
            )
        prediction = Prediction(
            cell_ids=inputs.cell_ids,
            gene_ids=config.gene_ids,
            state_ids=state_ids,
            coordinates_um=self._tensor(inputs.coordinates_um),
            assignment=assignment,
            conditional_assignment=q,
            transfer=transfer,
            log_correction=delta,
            expression=expression,
            assignment_entropy=-(assignment * assignment.clamp_min(1e-12).log()).sum(1),
            reference_support=1 - assignment[:, 0],
            registration_sensitivity=zeros.new_full((n,), float("nan")),
            supervised_genes=self.supervised_mask.clone(),
            independently_evaluated_genes=torch.zeros_like(self.supervised_mask),
            specimen_id=inputs.specimen_id,
            patient_id=inputs.patient_id,
            prediction_unit=inputs.prediction_unit,
            coordinate_units=inputs.coordinate_units,
            reference_support_calibrated=False,
            unsupported_policy="conditional_reference_unvalidated"
            if self.state_encoder
            else "direct_multimodal",
        )
        return prediction, diagnostics

    def forward(self, inputs, *, correction_enabled=True):
        return self._forward(inputs, correction_enabled=correction_enabled)[0]

    def diagnostic_arrays(self, inputs, *, correction_enabled=True):
        """Numeric/Unicode sidecar; no calibrated cell-fraction or support claim."""
        _, values = self._forward(inputs, correction_enabled=correction_enabled)
        return {
            name: value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
            for name, value in values.items()
        }

    def active_trainable_parameter_count(self):
        count = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        width = self.config.hidden_dim
        # The fusion layout is shared, but absent modality columns never activate.
        unused_blocks = (
            2 if "H" not in self.config.modalities or "P" not in self.config.modalities else 0
        )
        count -= unused_blocks * width * width
        omitted = len(self.config.gene_ids) - len(self.config.supervised_gene_ids)
        if self.correction_gene is not None and self.correction_gene.requires_grad:
            count -= omitted * self.config.residual_rank
        if self.direct_head is not None:
            count -= omitted * (self.config.hidden_dim + 1)
        return count
