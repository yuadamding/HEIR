"""Optional scVI/scANVI adapter with no import-time heavy dependency."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ..expression import EXPRESSION_SPACE_ID, EXPRESSION_TARGET_SUM
from ..models.rna import RNAVAE, RNAVAEConfig
from ..utils import optional_import_error, resolve_device

SCVI_DISTILLED_DECODER_SCHEMA = "heir.scvi_distilled_decoder.v3"
SCVI_EXPRESSION_NORMALIZATION_CONTRACT = "full_library_10000_then_panel_log1p_v2"
SCVI_BATCH_CORRECTION_MODES = frozenset({"none", "reference_batch_marginalization"})


def _array_sha256(values: np.ndarray) -> str:
    """Hash an array's exact shape, dtype, and contiguous byte content."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


class SCVIAdapter:
    """Fit the blueprint's default count model when ``scvi-tools`` is installed.

    The core HEIR tests use the lightweight RNA VAE. This adapter exists so a
    production run can use scVI/scANVI without coupling every preprocessing or
    inference command to that heavyweight environment.
    """

    def __init__(self, latent_dim: int = 32, likelihood: str = "nb") -> None:
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if likelihood not in {"nb", "zinb"}:
            raise ValueError("likelihood must be nb or zinb")
        self.latent_dim = latent_dim
        self.likelihood = likelihood
        self.model: Optional[Any] = None

    @staticmethod
    def validate_crossed_technical_batch(
        section_values: Sequence[object],
        batch_values: Sequence[object],
        *,
        key: str,
    ) -> Tuple[str, ...]:
        """Require an estimable technical design crossed with specimens."""

        normalized_key = str(key).strip().lower()
        reserved = {
            "section_id",
            "sample_id",
            "donor_id",
            "specimen_id",
            "block_id",
            "source_cell_id",
            "source_row",
            "major_annotation",
            "cell_type",
            "disease",
            "tissue",
            "anatomic_region",
        }
        if not normalized_key or normalized_key in reserved:
            raise ValueError("technical batch key names identity or biological information")
        sections_raw = np.asarray(section_values, dtype=object)
        batches_raw = np.asarray(batch_values, dtype=object)
        if sections_raw.ndim != 1 or batches_raw.shape != sections_raw.shape:
            raise ValueError("technical batch values must align one-to-one with specimens")

        def clean(values: np.ndarray, name: str) -> np.ndarray:
            result = []
            for value in values.tolist():
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    raise ValueError("%s contains missing values" % name)
                text = str(value).strip()
                if not text or text.lower() in {"nan", "none", "null"}:
                    raise ValueError("%s contains missing or empty values" % name)
                result.append(text)
            return np.asarray(result, dtype=np.str_)

        sections = clean(sections_raw, "section identity")
        batches = clean(batches_raw, "technical batch key")
        section_levels = tuple(sorted(set(sections.tolist())))
        batch_levels = tuple(sorted(set(batches.tolist())))
        if len(section_levels) < 2 or len(batch_levels) < 2:
            raise ValueError("technical batch correction requires at least two crossed levels")
        if any(
            set(sections[batches == level].tolist()) != set(section_levels)
            for level in batch_levels
        ):
            raise ValueError("every technical batch level must occur in every specimen")
        if any(len(set(batches[sections == level].tolist())) < 2 for level in section_levels):
            raise ValueError("every specimen must contain at least two technical batch levels")
        return batch_levels

    @staticmethod
    def _module() -> Any:
        try:
            import scvi  # type: ignore[import-not-found]
        except ImportError as error:
            raise optional_import_error("scvi-tools", "science") from error
        return scvi

    def fit(
        self,
        adata: Any,
        batch_key: Optional[str] = None,
        categorical_covariate_keys: Optional[Sequence[str]] = None,
        labels_key: Optional[str] = None,
        max_epochs: int = 400,
    ) -> "SCVIAdapter":
        scvi = self._module()
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key=batch_key,
            categorical_covariate_keys=list(categorical_covariate_keys or []),
        )
        base = scvi.model.SCVI(
            adata,
            n_latent=self.latent_dim,
            gene_likelihood=self.likelihood,
            n_layers=2,
            n_hidden=256,
        )
        base.train(max_epochs=max_epochs, check_val_every_n_epoch=1)
        if labels_key:
            scvi.model.SCANVI.setup_anndata(
                adata,
                batch_key=batch_key,
                labels_key=labels_key,
                categorical_covariate_keys=list(categorical_covariate_keys or []),
            )
            self.model = scvi.model.SCANVI.from_scvi_model(
                base,
                unlabeled_category="unknown",
                labels_key=labels_key,
            )
            self.model.train(max_epochs=max_epochs)
        else:
            self.model = base
        return self

    def latent(self, adata: Optional[Any] = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit or load the RNA model first")
        return np.asarray(self.model.get_latent_representation(adata), dtype=np.float32)

    def load(
        self,
        path: str,
        adata: Any,
        model_type: str = "scvi",
    ) -> "SCVIAdapter":
        """Load a native scVI or scANVI checkpoint in its owning environment."""

        scvi = self._module()
        normalized = model_type.strip().lower()
        if normalized == "scvi":
            self.model = scvi.model.SCVI.load(path, adata=adata)
        elif normalized == "scanvi":
            self.model = scvi.model.SCANVI.load(path, adata=adata)
        else:
            raise ValueError("model_type must be scvi or scanvi")
        inferred = getattr(self.model, "n_latent", self.latent_dim)
        if int(inferred) != self.latent_dim:
            raise ValueError("loaded scVI latent width differs from adapter latent_dim")
        return self

    def normalized_expression(
        self,
        adata: Optional[Any] = None,
        gene_list: Optional[Sequence[str]] = None,
        transform_batch: Optional[Sequence[object]] = None,
        batch_correction_mode: str = "reference_batch_marginalization",
        posterior_samples: int = 32,
    ) -> np.ndarray:
        """Return panel expression normalized against the full decoded library.

        scVI performs its 10,000-count scaling before selecting ``gene_list``.
        HEIR preserves that denominator and applies only ``log1p`` here.
        """

        if self.model is None:
            raise RuntimeError("fit or load the RNA model first")
        mode = str(batch_correction_mode).strip().lower()
        if mode not in SCVI_BATCH_CORRECTION_MODES:
            raise ValueError(
                "batch_correction_mode must be none or reference_batch_marginalization"
            )
        if isinstance(transform_batch, (str, bytes)):
            raise TypeError("transform_batch must be a sequence of batch identifiers")
        reference_batches = () if transform_batch is None else tuple(transform_batch)
        if any(not str(value).strip() for value in reference_batches):
            raise ValueError("transform_batch cannot contain empty batch identifiers")
        if isinstance(posterior_samples, bool) or not isinstance(posterior_samples, int):
            raise TypeError("posterior_samples must be an integer")
        if posterior_samples <= 0:
            raise ValueError("posterior_samples must be positive")
        if mode == "reference_batch_marginalization" and not reference_batches:
            raise ValueError(
                "transform_batch must name prespecified reference batches when batch "
                "marginalization is enabled"
            )
        if mode == "none" and reference_batches:
            raise ValueError("transform_batch must be empty when batch_correction_mode is none")
        genes = None if gene_list is None else tuple(str(value) for value in gene_list)
        if genes is not None and (
            not genes or len(set(genes)) != len(genes) or any(not value.strip() for value in genes)
        ):
            raise ValueError("gene_list must contain unique non-empty genes")
        normalization_options = {
            "gene_list": None if genes is None else list(genes),
            "library_size": EXPRESSION_TARGET_SUM,
            "n_samples": posterior_samples,
            "return_mean": True,
            "return_numpy": False,
        }
        # Omitting ``transform_batch`` is semantically different from passing a
        # list of every specimen: the latter explicitly averages away the batch
        # effect.  Keep the choice visible and hashable in exported metadata.
        if mode == "reference_batch_marginalization":
            normalization_options["transform_batch"] = list(reference_batches)
        values = self.model.get_normalized_expression(adata, **normalization_options)
        if hasattr(values, "columns"):
            columns = [str(value) for value in values.columns]
            if genes is not None:
                missing = sorted(set(genes) - set(columns))
                if missing:
                    raise ValueError("scVI output is missing genes: %s" % ", ".join(missing))
                matrix = np.asarray(values.loc[:, list(genes)], dtype=np.float32)
            else:
                matrix = np.asarray(values, dtype=np.float32)
        else:
            matrix = np.asarray(values, dtype=np.float32)
            if genes is not None:
                source_adata = adata if adata is not None else getattr(self.model, "adata", None)
                if source_adata is None or not hasattr(source_adata, "var_names"):
                    raise ValueError("cannot verify scVI gene order without AnnData var_names")
                requested = set(genes)
                returned_genes = [
                    str(value) for value in source_adata.var_names if str(value) in requested
                ]
                if matrix.shape[1] != len(returned_genes) or set(returned_genes) != requested:
                    raise ValueError("scVI output genes do not match gene_list")
                lookup = {name: index for index, name in enumerate(returned_genes)}
                matrix = matrix[:, [lookup[name] for name in genes]]
        if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("scVI returned invalid normalized expression")
        # scVI applies ``library_size`` to the complete decoded transcriptome
        # before selecting ``gene_list``.  Keep that full-library denominator:
        # renormalizing the selected columns would put scVI-derived targets on
        # a different scale from RNAReference and locked spatial truth.
        return np.log1p(matrix).astype(np.float32)

    def distill_transferable_decoder(
        self,
        adata: Any,
        gene_list: Sequence[str],
        validation_mask: np.ndarray,
        decoder_hidden_dims: Tuple[int, ...] = (128, 256),
        max_epochs: int = 200,
        batch_size: int = 1024,
        learning_rate: float = 1.0e-3,
        patience: int = 20,
        seed: int = 17,
        device: str = "auto",
        transform_batch: Optional[Sequence[object]] = None,
        batch_correction_mode: str = "reference_batch_marginalization",
        posterior_samples: int = 32,
        latent_target: Optional[np.ndarray] = None,
        expression_target: Optional[np.ndarray] = None,
    ) -> RNAVAE:
        """Distill the fitted scVI mean decoder into HEIR's portable decoder.

        scVI's native generative module carries batch/covariate state and is
        not safely portable across the isolated RNA and image environments.
        This method fits a topology-compatible decoder to frozen scVI latent /
        normalized-expression pairs. ``validation_mask`` is mandatory so the
        caller must make the split explicitly (normally by held-out donor).
        The returned VAE's encoder is untrained and must not be used.
        """

        if self.model is None:
            raise RuntimeError("fit or load the RNA model first")
        genes = tuple(str(value) for value in gene_list)
        if not genes or len(set(genes)) != len(genes):
            raise ValueError("gene_list must contain unique non-empty genes")
        if any(not value.strip() for value in genes):
            raise ValueError("gene_list cannot contain empty genes")
        if max_epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or patience <= 0:
            raise ValueError("decoder training settings must be positive")
        if isinstance(posterior_samples, bool) or not isinstance(posterior_samples, int):
            raise TypeError("posterior_samples must be an integer")
        if posterior_samples <= 0:
            raise ValueError("posterior_samples must be positive")
        latent = (
            self.latent(adata)
            if latent_target is None
            else np.asarray(latent_target, dtype=np.float32)
        )
        target = (
            self.normalized_expression(
                adata,
                genes,
                transform_batch=transform_batch,
                batch_correction_mode=batch_correction_mode,
                posterior_samples=posterior_samples,
            )
            if expression_target is None
            else np.asarray(expression_target, dtype=np.float32)
        )
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("precomputed scVI latent target has the wrong shape")
        if not np.isfinite(latent).all():
            raise ValueError("precomputed scVI latent target must be finite")
        if target.ndim != 2 or not np.isfinite(target).all() or np.any(target < 0):
            raise ValueError("precomputed scVI expression target is invalid")
        if target.shape != (latent.shape[0], len(genes)):
            raise ValueError("scVI decoder output does not match cells and gene_list")
        held_out = np.asarray(validation_mask, dtype=bool)
        if held_out.shape != (latent.shape[0],):
            raise ValueError("validation_mask must contain one value per RNA cell")
        if not held_out.any() or held_out.all():
            raise ValueError("validation_mask must leave non-empty train and validation donors")

        import torch
        from torch.nn import functional as F

        torch.manual_seed(seed)
        target_device = resolve_device(device)
        model = RNAVAE(
            RNAVAEConfig(
                input_dim=len(genes),
                latent_dim=self.latent_dim,
                hidden_dims=tuple(reversed(decoder_hidden_dims)),
                decoder_hidden_dims=decoder_hidden_dims,
                nonnegative_output=True,
            )
        ).to(target_device)
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        latent_tensor = torch.from_numpy(latent).to(target_device)
        target_tensor = torch.from_numpy(target).to(target_device)
        train_indices = torch.from_numpy(np.flatnonzero(~held_out)).long()
        validation_indices = torch.from_numpy(np.flatnonzero(held_out)).long().to(target_device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        optimizer = torch.optim.AdamW(
            model.decoder.parameters(), learning_rate, weight_decay=1.0e-4
        )
        best_loss = float("inf")
        best_state = None
        stale = 0
        for _ in range(max_epochs):
            order = train_indices[torch.randperm(len(train_indices), generator=generator)]
            model.decoder.train()
            for start in range(0, len(order), batch_size):
                selected = order[start : start + batch_size].to(target_device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model.decoder(latent_tensor.index_select(0, selected))
                loss = F.smooth_l1_loss(prediction, target_tensor.index_select(0, selected))
                loss.backward()
                optimizer.step()
            model.decoder.eval()
            with torch.no_grad():
                validation_prediction = model.decoder(
                    latent_tensor.index_select(0, validation_indices)
                )
                validation_loss = float(
                    F.smooth_l1_loss(
                        validation_prediction,
                        target_tensor.index_select(0, validation_indices),
                    ).cpu()
                )
            if validation_loss < best_loss - 1.0e-8:
                best_loss = validation_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.decoder.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError("scVI decoder distillation did not produce a checkpoint")
        model.decoder.load_state_dict(best_state)
        model.freeze_decoder(True)
        return model.cpu()

    def export_transferable_decoder_checkpoint(
        self,
        path: str,
        adata: Any,
        gene_list: Sequence[str],
        validation_mask: np.ndarray,
        training_donors: Sequence[str],
        latent_space_id: str,
        transform_batch: Optional[Sequence[object]] = None,
        batch_correction_mode: str = "reference_batch_marginalization",
        **training_options: Any,
    ) -> RNAVAE:
        """Distill and atomically export the metadata required by ``heir train``."""

        donors = sorted(set(str(value).strip() for value in training_donors))
        if not donors or any(not value for value in donors):
            raise ValueError("training_donors must contain non-empty donor IDs")
        if not latent_space_id.strip():
            raise ValueError("latent_space_id is required")
        mode = str(batch_correction_mode).strip().lower()
        if mode not in SCVI_BATCH_CORRECTION_MODES:
            raise ValueError(
                "batch_correction_mode must be none or reference_batch_marginalization"
            )
        if isinstance(transform_batch, (str, bytes)):
            raise TypeError("transform_batch must be a sequence of batch identifiers")
        raw_reference_batches = () if transform_batch is None else tuple(transform_batch)
        reference_batches = tuple(str(value).strip() for value in raw_reference_batches)
        if any(not value for value in reference_batches):
            raise ValueError("transform_batch cannot contain empty reference batches")
        if mode == "reference_batch_marginalization" and not reference_batches:
            raise ValueError(
                "transform_batch must contain reference batches when marginalization is enabled"
            )
        if mode == "none" and reference_batches:
            raise ValueError("transform_batch must be empty when batch_correction_mode is none")
        posterior_samples = training_options.pop("posterior_samples", 32)
        latent_target = training_options.pop("latent_target", None)
        expression_target = training_options.pop("expression_target", None)
        if latent_target is None:
            latent_target = self.latent(adata)
        else:
            latent_target = np.asarray(latent_target, dtype=np.float32)
        if expression_target is None:
            expression_target = self.normalized_expression(
                adata,
                gene_list,
                transform_batch=reference_batches,
                batch_correction_mode=mode,
                posterior_samples=posterior_samples,
            )
        else:
            expression_target = np.asarray(expression_target, dtype=np.float32)
        held_out = np.asarray(validation_mask, dtype=bool)
        model = self.distill_transferable_decoder(
            adata,
            gene_list,
            held_out,
            transform_batch=reference_batches,
            batch_correction_mode=mode,
            posterior_samples=posterior_samples,
            latent_target=latent_target,
            expression_target=expression_target,
            **training_options,
        )
        checkpoint = model.checkpoint()
        checkpoint["metadata"] = {
            "schema": SCVI_DISTILLED_DECODER_SCHEMA,
            "gene_names": [str(value) for value in gene_list],
            "training_donors": donors,
            "latent_space_id": latent_space_id,
            "expression_space_id": EXPRESSION_SPACE_ID,
            "expression_normalization_contract": SCVI_EXPRESSION_NORMALIZATION_CONTRACT,
            "expression_normalization": {
                "method": "scvi.get_normalized_expression",
                "library_size": EXPRESSION_TARGET_SUM,
                "library_basis": "full-transcriptome",
                "gene_selection": "after-library-normalization",
                "transform": "log1p",
                "version": 2,
            },
            "transform_batch": list(reference_batches),
            "batch_correction_mode": mode,
            "posterior_samples": posterior_samples,
            "distillation_latent_sha256": _array_sha256(latent_target),
            "distillation_target_sha256": _array_sha256(expression_target),
            "validation_mask_sha256": _array_sha256(held_out),
            "decoder_only": True,
        }
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=destination.name + ".",
            suffix=".pt.tmp",
            dir=str(destination.parent),
        )
        os.close(descriptor)
        try:
            import torch

            torch.save(checkpoint, temporary)
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return model

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("there is no fitted model to save")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(destination), overwrite=True, save_anndata=False)
