"""
GeoPillars W&B Logging Hook for mmengine Runner
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

import torch

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

if TYPE_CHECKING:
    from mmengine.runner import Runner

logger = logging.getLogger(__name__)


@HOOKS.register_module()
class GeoPillarsWandbHook(Hook):
    """
    Custom hook for logging Geo-Pillars–specific metrics via mmengine's `MessageHub` + `WandbVisBackend` pipeline.
    """

    priority = "BELOW_NORMAL"  # 60

    def __init__(
        self,
        log_interval: int = 50,
        histogram_interval: int = 500,
        ngfm_normal_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.log_interval = log_interval
        self.histogram_interval = histogram_interval
        self.ngfm_normal_weight = ngfm_normal_weight

        # Activation caches
        self._mask_mean: float = 0.0
        self._mask_std: float = 0.0

        # Histogram numpy arrays
        self._mask_hist_np: Optional[Any] = None

        # Forward hook handles
        self._hook_handles: list = []

        # CSV writer state
        self._csv_file = None
        self._csv_writer = None
        self._csv_columns: list = []

    def _mask_expand_hook(self, module, input, output):
        """Capture spatial mask statistics.  Output shape: (B, C, H, W)"""
        with torch.no_grad():
            t = output.detach().float()
            self._mask_mean = t.mean().item()
            self._mask_std = t.std().item()

            if self._should_log_histogram():

                sampled = t[:, ::4, ::2, ::2]
                self._mask_hist_np = sampled.cpu().numpy().ravel()

    def _should_log_histogram(self) -> bool:
        """Check if current iter aligns with histogram interval."""
        return (self._current_iter + 1) % self.histogram_interval == 0

    # CSV helpers
    def _init_csv(self, work_dir: str) -> None:
        """Open (or create) the CSV file for metric logging."""
        csv_path = os.path.join(work_dir, "metrics.csv")
        file_exists = os.path.exists(csv_path)

        # Open in append mode so resumed runs don't clobber prior data
        self._csv_file = open(csv_path, "a", newline="", buffering=1)
        self._csv_writer = None  # Created on first write (to learn columns)
        self._csv_columns = []

        if file_exists:
            # Read existing columns from header
            try:
                with open(csv_path, "r") as f:
                    header = f.readline().strip()
                    if header:
                        self._csv_columns = header.split(",")
            except Exception:
                pass

        logger.info(f"GeoPillarsWandbHook: CSV logging to {csv_path}")

    def _write_csv_row(self, metrics: Dict[str, Any]) -> None:
        """Append a single row to the CSV file."""
        if self._csv_file is None or self._csv_file.closed:
            return

        # On first write (or if columns changed), set up the writer
        new_columns = [k for k in metrics.keys() if k not in self._csv_columns]

        if not self._csv_columns or new_columns:
            # Need to rewrite header, but only if file is fresh
            if not self._csv_columns:
                # First write: establish columns
                self._csv_columns = list(metrics.keys())
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=self._csv_columns,
                    extrasaction="ignore",
                )
                self._csv_writer.writeheader()
            else:
                # Columns expanded mid-run, add new ones
                self._csv_columns.extend(new_columns)

                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=self._csv_columns,
                    extrasaction="ignore",
                )
        elif self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._csv_columns,
                extrasaction="ignore",
            )

        try:
            self._csv_writer.writerow(metrics)
            self._csv_file.flush()
        except Exception as e:
            logger.warning(f"GeoPillarsWandbHook: CSV write failed: {e}")

    # Hook lifecycle
    def _find_ngfm_module(self, runner: "Runner"):
        """Walk the model tree to find the NormalGuidedFeatureModulation."""
        model = runner.model
        # Handle DDP / DataParallel wrapping
        if hasattr(model, "module"):
            model = model.module
        # GeoPillarsDetector -> inner (GeoPillarsModel) -> ngfm
        inner = getattr(model, "inner", None)
        if inner is None:
            logger.warning(
                "GeoPillarsWandbHook: Could not find model.inner - "
                "NGFM diagnostics will be disabled."
            )
            return None
        ngfm = getattr(inner, "ngfm", None)
        if ngfm is None:
            logger.warning(
                "GeoPillarsWandbHook: Could not find model.inner.ngfm - "
                "NGFM diagnostics will be disabled."
            )
        return ngfm

    def before_train(self, runner: "Runner") -> None:
        """Register forward hooks, log one-time model summary, init CSV."""
        self._current_iter = 0

        # Register forward hooks on NGFM sub-modules
        ngfm = self._find_ngfm_module(runner)
        if ngfm is not None:
            h1 = ngfm.mask_expand.register_forward_hook(self._mask_expand_hook)
            self._hook_handles.append(h1)
            logger.info(
                "GeoPillarsWandbHook: Registered forward hook on " "NGFM.mask_expand"
            )

        # One-time model summary -> MessageHub
        model = runner.model
        if hasattr(model, "module"):
            model = model.module

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_m = total_params / 1e6
        trainable_m = trainable_params / 1e6

        # Push to MessageHub so it appears in logs
        runner.message_hub.update_info("summary/params_M", params_m)
        runner.message_hub.update_info("summary/trainable_params_M", trainable_m)

        # Also push to wandb.summary
        try:
            import wandb

            if wandb.run is not None:
                wandb.run.summary["summary/params_M"] = params_m
                wandb.run.summary["summary/trainable_params_M"] = trainable_m

                try:
                    import json

                    cfg_str = json.dumps(
                        runner.cfg.to_dict(),
                        default=str,
                        skipkeys=True,
                    )
                    safe_cfg = json.loads(cfg_str)
                    wandb.config.update(safe_cfg, allow_val_change=True)
                except Exception as cfg_err:
                    logger.warning(
                        f"GeoPillarsWandbHook: Config upload to W&B failed "
                        f"(non-fatal): {cfg_err}"
                    )

                logger.info(
                    f"GeoPillarsWandbHook: Logged model summary to W&B - "
                    f"{params_m:.2f}M params ({trainable_m:.2f}M trainable)"
                )
        except ImportError:
            logger.warning(
                "GeoPillarsWandbHook: wandb not installed, " "skipping summary logging."
            )
        except Exception as e:
            logger.warning(f"GeoPillarsWandbHook: wandb summary failed: {e}")

        # Init CSV file
        work_dir = runner.work_dir
        os.makedirs(work_dir, exist_ok=True)
        self._init_csv(work_dir)

    def after_train_iter(
        self,
        runner: "Runner",
        batch_idx: int,
        data_batch: Any = None,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Push NGFM diagnostics and GPU stats into MessageHub + CSV."""
        self._current_iter = runner.iter

        if (runner.iter + 1) % self.log_interval != 0:
            return

        hub = runner.message_hub
        log_dict: Dict[str, float] = {"global_step": runner.iter + 1}

        # NGFM diagnostics (populated by forward hooks)
        if self._hook_handles:
            ngfm_metrics = {
                "train/ngfm/spatial_mask_mean": self._mask_mean,
                "train/ngfm/spatial_mask_std": self._mask_std,
            }

            # Cosine similarity (back-computed from weighted NGFM loss)
            if outputs is not None and "loss_ngfm" in outputs:
                ngfm_loss_val = outputs["loss_ngfm"]
                if isinstance(ngfm_loss_val, torch.Tensor):
                    ngfm_loss_val = ngfm_loss_val.detach().item()
                if self.ngfm_normal_weight > 0:
                    cos_sim = 1.0 - (ngfm_loss_val / self.ngfm_normal_weight)
                    ngfm_metrics["train/ngfm/cos_sim_mean"] = cos_sim

            # Push to MessageHub -> LoggerHook -> WandbVisBackend
            for key, value in ngfm_metrics.items():
                hub.update_scalar(key, value)

            log_dict.update(ngfm_metrics)

        # Standard losses from outputs
        if outputs is not None:
            for key, val in outputs.items():
                if isinstance(val, torch.Tensor) and val.numel() == 1:
                    scalar_val = val.detach().item()
                    hub.update_scalar(f"train/{key}", scalar_val)
                    log_dict[f"train/{key}"] = scalar_val
                elif isinstance(val, (int, float)):
                    hub.update_scalar(f"train/{key}", val)
                    log_dict[f"train/{key}"] = val

        # GPU system metrics
        if torch.cuda.is_available():
            try:
                device = next(runner.model.parameters()).device
                if device.type == "cuda":
                    idx = device.index or 0
                    mem_alloc = torch.cuda.memory_allocated(idx) / (1024**2)
                    mem_reserved = torch.cuda.memory_reserved(idx) / (1024**2)

                    hub.update_scalar("train/system/gpu_mem_alloc_mb", mem_alloc)
                    hub.update_scalar("train/system/gpu_mem_reserved_mb", mem_reserved)
                    log_dict["train/system/gpu_mem_alloc_mb"] = mem_alloc
                    log_dict["train/system/gpu_mem_reserved_mb"] = mem_reserved

                    try:
                        import pynvml

                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        hub.update_scalar("train/system/gpu_util_pct", util.gpu)
                        log_dict["train/system/gpu_util_pct"] = util.gpu
                    except Exception:
                        pass  # pynvml not installed
            except StopIteration:
                pass  # No parameters on device yet

        # Write to CSV
        self._write_csv_row(log_dict)

        if (runner.iter + 1) % self.histogram_interval == 0:
            self._log_histograms(runner)

    def _log_histograms(self, runner: "Runner") -> None:
        """Log NGFM activation histograms directly to wandb."""
        try:
            import wandb

            if wandb.run is None:
                return

            log_dict = {}

            if self._mask_hist_np is not None:
                log_dict["ngfm/spatial_mask_hist"] = wandb.Histogram(self._mask_hist_np)
                self._mask_hist_np = None  # Free memory

            if log_dict:
                # NO step= parameter let WandbVisBackend own the step counter
                wandb.log(log_dict, commit=False)

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"GeoPillarsWandbHook: Histogram logging failed: {e}")

    def after_val_epoch(
        self,
        runner: "Runner",
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log validation metrics (NDS, mAP, per-class AP)."""
        self._log_eval_metrics(runner, metrics, prefix="val")

    def after_test_epoch(
        self,
        runner: "Runner",
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log test metrics (NDS, mAP, per-class AP)."""
        self._log_eval_metrics(runner, metrics, prefix="test")

    def _log_eval_metrics(
        self,
        runner: "Runner",
        metrics: Optional[Dict[str, float]],
        prefix: str,
    ) -> None:
        """Push evaluation metrics into MessageHub + CSV."""
        if metrics is None:
            return

        hub = runner.message_hub
        csv_row: Dict[str, Any] = {
            "global_step": runner.iter + 1,
            "epoch": runner.epoch + 1,
            "phase": prefix,
        }

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_key = key.replace("/", "_")
                metric_name = f"{prefix}/{clean_key}"
                hub.update_scalar(metric_name, value)
                csv_row[metric_name] = value

        # Write eval row to CSV
        self._write_csv_row(csv_row)

    def after_train(self, runner: "Runner") -> None:
        """Clean up forward hooks and close CSV file."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        # Close CSV file
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()
            logger.info("GeoPillarsWandbHook: Closed CSV metrics file.")

        logger.info("GeoPillarsWandbHook: Removed all forward hooks.")
