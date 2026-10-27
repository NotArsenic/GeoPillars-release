"""
Geo-Pillars Inference Profiling Hook
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
from mmengine.hooks import Hook
from mmengine.logging import MMLogger

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class InferenceProfilerHook(Hook):
    """Profiles single-frame inference latency and logs model size.

    Args:
        warmup_iters: Number of initial iterations to skip for latency
            measurement (GPU warmup, JIT compilation, etc.).
    """

    priority = "VERY_LOW"  # run after everything else

    def __init__(self, warmup_iters: int = 10) -> None:
        super().__init__()
        self.warmup_iters = warmup_iters
        self._latencies: list[float] = []
        self._iter_count: int = 0
        self._t0: float = 0.0

    # Model summary at start of testing

    def before_test(self, runner) -> None:
        """Log parameter counts before testing begins."""
        logger = MMLogger.get_current_instance()
        model = runner.model

        # Handle DDP / DataParallel wrappers
        if hasattr(model, "module"):
            model = model.module

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        sep = "=" * 64
        logger.info(f"\n{sep}")
        logger.info("  Geo-Pillars Model Summary")
        logger.info(sep)
        logger.info(f"  Total parameters     : {total_params:>12,d}")
        logger.info(f"  Trainable parameters : {trainable_params:>12,d}")
        logger.info(f"  Frozen parameters    : {frozen_params:>12,d}")
        logger.info(f"  Model size (fp32)    : {total_params * 4 / 1e6:>10.2f} MB")
        logger.info(sep)

        self._latencies = []
        self._iter_count = 0

    # Per-iteration timing
    def before_test_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
    ) -> None:
        # Sync GPU before timing for accurate measurement
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        self._t0 = time.perf_counter()

    def after_test_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[Sequence[dict]] = None,
    ) -> None:
        # Sync GPU after inference completes
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self._iter_count += 1

        if self._iter_count > self.warmup_iters:
            self._latencies.append(elapsed_ms)

    # Summary after testing
    def after_test(self, runner) -> None:
        """Log latency statistics after testing completes."""
        logger = MMLogger.get_current_instance()

        if not self._latencies:
            logger.warning(
                "[InferenceProfilerHook] No latency samples collected "
                f"(total iters={self._iter_count}, "
                f"warmup={self.warmup_iters})."
            )
            return

        lat = np.array(self._latencies)
        sep = "=" * 64

        logger.info(f"\n{sep}")
        logger.info("  Geo-Pillars Inference Latency Report")
        logger.info(sep)
        logger.info(
            f"  Frames measured  : {len(lat):>10,d}  "
            f"(warmup skipped: {self.warmup_iters})"
        )
        logger.info(f"  Mean latency     : {lat.mean():>10.2f} ms")
        logger.info(f"  Std dev          : {lat.std():>10.2f} ms")
        logger.info(f"  Median (p50)     : {np.percentile(lat, 50):>10.2f} ms")
        logger.info(f"  p95              : {np.percentile(lat, 95):>10.2f} ms")
        logger.info(f"  p99              : {np.percentile(lat, 99):>10.2f} ms")
        logger.info(f"  Min              : {lat.min():>10.2f} ms")
        logger.info(f"  Max              : {lat.max():>10.2f} ms")
        fps = 1000.0 / lat.mean() if lat.mean() > 0 else 0.0
        logger.info(f"  Throughput       : {fps:>10.1f} FPS")
        logger.info(sep)
