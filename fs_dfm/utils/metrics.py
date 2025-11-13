#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


"""
Flexible metrics logging system that supports multiple backends.
Allows easy switching between wandb, tensorboard metrics tracking.
"""

import os
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to the backend."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up and close the backend."""
        pass


class WandBBackend(MetricsBackend):
    """Weights & Biases metrics backend."""

    def __init__(
        self,
        project: str = "odllm",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        try:
            import wandb

            self.wandb = wandb
            self.run = wandb.init(project=project, name=name, config=config)
            self.available = True
        except ImportError:
            self.wandb = None
            self.available = False
            logger.warning("wandb not installed. Install with: pip install wandb")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.available and self.wandb:
            self.wandb.log(metrics, step=step)

    def close(self) -> None:
        if self.available and self.wandb:
            self.wandb.finish()


class TensorBoardBackend(MetricsBackend):
    """TensorBoard metrics backend."""

    def __init__(self, log_dir: str = "./runs"):
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
            self.available = True
        except ImportError:
            self.writer = None
            self.available = False
            logger.warning(
                "tensorboard not available. Install with: pip install tensorboard"
            )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.available and self.writer:
            for key, value in metrics.items():
                try:
                    # TensorBoard expects scalar values
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(key, value, global_step=step)
                except Exception as e:
                    logger.debug(f"Could not log {key} to tensorboard: {e}")

    def close(self) -> None:
        if self.available and self.writer:
            self.writer.close()


class ConsoleBackend(MetricsBackend):
    """Simple console logging backend for debugging."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.verbose:
            step_str = f"[Step {step}] " if step is not None else ""
            for key, value in metrics.items():
                logger.info(f"{step_str}{key}: {value}")

    def close(self) -> None:
        pass


class NoOpBackend(MetricsBackend):
    """No-operation backend that discards all metrics."""

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def close(self) -> None:
        pass


class MetricsLogger:
    """
    Unified metrics logger that can use multiple backends.

    Usage:
        # Initialize with desired backend
        metrics = MetricsLogger(backend="wandb", project="my_project")

        # Or use auto mode to detect available backends
        metrics = MetricsLogger(backend="auto")

        # Log metrics
        metrics.log({"loss": 0.5, "accuracy": 0.95}, step=100)

        # Clean up
        metrics.close()
    """

    def __init__(
        self,
        backend: str = "auto",
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the metrics logger.

        Args:
            backend: One of "auto", "wandb", "tensorboard", "console", "noop"
            project: Project name for wandb
            name: Run name for wandb
            config: Configuration dict for wandb
            log_dir: Directory for tensorboard logs
            verbose: Whether to use verbose console logging
        """
        self.backends = []

        if backend == "auto":
            if not self.backends:
                wandb_backend = WandBBackend(
                    project=project or "odllm", name=name, config=config
                )
                if wandb_backend.available:
                    self.backends.append(wandb_backend)
                    logger.info("Using WandB backend for metrics")

            # Then tensorboard
            if not self.backends:
                tb_backend = TensorBoardBackend(log_dir=log_dir or "./runs")
                if tb_backend.available:
                    self.backends.append(tb_backend)
                    logger.info("Using TensorBoard backend for metrics")

            # Fallback to console
            if not self.backends:
                self.backends.append(ConsoleBackend(verbose=True))
                logger.info("Using console backend for metrics")

        elif backend == "wandb":
            self.backends.append(
                WandBBackend(project=project or "odllm", name=name, config=config)
            )
        elif backend == "tensorboard":
            self.backends.append(TensorBoardBackend(log_dir=log_dir or "./runs"))
        elif backend == "console":
            self.backends.append(ConsoleBackend(verbose=verbose))
        elif backend == "noop":
            self.backends.append(NoOpBackend())
        elif backend == "multi":
            wandb_backend = WandBBackend(
                project=project or "odllm", name=name, config=config
            )
            if wandb_backend.available:
                self.backends.append(wandb_backend)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to all configured backends."""
        for backend in self.backends:
            try:
                backend.log_metrics(metrics, step=step)
            except Exception as e:
                logger.debug(
                    f"Failed to log metrics to {backend.__class__.__name__}: {e}"
                )

    def log_single(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Convenience method to log a single metric."""
        self.log({name: value}, step=step)

    def close(self) -> None:
        """Clean up and close all backends."""
        for backend in self.backends:
            try:
                backend.close()
            except Exception as e:
                logger.debug(f"Failed to close {backend.__class__.__name__}: {e}")


# Global instance for easy access
_global_metrics: Optional[MetricsLogger] = None


def init_metrics(
    backend: str = "auto",
    project: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict] = None,
    log_dir: Optional[str] = None,
    verbose: bool = False,
) -> MetricsLogger:
    """Initialize the global metrics logger."""
    global _global_metrics
    _global_metrics = MetricsLogger(
        backend=backend,
        project=project,
        name=name,
        config=config,
        log_dir=log_dir,
        verbose=verbose,
    )
    return _global_metrics


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics using the global logger."""
    if _global_metrics is not None:
        _global_metrics.log(metrics, step=step)


def close_metrics() -> None:
    """Close the global metrics logger."""
    global _global_metrics
    if _global_metrics is not None:
        _global_metrics.close()
        _global_metrics = None
