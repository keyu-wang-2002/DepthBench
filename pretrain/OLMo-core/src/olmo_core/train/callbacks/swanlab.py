import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class SwanLabCallback(Callback):
    """
    Logs metrics to SwanLab from rank 0.

    .. note::
        Requires the ``swanlab`` package. Authentication can be provided via
        ``swanlab login`` or the ``SWANLAB_API_KEY`` environment variable.

    .. note::
        This callback logs metrics from every single step to SwanLab, regardless of the value
        of :data:`Trainer.metrics_collect_interval <olmo_core.train.Trainer.metrics_collect_interval>`.
    """

    enabled: bool = True
    """
    Set to false to disable this callback.
    """

    project: Optional[str] = None
    """
    The SwanLab project to use.
    """

    workspace: Optional[str] = None
    """
    The SwanLab workspace to use.
    """

    experiment_name: Optional[str] = None
    """
    The name to give the SwanLab run.
    """

    description: Optional[str] = None
    """
    A note/description of the run.
    """

    group: Optional[str] = None
    """
    The SwanLab group to use.
    """

    tags: Optional[List[str]] = None
    """
    Tags to assign the run.
    """

    mode: Optional[str] = None
    """
    SwanLab mode. Common values are ``cloud``, ``local``, or ``disabled``.
    """

    config: Optional[Dict[str, Any]] = None
    """
    The config to load to SwanLab.
    """

    logdir_name: str = "swanlab"
    """
    Relative log directory under the trainer work directory.
    """

    _swanlab = None
    _run = None
    _finalized: bool = False

    @property
    def swanlab(self):
        if self._swanlab is None:
            import swanlab  # type: ignore

            self._swanlab = swanlab
        return self._swanlab

    @property
    def run(self):
        return self._run

    @property
    def finalized(self) -> bool:
        return self._finalized

    def finalize(self):
        if not self.finalized:
            log.info("Finalizing SwanLab run...")
            self.swanlab.finish()
            self._finalized = True

    def pre_train(self):
        if self.enabled and get_rank() == 0:
            logdir = self.trainer.work_dir / self.logdir_name
            logdir.mkdir(parents=True, exist_ok=True)
            self._run = self.swanlab.init(
                project=self.project or None,
                workspace=self.workspace or None,
                experiment_name=self.experiment_name or None,
                description=self.description or None,
                group=self.group or None,
                tags=self.tags or None,
                config=self.config,
                logdir=str(logdir),
                mode=self.mode or None,
            )

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if self.enabled and get_rank() == 0:
            self.swanlab.log(metrics, step=step)

    def on_error(self, exc: BaseException):
        del exc
        if self.enabled and get_rank() == 0 and self.run is not None:
            self.finalize()

    def close(self):
        if self.enabled and get_rank() == 0 and self.run is not None:
            self.finalize()
