import os
import sys
import logging
from typing import Any
import wandb


class SeverityLevelBetween(logging.Filter):
    def __init__(self, min_level: int, max_level: int) -> None:
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record) -> bool:
        return self.min_level <= record.levelno < self.max_level


def setup_logging() -> None:
    root = logging.getLogger()
    # Perform setup only if logging has not been configured
    target_logging_level = getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper())
    root.setLevel(target_logging_level)
    if not root.hasHandlers():
        log_formatter = logging.Formatter(
            "%(asctime)s (%(levelname)s): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Send INFO (or target) to stdout
        handler_out = logging.StreamHandler(sys.stdout)
        handler_out.addFilter(
            SeverityLevelBetween(target_logging_level, logging.WARNING)
        )
        handler_out.setFormatter(log_formatter)
        root.addHandler(handler_out)

        # Send WARNING (and higher) to stderr
        handler_err = logging.StreamHandler(sys.stderr)
        handler_err.setLevel(logging.WARNING)
        handler_err.setFormatter(log_formatter)
        root.addHandler(handler_err)


class Logger():
    """Generic class to interface with various logging modules, e.g. wandb,
    tensorboard, etc.
    """

    def __init__(self, config=None) -> None:
        self.config = config

    # @abstractmethod
    # def watch(self, model, log_freq: int = 1000):
    #     """
    #     Monitor parameters and gradients.
    #     """

    def log(self, update_dict, split: str = ""):
        """
        Log some values.
        """
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict[f"{split}/{key}"] = update_dict[key]
            update_dict = new_dict
        return update_dict

    # @abstractmethod
    # def log_plots(self, plots) -> None:
    #     pass

    # @abstractmethod
    # def mark_preempting(self) -> None:
    #     pass

    # @abstractmethod
    # def log_summary(self, summary_dict: dict[str, Any]) -> None:
    #     pass

    # @abstractmethod
    # def log_artifact(self, name: str, type: str, file_location: str) -> None:
    #     pass


class WandBLogger(Logger):
    def __init__(
        self,
        name,
        project,
        entity=None,
        group=None,
        mode='online',
        savedir=None,
        config=None,
        **kwargs,
    ) -> None:
        super().__init__(config)

        # wandb writes its run files to `savedir`/wandb (defaults to the cwd); point
        # it at the run's save_dir so artifacts live alongside the rest of the outputs.
        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)

        wandb.init(
            name=name,
            project=project,
            entity=entity,
            group=group,
            mode=mode,
            dir=savedir,
            config=config,
            **kwargs,
        )

    def watch(self, model, log="all", log_freq: int = 1000) -> None:
        wandb.watch(model, log=log, log_freq=log_freq)

    def log(self, update_dict, split: str = "") -> None:
        update_dict = super().log(update_dict, split)
        wandb.log(update_dict)

    def log_plots(self, plots, caption: str = "") -> None:
        assert isinstance(plots, list)
        plots = [wandb.Image(x, caption=caption) for x in plots]
        wandb.log({"data": plots})

    def log_table(
        self, name: str, cols: list, data: list, commit=False
    ) -> None:
        # cols are 1D list of N elements, data must be NxK where the number of cols must match cols
        # see https://docs.wandb.ai/guides/tables
        table = wandb.Table(columns=cols, data=data)
        wandb.log({name: table}, commit=commit)

    def log_summary(self, summary_dict: dict[str, Any]):
        for k, v in summary_dict.items():
            wandb.run.summary[k] = v

    def mark_preempting(self) -> None:
        wandb.mark_preempting()

    def log_artifact(self, name: str, type: str, file_location: str) -> None:
        art = wandb.Artifact(name=name, type=type)
        if os.path.isdir(file_location):
            art.add_dir(file_location)
        else:
            art.add_file(file_location)
        art.save()
