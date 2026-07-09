import os
try:
    import hydra
except ImportError:
    hydra = None
from typing import List, Literal, Optional, Dict, Any
try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = None
    OmegaConf = None
import torch


AVA_MODEL_NAME = Literal[
    "drugflow",
]


def get_device(device: Optional[str] = None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    return torch.device(device)


class ModelSuite:
    def __init__(
        self,
        model_name: str,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        model_path: Optional[int] = None,
        config_overrides: list[str] = [],
        device: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.sample_cfg = sample_cfg
        self.finetune_cfg = finetune_cfg
        self.model_path = model_path
        self.config_overrides = config_overrides
        self.device = get_device(device)
        self.cfg = OmegaConf.create(kwargs)

    def load_model(self):
        raise NotImplementedError

    def get_sampler(self):
        raise NotImplementedError

    def get_dataloader(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError
