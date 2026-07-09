import os
import logging
from typing import List, Optional, Dict
try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = None
    OmegaConf = None
import torch
from pathlib import Path
import time

from pipeline.utils.logger import Logger
from models.suite.base import ModelSuite
#from memory.replay_buffer import ReplayBuffer
from memory.ltm import LongTimeMem
from memory.diversity_filter import DiversityFilter
from rewards.reward import Reward
import numpy as np
from rdkit import Chem

def get_device(device: Optional[str]):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    return torch.device(device)


class ReinL:
    def __init__(
        self,
        rl_epoch: int,
        model_suite: ModelSuite,
        reward: Reward,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        save_dir: str,
        save_freq: int,
        device: str = None,
        logger: Logger = None,
        replay: bool = False,
        replay_args: Dict = None,
        diversity_filter: DiversityFilter = None,
        **kwargs,
    ) -> None:
        self.rl_epoch = rl_epoch
        self.model_suite = model_suite
        self.reward = reward
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.logger = logger
        self.device = get_device(device)
        self.cfg = OmegaConf.create(kwargs)
        self.step = 0
        self.cost = 0
        self.ft_global_step = 0

        self.sample_cfg = OmegaConf.merge(
            model_suite.sample_cfg, sample_cfg
        )

        self.finetune_cfg = OmegaConf.merge(
            model_suite.finetune_cfg, finetune_cfg
        )

        self.sampler = model_suite.get_sampler(self.sample_cfg, logger=self.logger)

        # long-term memory
        self.ltm = LongTimeMem()

        # diversity filter (REINVENT-style scaffold penalty); NoFilter if unset
        self.div_filter = diversity_filter if diversity_filter is not None else DiversityFilter()
        logging.info(f"Diversity filter: {self.div_filter.name}")

        self.models_dir = os.path.join(save_dir, 'models')
        self.sample_dir = os.path.join(save_dir, 'samples')
        # self.reward_dir = os.path.join(save_dir, 'rewards')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        # if not os.path.exists(self.reward_dir):
        #     os.makedirs(self.reward_dir)

        if replay:
            #self.replay = ReplayBuffer(**replay_args) #TODO: adapt replay buffer to the sbdd pipeline
            self.replay = None
        else:
            self.replay = None

    def init_optimizer(self, lr=5e-4, optimizer=torch.optim.Adam):
        self.optimizer = optimizer(self.agent.parameters(), lr=lr)

    def init_scheduler(self, start_factor=0.1, total_iters=10):
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=start_factor,
            total_iters=total_iters,
            verbose=False,
        )

    def freeze_model(self, freeze):
        n_freeze = freeze * 4 + 1
        for i, param in enumerate(self.agent.network.parameters()):
            if i < n_freeze:  # Freeze parameter
                param.requires_grad = False

    def reward_step(
        self,
        sample_data: dict,
        label: str = 'tmp'
    ):
        rewards, prop_dict, failed_mask = self.reward.scoring(sample_data, label)
        self.cost += len(sample_data['ligands']['mols'])

        # Keep all samples, but failed ones have 0 reward (set in Reward.scoring)
        success_rewards = rewards.astype(float)
        success_prop_dict = prop_dict
        success_data = sample_data

        # Robust SMILES generation
        smiles_list = []
        for mol in success_data['ligands']['mols']:
            try:
                if mol is not None:
                    s = Chem.MolToSmiles(mol)
                else:
                    s = ""
                smiles_list.append(s)
            except Exception:
                smiles_list.append("")

        smiles = np.array(smiles_list)

        logging.info(f'Evaluation costs to date: {self.cost}')
        logging.info(f'Number of samples: {len(success_data["ligands"]["mols"])}')

        r_mean = float(np.nanmean(success_rewards)) if len(success_rewards) else float('nan')
        r_std  = float(np.nanstd(success_rewards)) if len(success_rewards) else float('nan')
        log_payload = {'reward mean': r_mean, 'reward std': r_std}
        logging.info(f'reward mean={r_mean:.4f} std={r_std:.4f}')

        if self.logger is not None:
            self.logger.log(log_payload)

        return success_data, smiles, success_rewards, success_prop_dict, failed_mask

    def load_model(self):
        raise NotImplementedError

    def sample_step(self):
        raise NotImplementedError

    def ft_step(self, data_list):
        raise NotImplementedError

    def rl_step(self):
        raise NotImplementedError

    def run_rl(self):
        raise NotImplementedError
