import os
import logging
from typing import List, Literal, Optional
from pathlib import Path
import sys
from argparse import Namespace

import pytorch_lightning as pl
from pipeline.utils.logger import Logger
try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = None
    OmegaConf = None
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader

from models.suite.base import ModelSuite
from src.data.dataset import ProcessedLigandPocketDataset #, TrajectoryDataset # TODO: add TrajectoryDataset
from models.drugflow.pl_module import SBDDModule
from models.drugflow.nft_module import SBDDNFTModule
from models.drugflow.sampler import SBDDSampler, SBDDNFTSampler
from src.data.data_utils import prepare_ligand #, prepare_ligand_from_tensors
from src.constants import atom_encoder, bond_encoder

AVA_MODEL_NAME = Literal[
    "drugflow",
]


def _ns_to_dict(x):
    """Convert Namespace/dict/iterables to plain Python containers for serialization."""
    if isinstance(x, Namespace):
        return {k: _ns_to_dict(getattr(x, k)) for k in vars(x)}
    if isinstance(x, dict):
        return {k: _ns_to_dict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_ns_to_dict(v) for v in x]
    return x


def get_model_params(checkpoint_path, args):
    model_params = torch.load(checkpoint_path, map_location=args["device"])['hyper_parameters']

    # Update parameters
    if args["datadir"] is not None:
        model_params['train_params'].datadir = args["datadir"]

    if args["n_steps"] is not None:
        model_params['simulation_params'].n_steps = args["n_steps"]

    if args["sampler"] is not None:
        model_params['simulation_params'].sampler = args["sampler"]

    if args["noise_scale"] is not None:
        model_params['simulation_params'].noise_scale_x = args["noise_scale"]
        logging.info(f"Using SDE sampling with noise scale: {args['noise_scale']}")
    else:
        logging.info("Using ODE sampling")

    return model_params

def preprocess_samples(samples, atom_encoder, bond_encoder):
    """
    Preprocesses generated samples for training by preparing ligands and organizing data.

    Args:
        samples (dict): Dictionary containing 'ligands' and 'pockets' data
                       - samples['ligands']['mols']: List of molecular objects
                       - samples['pockets']: List of pocket data dictionaries
        atom_encoder: Encoder for atom types
        bond_encoder: Encoder for bond types

    Returns:
        dict: Preprocessed generations dictionary with 'ligands' and 'pockets' keys,
              each containing lists of processed data ready for training
    """
    data = {
        'ligands': {},
        'pockets': {}
    }

    count = 0
    for i, mol in enumerate(samples["ligands"]["generated_mols"]):
        try:
            ligand = prepare_ligand(mol, atom_encoder, bond_encoder)

            # Initialize the structure on first successful ligand
            if count == 0:
                for key in ligand.keys():
                    data['ligands'][key] = []

                for key in samples['pockets'].keys():
                    data['pockets'][key] = []

            # Process and append ligand data
            for key, value in ligand.items():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    value = value.detach().cpu()
                data['ligands'][key].append(value)

            # Add name to ligand data
            if 'name' not in data['ligands']:
                data['ligands']['name'] = []
            data['ligands']['name'].append(f"generated_mol_{i}")

            # Process and append corresponding pocket data
            for key in samples['pockets'].keys():
                value = samples['pockets'][key][i]
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    value = value.detach().cpu()
                data['pockets'][key].append(value)

            count += 1

        except Exception as e:
            print(f"Error preparing ligand {i}: {e}")
            continue

    return data

'''
def preprocess_samplesv2(samples, atom_encoder, bond_encoder):
    """
    Preprocesses generated samples for training by preparing ligands and organizing data.

    Args:
        samples (dict): Dictionary containing 'ligands' and 'pockets' data
                       - samples['ligands']['mols']: List of molecular objects
                       - samples['pockets']: List of pocket data dictionaries
        atom_encoder: Encoder for atom types
        bond_encoder: Encoder for bond types

    Returns:
        dict: Preprocessed generations dictionary with 'ligands' and 'pockets' keys,
              each containing lists of processed data ready for training
    """
    data = {
        'ligands': {},
        'pockets': {}
    }

    count = 0
    for i, mol in enumerate(samples["ligands"]["mols"]):
        try:
            ligand = prepare_ligand_from_tensors(coords=samples["ligands"]["x"][i], atom_types=samples["ligands"]["one_hot"][i], bonds=samples["ligands"]["bonds"][i], bond_types=samples["ligands"]["bond_one_hot"][i], atom_encoder=atom_encoder, bond_encoder=bond_encoder)

            # Initialize the structure on first successful ligand
            if count == 0:
                for key in ligand.keys():
                    data['ligands'][key] = []

                for key in samples['pockets'].keys():
                    data['pockets'][key] = []

            # Process and append ligand data
            for key, value in ligand.items():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    value = value.detach().cpu()
                data['ligands'][key].append(value)

            # Process and append corresponding pocket data
            for key in samples['pockets'].keys():
                value = samples['pockets'][key][i]
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    value = value.detach().cpu()
                data['pockets'][key].append(value)

            count += 1

        except Exception as e:
            print(f"Error preparing ligand {i}: {e}")
            continue

    return data
'''

class SBDDSuite(ModelSuite):
    def __init__(
        self,
        model_name: AVA_MODEL_NAME,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        model_path: Optional[int] = None,
        config_overrides: list[str] = [],
        device: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            model_path=model_path,
            config_overrides=config_overrides,
            device=device,
            **kwargs,
        )

    def load_model(self):
        assert self.model_path is not None, "Model path is not provided"
        ckpt_path = Path(self.model_path)

        cfg = OmegaConf.create(self.finetune_cfg)
        cfg["device"] = str(self.device)
        model_params = get_model_params(ckpt_path, cfg)
        model = SBDDNFTModule(cfg, model_params)
        return model

    def get_sampler(self, sampler_cfg: DictConfig, logger: Optional[Logger] = None):
        logging.info(f"Using {sampler_cfg.sampler_type} sampler")
        if sampler_cfg.sampler_type == "drugflow":
            sampler = SBDDSampler(**sampler_cfg, logger=logger)
        elif sampler_cfg.sampler_type == "nft":
            sampler = SBDDNFTSampler(**sampler_cfg, logger=logger)
        else:
            raise ValueError(f"Sampler {sampler_cfg.sampler_type} not found")
        return sampler

    def get_grpo_sampler(self, sampler_cfg: DictConfig, logger: Optional[Logger] = None):
        sampler = SBDDGRPOSampler(**sampler_cfg, logger=logger)
        return sampler

    def get_dataloader(
        self,
        samples: dict,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
    ):
        if batch_size is None:
            batch_size = self.finetune_cfg.batch_size
        data = preprocess_samples(samples, atom_encoder, bond_encoder)
        dataset = ProcessedLigandPocketDataset(data)
        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle,
                                sampler=None,
                                num_workers=0,
                                collate_fn=ProcessedLigandPocketDataset.collate_fn,
                                pin_memory=True)
        return dataloader

    # TODO: add TrajectoryDataset
    '''
    def get_trajectory_dataloader(self, samples: dict, batch_size: Optional[int] = None, shuffle: bool = False, num_workers: int = 0):
        if batch_size is None:
            batch_size = self.finetune_cfg.batch_size
        dataset = TrajectoryDataset(samples)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=lambda batch: batch,  # keep as list; we build per-timestep concatenations
            num_workers=num_workers,
        )
        return loader
    '''
    def save_model_params(self, save_dir: str, model: SBDDModule):
        os.makedirs(save_dir, exist_ok=True)

        mp = getattr(model, "model_params", None)
        if mp is not None:
            mp_yaml = OmegaConf.create(_ns_to_dict(mp))
            OmegaConf.save(mp_yaml, os.path.join(save_dir, "model_parameters.yaml"))

    def save_model(
        self,
        model: SBDDModule,
        save_dir: str,
    ):
        os.makedirs(save_dir, exist_ok=True)

        logging.info(f"Saving model to {save_dir}")
        mp = getattr(model, "model_params", {})
        ckpt_dict = {
            "state_dict": model.state_dict(),
            "hyper_parameters": _ns_to_dict(mp),
            "pytorch-lightning_version": pl.__version__,
            "epoch": int(getattr(model, "current_epoch", 0)),
            "global_step": int(getattr(model, "global_step", 0)),
            "callbacks": {},
            "optimizer_states": [],
            "lr_schedulers": [],
        }
        torch.save(ckpt_dict, os.path.join(save_dir, f"model.ckpt"))
