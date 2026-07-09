from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict, Optional, Union
import contextlib

import torch
try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = None
    OmegaConf = None

from src.model.lightning import DrugFlow
from src.utils import dict_to_namespace


def _to_ns(x: Optional[Union[Dict, Namespace]]) -> Optional[Namespace]:
    if x is None:
        return None
    if isinstance(x, Namespace):
        return x
    return dict_to_namespace(x)

class SBDDModule(DrugFlow):
    """
    Thin wrapper over DrugFlow that:
    - accepts a DictConfig/dict (from saved checkpoint hyper_parameters)
    - converts nested dicts to argparse.Namespace as DrugFlow expects
    - exposes ft_step(batch) for our finetuning loops
    - stores cfg for saving (hparams.yaml) compatibility
    """

    def __new__(cls, args, model_params: Dict[str, Any]):
        # --- resolve inputs ---
        device = args["device"]
        chkpt_path = args.get("checkpoint", None) or model_params.get("chkpt_path")
        if not chkpt_path:
            raise ValueError("Provide chkpt_path via args.chkpt_path or model_params['chkpt_path'].")

        params = dict(model_params)
        params.pop("chkpt_path", None)

        # --- create the instance from checkpoint ---
        # If load_from_checkpoint is @classmethod, this returns an instance of `cls` (ideal).
        # Otherwise, fall back to the base class method.
        try:
            obj = cls.load_from_checkpoint(chkpt_path, map_location=device, **params)
        except TypeError:
            obj = DrugFlow.load_from_checkpoint(chkpt_path, map_location=device, **params)

        # If the loader ignored `cls` and returned a pure DrugFlow, upgrade its class
        # so your overrides in Wrapper still apply (only if compatible).
        if not isinstance(obj, cls):
            obj.__class__ = cls

        obj.model_params = model_params

        return obj

    def __init__(self, cfg: Union[DictConfig, Dict[str, Any]], model_params: Dict[str, Any]):
        """
        self.cfg = cfg
        self.model_params = model_params

        cfg_dict: Dict[str, Any]
        if isinstance(cfg, DictConfig):
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # nested dicts
        else:
            cfg_dict = cfg

        # Convert nested dicts to Namespace objects as DrugFlow expects
        train_params = _to_ns(model_params.get('train_params'))
        simulation_params = _to_ns(model_params.get('simulation_params'))
        dataset_params = _to_ns(model_params.get('dataset_params'))
        loss_params = _to_ns(model_params.get('loss_params'))
        eval_params = _to_ns(model_params.get('eval_params'))
        predictor_params = _to_ns(model_params.get('predictor_params'))

        # Extract other required parameters
        pocket_representation = model_params.get('pocket_representation', 'side_chain_bead')
        virtual_nodes = model_params.get('virtual_nodes')
        flexible = model_params.get('flexible', False)
        flexible_bb = model_params.get('flexible_bb', False)
        debug = model_params.get('debug', False)
        overfit = model_params.get('overfit', False)

        print("================================================")
        print("Model params: ", model_params)
        print("================================================")
        # Initialize parent DrugFlow class with the expected parameters
        super().__init__(
            pocket_representation=pocket_representation,
            train_params=train_params,
            dataset_params=dataset_params,
            loss_params=loss_params,
            eval_params=eval_params,
            predictor_params=predictor_params,
            simulation_params=simulation_params,
            virtual_nodes=virtual_nodes,
            flexible=flexible,
            flexible_bb=flexible_bb,
            debug=debug,
            overfit=overfit
        )"""
        pass

    def ft_step(self, batch) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Finetuning step compatible with our RL pipelines.
        Expects a batch dict with 'ligand' and 'pocket' entries (from ProcessedLigandPocketDataset.collate_fn).
        Returns (loss, loss_dict).
        """
        ligand = batch["ligand"]
        pocket = batch["pocket"]

        loss, info = self.compute_loss(ligand, pocket, return_info=True)

        # Flatten info into scalars when possible
        loss_dict: Dict[str, float] = {"loss": loss if isinstance(loss, float) else float(loss.detach().cpu())}
        for k, v in info.items():
            if isinstance(v, (int, float)):
                loss_dict[k] = float(v)
            elif torch.is_tensor(v) and v.numel() == 1:
                loss_dict[k] = float(v.detach().cpu())
        return loss, loss_dict