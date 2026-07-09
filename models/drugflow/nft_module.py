from argparse import Namespace
from typing import Any, Dict, Optional, Union
import contextlib

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = None
    OmegaConf = None

from src.model.lightning import DrugFlow
from src.utils import dict_to_namespace
from src.data.data_utils import center_data


def _to_ns(x: Optional[Union[Dict, Namespace]]) -> Optional[Namespace]:
    if x is None:
        return None
    if isinstance(x, Namespace):
        return x
    return dict_to_namespace(x)

class SBDDNFTModule(DrugFlow):

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

    def compute_loss_h_or_e(
        self,
        pred_logits,
        pred_logits_old,
        pred_logits_reference,
        z0_m,
        m0,
        zt_m,
        t,
        advantages,
        beta,
        kl_beta,
        adv_clip_max,
        batch_mask,
        name='m',  # 'h' or 'e' for logging
        adv_mode='',
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute NFT loss for discrete variables.

        Args:
            t: timestep tensor of shape [batch_size, 1]
            advantages: advantage values from rewards [batch_size]
            batch_mask: mask for batching [num_nodes/edges], each element indicates which graph/batch
        """

        loss_dict = {}

        # t has shape [batch_size, 1], squeeze to [batch_size] then index by batch_mask
        # batch_mask contains indices like [0, 0, 0, 1, 1, 2, 2, 2] for concatenated graphs
        t_squeezed = t.squeeze(-1)  # [batch_size]
        t_per_item = t_squeezed[batch_mask]  # [num_nodes/edges]
        t_expanded = t_per_item.view(-1, *([1] * (len(m0.shape) - 1)))  # [num_nodes/edges, 1, ...]

        # Expand advantages from [batch_size] to [num_nodes/edges]
        advantages_per_item = advantages[batch_mask]  # [num_nodes/edges]

        # Clip advantages
        advantages_clip = torch.clamp(advantages_per_item, -adv_clip_max, adv_clip_max)

        if adv_mode == 'binary':
            advantages_clip = torch.sign(advantages_clip)

        # Normalize advantage to [0, 1] range for mixing positive/negative losses
        normalized_advantages = (advantages_clip / adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(normalized_advantages, 0, 1)

        # Markov Bridge: work with probability distributions
        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_probs_old = F.softmax(pred_logits_old, dim=-1)
        pred_probs_reference = F.softmax(pred_logits_reference, dim=-1)

        # Mix probabilities
        r_expanded = r.unsqueeze(-1) if r.dim() < pred_probs.dim() else r
        positive_probs = beta * pred_probs + (1 - beta) * pred_probs_old.detach()
        positive_probs = positive_probs / (positive_probs.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize
        implicit_negative_probs = (1 + beta) * pred_probs_old.detach() - beta * pred_probs

        # Ensure valid probabilities
        implicit_negative_probs = torch.clamp(implicit_negative_probs, min=0.0)

        # Normalize, but handle case where sum is 0 (all values were negative)
        prob_sum = implicit_negative_probs.sum(dim=-1, keepdim=True)
        implicit_negative_probs = torch.where(
            prob_sum > 1e-6,
            implicit_negative_probs / prob_sum,
            torch.ones_like(implicit_negative_probs) / implicit_negative_probs.shape[-1]  # uniform fallback
        )

        # Cross-entropy losses
        m0_target = m0.argmax(dim=-1) if m0.dim() > 1 else m0
        positive_loss = F.cross_entropy(
            positive_probs.clamp(min=1e-6).log(),
            m0_target,
            reduction='none'
        )
        negative_loss = F.cross_entropy(
            implicit_negative_probs.clamp(min=1e-6).log(),
            m0_target,
            reduction='none'
        )

        # Mix losses
        ori_policy_loss = r.squeeze() * positive_loss / beta + (1.0 - r.squeeze()) * negative_loss / beta
        policy_loss = (ori_policy_loss * adv_clip_max).mean()

        # KL divergence
        kl_div_loss = F.kl_div(
            pred_probs.log().clamp(min=-100),
            pred_probs_reference.detach(),
            reduction='batchmean'
        )

        total_loss = policy_loss + kl_beta * kl_div_loss

        # Logging
        loss_dict[f'policy_loss_{name}'] = policy_loss.detach()
        loss_dict[f'unweighted_policy_loss_{name}'] = ori_policy_loss.mean().detach() if isinstance(ori_policy_loss, torch.Tensor) else torch.tensor(ori_policy_loss).detach()
        loss_dict[f'kl_div_loss_{name}'] = kl_div_loss.detach()

        return total_loss, loss_dict

    def compute_loss_x(
        self,
        pred_vel,
        pred_vel_old,
        pred_vel_reference,
        z0_x,
        x0,
        zt_x,
        t,
        advantages,
        beta,
        kl_beta,
        adv_clip_max,
        batch_mask,
        adv_mode
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute NFT loss for x (coordinates) following train_nft_sd3.py logic.

        Args:
            pred_vel: predicted velocity from current model
            pred_vel_old: predicted velocity from old (EMA) model
            pred_vel_reference: predicted velocity from reference model
            z0_x: noise sample (starting point)
            x0: ground truth final coordinates
            zt_x: noisy coordinates at timestep t
            t: timestep tensor of shape [batch_size, 1]
            advantages: advantage values from rewards [batch_size]
            beta: beta parameter for positive/negative prediction mixing
            kl_beta: coefficient for KL divergence loss
            adv_clip_max: max clipping value for advantages
            batch_mask: mask for batching [num_nodes], each element indicates which graph/batch
        """
        loss_dict = {}

        # t has shape [batch_size, 1], squeeze to [batch_size] then index by batch_mask
        # batch_mask contains indices like [0, 0, 0, 1, 1, 2, 2, 2] for concatenated graphs
        t_squeezed = t.squeeze(-1)  # [batch_size]
        t_per_node = t_squeezed[batch_mask]  # [num_nodes]
        t_expanded = t_per_node.view(-1, *([1] * (len(x0.shape) - 1)))  # [num_nodes, 1]

        # Expand advantages from [batch_size] to [num_nodes]
        advantages_per_node = advantages[batch_mask]  # [num_nodes]

        # Clip advantages
        advantages_clip = torch.clamp(advantages_per_node, -adv_clip_max, adv_clip_max)

        if adv_mode == 'binary':
            advantages_clip = torch.sign(advantages_clip)

        # Normalize advantage to [0, 1] range for mixing positive/negative losses
        # r=1 means high advantage (good), r=0 means low advantage (bad)
        normalized_advantages = (advantages_clip / adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(normalized_advantages, 0, 1)

        # Compute positive and negative predictions
        positive_prediction = beta * pred_vel + (1 - beta) * pred_vel_old.detach()
        implicit_negative_prediction = (1.0 + beta) * pred_vel_old.detach() - beta * pred_vel

        # Predict x0 from positive prediction
        # DrugFlow CoordICFM: zt = (1-t)*z0 + t*z1, velocity v = (z1-z0)/scale
        # So: z1 = z0 + v*scale, and zt = z0 + t*v*scale
        # Therefore: z1 = zt + (1-t)*v*scale
        scale = self.module_x.scale  # Get scale from CoordICFM module
        x0_prediction_positive = zt_x + (1 - t_expanded) * positive_prediction * scale

        # Adaptive weighting (following NFT)
        with torch.no_grad():
            weight_factor = (
                torch.abs(x0_prediction_positive.double() - x0.double())
                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                .clamp(min=0.00001)
            )

        # Positive loss (per-sample)
        positive_loss = ((x0_prediction_positive - x0) ** 2 / weight_factor).mean(
            dim=tuple(range(1, x0.ndim))
        )

        # Predict x0 from negative prediction
        x0_prediction_negative = zt_x + (1 - t_expanded) * implicit_negative_prediction * scale

        # Adaptive weighting (following NFT)
        with torch.no_grad():
            negative_weight_factor = (
                torch.abs(x0_prediction_negative.double() - x0.double())
                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                .clamp(min=0.00001)
            )

        # Negative loss (per-sample)
        negative_loss = ((x0_prediction_negative - x0) ** 2 / negative_weight_factor).mean(
            dim=tuple(range(1, x0.ndim))
        )

        # Combine positive and negative losses with advantages
        # r close to 1: mostly positive loss, r close to 0: mostly negative loss
        ori_policy_loss = r.squeeze() * positive_loss / beta + (1.0 - r.squeeze()) * negative_loss / beta
        policy_loss = (ori_policy_loss * adv_clip_max).mean()

        # KL divergence with reference model
        kl_div_loss_per_sample = ((pred_vel - pred_vel_reference.detach()) ** 2).mean(
            dim=tuple(range(1, x0.ndim))
        )
        kl_div_loss = kl_div_loss_per_sample.mean()

        # Total loss
        total_loss = policy_loss + kl_beta * kl_div_loss

        # Store loss components for logging
        loss_dict['policy_loss_x'] = policy_loss.detach()
        loss_dict['unweighted_policy_loss_x'] = ori_policy_loss.mean().detach()
        loss_dict['kl_div_loss_x'] = kl_div_loss.detach()
        loss_dict['x0_norm'] = torch.mean(x0**2).detach()
        loss_dict['x0_norm_max'] = torch.max(x0**2).detach()
        loss_dict['old_deviate_x'] = torch.mean((pred_vel - pred_vel_old.detach()) ** 2).detach()
        loss_dict['old_deviate_x_max'] = torch.max((pred_vel - pred_vel_old.detach()) ** 2).detach()
        loss_dict['old_kl_div_x'] = torch.mean(
            ((pred_vel_old.detach() - pred_vel_reference.detach()) ** 2).mean(dim=tuple(range(1, x0.ndim)))
        ).detach()

        return total_loss, loss_dict

    def nft_step(
        self,
        ligand,
        pocket,
        timesteps,
        old_model,
        reference_model,
        advantages,
        beta_x=0.5,
        beta_h=0.5,
        beta_e=0.5,
        kl_beta=0.001,
        adv_clip_max=5.0,
        adv_mode='',
        sc_ligand=None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """NFT step including x, h, and e components"""

        batch_size = len(ligand['size'])

        # Center sample
        ligand, pocket = center_data(ligand, pocket)
        if pocket['x'].numel() > 0:
            pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        else:
            pocket_com = scatter_mean(ligand['x'], ligand['mask'], dim=0)

        # Sample noise
        z0_x = self.module_x.sample_z0(pocket_com, ligand['mask'])
        z0_h = self.module_h.sample_z0(ligand['mask'])
        z0_e = self.module_e.sample_z0(ligand['bond_mask'])

        # Get ground truth final states
        x0 = ligand['x']
        h0 = ligand['one_hot']
        e0 = ligand['bond_one_hot']

        # Sample noisy states at timestep t
        # timesteps comes as [batch_size], but sample_zt expects [batch_size, 1] for proper broadcasting
        t = timesteps.unsqueeze(-1) if timesteps.dim() == 1 else timesteps  # Ensure [batch_size, 1]
        zt_x = self.module_x.sample_zt(z0_x, x0, t, ligand['mask'])
        zt_h = self.module_h.sample_zt(z0_h, h0, t, ligand['mask'])
        zt_e = self.module_e.sample_zt(z0_e, e0, t, ligand['bond_mask'])

        # Get side-chain transform if used
        sc_transform = self.get_sc_transform_fn(None, zt_x, timesteps, None, ligand['mask'], pocket)

        # Predict from current model
        self.dynamics.prev_ligand = sc_ligand; self.dynamics.prev_residues = None # use prediction from sampling step for self-conditioning
        pred_ligand, _, _ = self.dynamics(
            zt_x, zt_h, ligand['mask'], pocket, timesteps,
            bonds_ligand=(ligand['bonds'], zt_e),
            sc_transform=sc_transform,
            ignore_nan=True
        )
        pred_vel = pred_ligand['vel']
        pred_h = pred_ligand['logits_h']
        pred_e = pred_ligand['logits_e']

        # Predict from old model (no gradients)
        with torch.no_grad():
            old_model.dynamics.prev_ligand = sc_ligand; old_model.dynamics.prev_residues = None # use prediction from sampling step for self-conditioning
            pred_ligand_old, _, _ = old_model.dynamics(
                zt_x, zt_h, ligand['mask'], pocket, timesteps,
                bonds_ligand=(ligand['bonds'], zt_e),
                sc_transform=sc_transform,
                ignore_nan=True
            )
            pred_vel_old = pred_ligand_old['vel'].detach()
            pred_h_old = pred_ligand_old['logits_h'].detach()
            pred_e_old = pred_ligand_old['logits_e'].detach()

        # Predict from reference model (no gradients)
        with torch.no_grad():
            if reference_model is not None:
                #reference_model.dynamics.prev_ligand = sc_ligand; reference_model.dynamics.prev_residues = None
                pred_ligand_reference, _, _ = reference_model.dynamics(
                    zt_x, zt_h, ligand['mask'], pocket, timesteps,
                    bonds_ligand=(ligand['bonds'], zt_e),
                    sc_transform=sc_transform,
                    ignore_nan=True
                )
                pred_vel_reference = pred_ligand_reference['vel'].detach()
                pred_h_reference = pred_ligand_reference['logits_h'].detach()
                pred_e_reference = pred_ligand_reference['logits_e'].detach()
            else:
                # Use old model as reference
                pred_vel_reference = pred_vel_old.detach()
                pred_h_reference = pred_h_old.detach()
                pred_e_reference = pred_e_old.detach()

        # Compute NFT loss for x component
        loss_x, loss_dict_x = self.compute_loss_x(
            pred_vel=pred_vel,
            pred_vel_old=pred_vel_old,
            pred_vel_reference=pred_vel_reference,
            z0_x=z0_x,
            x0=x0,
            zt_x=zt_x,
            t=t,
            advantages=advantages,
            beta=beta_x,
            kl_beta=kl_beta,
            adv_clip_max=adv_clip_max,
            batch_mask=ligand['mask'],
            adv_mode=adv_mode
        )

        total_loss = loss_x
        loss_dict = {"total_loss_x": float(loss_x.detach().cpu())}

        # Add loss dict for x
        for k, v in loss_dict_x.items():
            if torch.is_tensor(v) and v.numel() == 1:
                loss_dict[k] = float(v.detach().cpu())
            elif isinstance(v, (int, float)):
                loss_dict[k] = float(v)

        # Compute NFT loss for h (atom types)
        loss_h, loss_dict_h = self.compute_loss_h_or_e(
            pred_logits=pred_h,
            pred_logits_old=pred_h_old,
            pred_logits_reference=pred_h_reference,
            z0_m=z0_h,
            m0=h0,
            zt_m=zt_h,
            t=t,
            advantages=advantages,
            beta=beta_h,
            kl_beta=kl_beta,
            adv_clip_max=adv_clip_max,
            batch_mask=ligand['mask'],
            name='h',
            adv_mode=adv_mode
        )
        total_loss += loss_h
        loss_dict["total_loss_h"] = float(loss_h.detach().cpu())

        for k, v in loss_dict_h.items():
            if torch.is_tensor(v) and v.numel() == 1:
                loss_dict[k] = float(v.detach().cpu())
            elif isinstance(v, (int, float)):
                loss_dict[k] = float(v)

        loss_e, loss_dict_e = self.compute_loss_h_or_e(
            pred_logits=pred_e,
            pred_logits_old=pred_e_old,
            pred_logits_reference=pred_e_reference,
            z0_m=z0_e,
            m0=e0,
            zt_m=zt_e,
            t=t,
            advantages=advantages,
            beta=beta_e,
            kl_beta=kl_beta,
            adv_clip_max=adv_clip_max,
            batch_mask=ligand['bond_mask'],
            name='e',
            adv_mode=adv_mode
        )
        total_loss += loss_e
        loss_dict["total_loss_e"] = float(loss_e.detach().cpu())

        for k, v in loss_dict_e.items():
            if torch.is_tensor(v) and v.numel() == 1:
                loss_dict[k] = float(v.detach().cpu())
            elif isinstance(v, (int, float)):
                loss_dict[k] = float(v)

        # Update total loss
        loss_dict["total_loss"] = float(total_loss.detach().cpu())

        return total_loss, loss_dict
