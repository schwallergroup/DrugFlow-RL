import os
import time
import copy
import logging
from collections import defaultdict
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from pipeline.base import ReinL
from pipeline.utils.save import save_smiles
from pipeline.utils.logger import Logger
from pipeline.utils.nft_utils import PerPocketStatTracker, return_decay
from models.suite.base import ModelSuite
from rewards.reward import Reward
from src.data.data_utils import TensorDict, Residues
from src.constants import atom_encoder, bond_encoder

torch.autograd.set_detect_anomaly(False)

class DiffusionNFT(ReinL):
    def __init__(
        self,
        rl_epoch: int,
        model_suite: ModelSuite,
        reward: Reward,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        nft_cfg: DictConfig,
        save_dir: str,
        save_freq: int,
        device: str = None,
        logger: Logger = None,
        replay: bool = False,
        replay_args: Dict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            rl_epoch=rl_epoch,
            model_suite=model_suite,
            reward=reward,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            save_dir=save_dir,
            save_freq=save_freq,
            device=device,
            logger=logger,
            replay=replay,
            replay_args=replay_args,
            **kwargs,
        )

        self.nft_cfg = nft_cfg
        self.ft_global_step = 0

        # Setup advantage computation based on config
        self.use_multiple_pockets = nft_cfg.get('use_multiple_pockets', False)
        if self.use_multiple_pockets:
            self.stat_tracker = PerPocketStatTracker(
                global_std=nft_cfg.get('global_std', False)
            )
            logging.info("Using per-pocket statistics for advantages")
        else:
            self.stat_tracker = None
            logging.info("Using global statistics for advantages")

        # NFT hyperparameters
        self.beta_x = nft_cfg.get('beta_x', 0.5)
        self.beta_h = nft_cfg.get('beta_h', 0.5)
        self.beta_e = nft_cfg.get('beta_e', 0.5)
        self.kl_beta = nft_cfg.get('kl_beta', 0.001)
        self.adv_clip_max = self.reward.cfg.get('adv_clip_max', 1.0)
        self.adv_mode = nft_cfg.get('adv_mode', '')
        self.num_timestep_buckets = nft_cfg.get('num_timestep_buckets', 10)
        self.num_timesteps = nft_cfg.get('num_timesteps', 500)

        self.load_model()

    def load_model(self):
        """Load main agent, old model, and reference model."""
        # Main trainable model
        self.agent = self.model_suite.load_model()
        for param in self.agent.parameters():
            param.requires_grad = True
        self.agent.to(self.device)

        # Old model (for NFT positive/negative predictions)
        self.old_model = self.model_suite.load_model()
        for param in self.old_model.parameters():
            param.requires_grad = False
        self.old_model.to(self.device)

        # Reference model (frozen, for KL regularization)
        if self.kl_beta > 0.0:
            self.reference_model = self.model_suite.load_model()
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.to(self.device)
        else:
            self.reference_model = None

        # Initialize sampling distributions (e.g., size_distribution) for all models
        self.agent.setup_sampling()
        self.old_model.setup_sampling()
        if self.reference_model is not None:
            self.reference_model.setup_sampling()

        self.model_suite.save_model_params(self.save_dir, self.agent)
        logging.info("Loaded agent, old_model, and reference_model")

    def update_old_model(self, decay: float):
        """Update old model parameters using EMA-like update."""
        with torch.no_grad():
            for old_param, new_param in zip(
                self.old_model.parameters(),
                self.agent.parameters(),
                strict=True,
            ):
                old_param.data.copy_(
                    decay * old_param.data + (1.0 - decay) * new_param.data
                )

    def get_n_timesteps(self, batch_size, start_time=0.0, end_time=1.0, timesteps=None, n=None):
        """Divide time range into N buckets and sample one timestep from each bucket for all batch."""
        if timesteps is None:
            timesteps = self.sample_cfg.timesteps
        if n is None:
            n = self.sample_cfg.n
        delta_t = (end_time - start_time) / timesteps
        time_steps = torch.linspace(start_time, end_time-delta_t, timesteps)
        bucket_size = timesteps // n
        buckets = [time_steps[i:i+bucket_size] for i in range(0, timesteps, bucket_size)]
        indexes = [torch.randint(0, len(bucket), (1,)) for bucket in buckets]
        time_steps = [buckets[i][indexes[i]].repeat(batch_size) for i in range(n)]
        return time_steps

    def sample_step(self):
        """Sample molecules using current model."""
        self.agent.eval()

        samples, sc_trajectory = self.sampler.generate(
            model=self.old_model,
            save_dir=self.sample_dir,
            start_time_str=self.start_time_str,
            step=self.step,
            **self.sample_cfg,
        )

        return samples, sc_trajectory

    def compute_advantages(self, samples, rewards):
        """Compute advantages from rewards."""
        if self.use_multiple_pockets:
            # Per-pocket statistics
            pocket_names = samples['pockets']['names']
            advantages = self.stat_tracker.update(pocket_names, rewards)
        else:
            # Global statistics
            advantages = (rewards - rewards.mean())  #/ (rewards.std() + 1e-6) TODO: cite https://arxiv.org/abs/2503.20783 ; (multiple pocket difficulty bias, especially for search-reward)

        return advantages

    def get_batch_timestep_sc_trajectory(self, sc_traj, timestep, node_counts, edge_counts):
        pieces = {k: [] for k in ('vel', 'logits_h', 'logits_e')}
        noatom_idx = atom_encoder['NOATOM']
        nobond_idx = bond_encoder['NOBOND']

        for traj, n_nodes_raw, n_edges_raw in zip(sc_traj, node_counts, edge_counts):
            n_nodes = int(n_nodes_raw)
            n_edges = int(n_edges_raw)

            vel_t = traj['vel'][timestep]
            logits_h_t = traj['logits_h'][timestep]
            logits_e_t = traj['logits_e'][timestep]

            if vel_t.size(0) > n_nodes:
                raise AssertionError(
                    'Self-conditioning atoms and molecules are not aligned'
                )

            if logits_e_t.size(0) > n_edges:
                raise AssertionError(
                    'Self-conditioning edges and bonds are not aligned'
                )

            pieces['vel'].append(vel_t[:n_nodes])
            pieces['logits_h'].append(logits_h_t[:n_nodes])
            pieces['logits_e'].append(logits_e_t[:n_edges])

        return {k: torch.cat(v, dim=0) for k, v in pieces.items()}

    def _setup_optimizer(self):
        return torch.optim.AdamW(
            self.agent.parameters(),
            lr=self.finetune_cfg.lr,
            betas=(self.nft_cfg.get('adam_beta1', 0.9), self.nft_cfg.get('adam_beta2', 0.999)),
            weight_decay=self.nft_cfg.get('adam_weight_decay', 1e-4),
        )

    def _run_inner_epoch(self, epoch_idx, loader, optimizer, sc_trajectory, advantages_tensor):
        """Runs a single inner epoch of NFT training."""
        logging.info(f"Inner Epoch {epoch_idx} starts:")

        # Set all models to eval mode for consistent self-conditioning behavior
        # This ensures deterministic outputs from all models
        self.agent.eval()
        self.old_model.eval()
        if self.reference_model is not None:
            self.reference_model.eval()

        total_loss = 0.0
        num_samples = 0
        epoch_losses = {}
        cfg = self.finetune_cfg
        nft_cfg = self.nft_cfg

        for batch_idx, batch in enumerate(loader):
            # Move batch to device AND create fresh copies with detached gradients to avoid in-place modification issues
            batch = {
                "ligand": TensorDict(**{k: v.clone().detach() if torch.is_tensor(v) else v for k, v in batch["ligand"].items()}).to(self.device),
                "pocket": TensorDict(**{k: v.clone().detach() if torch.is_tensor(v) else v for k, v in batch["pocket"].items()}).to(self.device),
            }

            ligand = TensorDict(**batch["ligand"])
            pocket = Residues(**batch["pocket"])
            batch_size = len(ligand['size'])

            # Get corresponding rewards and advantages for this batch
            # NOTE: We need to use the actual batch_size, not cfg.batch_size
            start_idx = batch_idx * cfg.batch_size
            end_idx = start_idx + batch_size  # Use actual batch size, not min()
            batch_advantages = advantages_tensor[start_idx:end_idx].to(self.device)
            batch_sc_trajectory = sc_trajectory[start_idx:end_idx]

            # Sample timesteps from buckets
            # get_n_timesteps returns a list of tensors, one per bucket
            timesteps_list = self.get_n_timesteps(
                batch_size=batch_size,
                start_time=0.0,
                end_time=1.0,
                timesteps=self.num_timesteps,
                n=self.num_timestep_buckets
            )

            # Zero gradients before accumulating over timesteps
            optimizer.zero_grad()

            # Accumulate gradients over all sampled timesteps
            batch_loss = 0.0
            batch_loss_dict = {}

            for timestep_idx, timesteps in enumerate(timesteps_list):
                # timesteps shape: [batch_size]
                timesteps = timesteps.to(self.device)
                node_counts = ligand['size'].tolist()
                edge_counts = ligand['n_bonds'].tolist()

                batch_sc_trajectory_timestep = self.get_batch_timestep_sc_trajectory(batch_sc_trajectory, timestep_idx, node_counts, edge_counts)

                # NFT step (ligand and pocket are already fresh clones per batch)
                loss, loss_dict = self.agent.nft_step(
                    ligand=ligand,
                    pocket=pocket,
                    timesteps=timesteps,
                    old_model=self.old_model,
                    reference_model=self.reference_model,
                    advantages=batch_advantages,
                    beta_x=self.beta_x,
                    beta_h=self.beta_h,
                    beta_e=self.beta_e,
                    kl_beta=self.kl_beta,
                    adv_clip_max=self.adv_clip_max,
                    adv_mode=self.adv_mode,
                    sc_ligand=batch_sc_trajectory_timestep,
                )

                # Normalize loss by number of timesteps (for gradient accumulation)
                normalized_loss = loss / self.num_timestep_buckets
                normalized_loss.backward()

                # Accumulate loss for logging
                batch_loss += loss.item()
                for k, v in loss_dict.items():
                    if k not in batch_loss_dict:
                        batch_loss_dict[k] = 0.0
                    batch_loss_dict[k] += v

            # Average accumulated losses
            batch_loss /= self.num_timestep_buckets
            for k in batch_loss_dict:
                batch_loss_dict[k] /= self.num_timestep_buckets

            # Gradient clipping
            if nft_cfg.get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(),
                    nft_cfg.max_grad_norm
                )

            # Optimizer step after accumulating over all timesteps
            optimizer.step()

            # Accumulate epoch statistics
            total_loss += batch_loss * batch_size
            num_samples += batch_size

            for k, v in batch_loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)

            # Log batch-level metrics
            if self.logger is not None:
                self.logger.log({
                    **{f"train/{k}": v for k, v in batch_loss_dict.items()},
                    "global_step": self.ft_global_step,
                    "advantage_mean": batch_advantages.mean().item(),
                    "advantage_std": batch_advantages.std().item(),
                })

            self.ft_global_step += 1

            # Log batch info
            log_str = [f'{k}: {v:.4f}' for k, v in batch_loss_dict.items()]
            logging.info(
                f"Batch {self.ft_global_step} - {', '.join(log_str)}, "
                f"adv: {batch_advantages.mean():.2f}±{batch_advantages.std():.2f}"
            )

        return total_loss, num_samples, epoch_losses

    def ft_step(self, samples, sc_trajectory, rewards, advantages):
        cfg = self.finetune_cfg
        nft_cfg = self.nft_cfg

        num_samples = len(samples['ligands']['generated_mols'])
        logging.info(
            f"NFT training with {num_samples} samples, "
            f"batch size: {cfg.batch_size}, "
            f"inner epochs: {nft_cfg.inner_epochs}, "
            f"num timestep buckets: {nft_cfg.num_timestep_buckets}"
        )

        if num_samples == 0:
            logging.info("No samples to finetune, skipping finetuning")
            return

        # Get dataloader from samples (without reward/advantage)
        loader = self.model_suite.get_dataloader(
            samples=samples,
            batch_size=cfg.batch_size,
            shuffle=False,                        # !!!!No shuffling to keep the order of the samples are the same as sc-trajectory
        )

        # Convert rewards and advantages to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

        # Setup optimizer
        optimizer = self._setup_optimizer()

        # Training loop with inner epochs
        for inner_epoch in range(nft_cfg.get('inner_epochs', 1)):
            total_loss, num_samples, epoch_losses = self._run_inner_epoch(
                inner_epoch, loader, optimizer, sc_trajectory, advantages_tensor
            )

            # Calculate and log epoch-level metrics
            epoch_loss = total_loss / num_samples
            epoch_summary = {k: np.mean(v) for k, v in epoch_losses.items()}
            log_str = [f'{k}: {v:.4f}' for k, v in epoch_summary.items()]
            logging.info(
                f"Inner Epoch {inner_epoch} finished - "
                f"avg_loss: {epoch_loss:.4f}, {', '.join(log_str)}"
            )

        # Update old model after all inner epochs
        decay = return_decay(self.step, nft_cfg.get('decay_type', 1))
        self.update_old_model(decay)
        logging.info(f"Updated old_model with decay={decay:.4f}")

    def rl_step(self):
        """Single RL iteration: sample -> score -> train."""
        logging.info(f'***** LOOP {self.step} START *****')

        # Sample
        logging.info('SAMPLING:')
        sample_list, sc_trajectory = self.sample_step()

        # Score and reward
        logging.info('SCORING:')
        sample_list, smiles, rewards, prop_dict, failed_mask = self.reward_step(
            sample_list,
            f'step_{self.step:04d}'
        )

        # No longer filtering sc_trajectory since reward_step keeps failed samples with 0 reward
        # if sc_trajectory is not None:
        #     sc_trajectory = [t for t, failed in zip(sc_trajectory, failed_mask) if not failed]

        # Diversity filter: penalize rewards of over-represented scaffolds before
        # computing advantages (raw rewards are kept for long-term-memory metrics).
        filtered_rewards, df_stats = self.div_filter.update(smiles, rewards, self.step)
        if self.div_filter.enabled:
            logging.info(
                f"DIVERSITY FILTER: {df_stats['penalized']} penalized, "
                f"{df_stats['buckets']} scaffold buckets, "
                f"mean mult {df_stats['mean_multiplier']:.3f}"
            )

        # Compute advantages
        logging.info('COMPUTING ADVANTAGES:')

        if self.reward.cfg.get('adv_clip_max') == 'dynamic':
            self.adv_clip_max = np.max(np.abs(filtered_rewards)) if len(filtered_rewards) else 1.0
            logging.info(f'Dynamic adv_clip_max: {self.adv_clip_max:.4f}')
        else:
             self.adv_clip_max = self.reward.cfg.get('adv_clip_max', 1.0)

        advantages = self.compute_advantages(sample_list, filtered_rewards)

        # Log statistics
        log_dict = {
            f'reward/{k}_mean': v.mean() for k, v in prop_dict.items()
        }
        log_dict.update({
            f'reward/{k}_std': v.std() for k, v in prop_dict.items()
        })
        log_dict.update({
            'reward/mean': rewards.mean(),
            'reward/std': rewards.std(),
            'reward/filtered_mean': filtered_rewards.mean(),
            'advantage/mean': advantages.mean(),
            'advantage/std': advantages.std(),
            'diversity_filter/penalized': df_stats['penalized'],
            'diversity_filter/buckets': df_stats['buckets'],
            'diversity_filter/mean_multiplier': df_stats['mean_multiplier'],
        })

        # Stat tracker metrics (only for per-pocket mode)
        if self.use_multiple_pockets and self.stat_tracker:
            group_size, trained_pockets = self.stat_tracker.get_stats()
            log_dict.update({
                'stats/group_size': group_size,
                'stats/trained_pockets': trained_pockets,
                'stats/mean_reward_top10': self.stat_tracker.get_mean_of_top_rewards(90),
                'stats/mean_reward_top25': self.stat_tracker.get_mean_of_top_rewards(75),
                'stats/mean_reward_top50': self.stat_tracker.get_mean_of_top_rewards(50),
            })
        # Long-term memory
        self.ltm.extend(smiles, rewards, self.step)
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        self.ltm.save(os.path.join(self.sample_dir, 'long_term_memory.csv'))
        burden_str = f'{metrics[0]:.3f}' if metrics[0] is not None else 'N/A'
        div_ratio_str = f'{metrics[1]:.3f}' if metrics[1] is not None else 'N/A'

        logging.info(
            f'{len(self.ltm)} molecules generated, '
            f'{len(self.ltm.unique_smiles)} unique | '
            f'Burden: {burden_str}, Div. Ratio: {div_ratio_str}'
        )

        log_dict.update({
            'ltm/total_molecules': len(self.ltm),
            'ltm/unique_molecules': len(self.ltm.unique_smiles),
            'ltm/burden': metrics[0] if metrics[0] is not None else 0.0,
            'ltm/diversity_ratio': metrics[1] if metrics[1] is not None else 0.0,
            'cost': self.cost,
        })

        if self.logger is not None:
            self.logger.log(log_dict)
        logging.info('NFT TRAINING:')
        self.ft_step(sample_list, sc_trajectory, rewards, advantages)

        # Clear stat tracker periodically (only for multiple pockets case)
        if self.use_multiple_pockets and self.stat_tracker:
            if (self.step + 1) % self.nft_cfg.get('clear_stats_freq', 10) == 0:
                self.stat_tracker.clear()
                logging.info("Cleared per-pocket stat tracker")

        logging.info(f'***** LOOP {self.step} FINISH *****\n')

    def run_rl(self):
        """Run the complete NFT RL loop."""
        logging.info('***** NFT RL START *****')
        start_time = time.time()
        self.start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        logging.info(f'Start time: {self.start_time_str}')

        for step in range(self.rl_epoch):
            self.step = step
            self.rl_step()

            # Save checkpoints
            if (step + 1) % self.save_freq == 0:
                ckpt_dir = os.path.join(self.models_dir, f'loop_{step:04d}')
                self.model_suite.save_model(self.agent, ckpt_dir)
                # Also save old model
                if self.old_model is not None:
                    old_dir = os.path.join(self.models_dir, f'loop_{step:04d}_old')
                    self.model_suite.save_model(self.old_model, old_dir)

        # Final save
        ckpt_dir = os.path.join(self.models_dir, 'final')
        self.model_suite.save_model(self.agent, ckpt_dir)

        logging.info('***** NFT RL END *****')
        end_time = time.time()
        logging.info(f'Total time: {int(end_time - start_time)} seconds')
