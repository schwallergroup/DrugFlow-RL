import os
import time
import logging
from typing import Dict
import numpy as np
import torch
from omegaconf import DictConfig

from pipeline.base import ReinL
from pipeline.utils.save import save_smiles
from pipeline.utils.logger import Logger
from models.suite.base import ModelSuite
from rewards.reward import Reward

from src.data.data_utils import TensorDict

class HillClimb(ReinL):
    def __init__(
        self,
        rl_epoch: int,
        model_suite: ModelSuite,
        reward: Reward,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        topk_ratio: float,
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
        assert topk_ratio > 0.0 and topk_ratio <= 1.0
        self.topk_ratio = topk_ratio
        self.load_model()

    def load_model(self):
        self.agent = self.model_suite.load_model()
        for param in self.agent.parameters():
            param.requires_grad = True
        self.agent.to(self.device)

        self.model_suite.save_model_params(self.save_dir, self.agent)

    def sample_step(self):
        samples = self.sampler.generate(
            model=self.agent, **self.sample_cfg,
        )

        # max sample size to score/reward
        if self.sample_cfg.get('eval_size'):
            max_num = int(self.sample_cfg.eval_size)
            for k in samples['ligands']:
                samples['ligands'][k] = samples['ligands'][k][:max_num]
            for k in samples['pockets']:
                samples['pockets'][k] = samples['pockets'][k][:max_num]

        save_smiles(
            molecules=samples['ligands']['mols'],
            save_dir=self.sample_dir,
            filename=f'{self.start_time_str}_samples.txt',
            step=self.step,
        )

        return samples

    def ft_step(self, data_list):
        cfg = self.finetune_cfg

        logging.info(f"Finetuning with {len(data_list['ligands']['mols'])} samples, batch size: {cfg.batch_size}")

        if len(data_list['ligands']['mols']) == 0:
            logging.info("No samples to finetune, skipping finetuning")
            return

        loader = self.model_suite.get_dataloader(
            samples=data_list,
            batch_size=cfg.batch_size,
        )

        optimizer = torch.optim.AdamW(self.agent.parameters(), lr=cfg.lr)
        for epoch in range(cfg.epochs):
            logging.info(f"Epoch {epoch} starts:")
            self.agent.train()

            total_loss = 0.0
            num_samples = 0

            for batch in loader:
                batch = {
                    "ligand": TensorDict(**batch["ligand"]).to(self.device),
                    "pocket": TensorDict(**batch["pocket"]).to(self.device),
                }
                optimizer.zero_grad()

                loss, loss_dict = self.agent.ft_step(batch)
                loss.backward()
                optimizer.step()

                if self.logger is not None:
                    payload = {f"{k}": v for k, v in loss_dict.items()}
                    self.logger.log(payload)
                self.ft_global_step += 1

                batch_size = len(batch["ligand"]["size"])
                total_loss += loss.item() * batch_size
                num_samples += batch_size

                # Log batch-level metrics
                log_str = [f'{k}: {v:.4f}' for k, v in loss_dict.items()]
                logging.info(', '.join(log_str))

            # Calculate and log epoch-level metrics
            epoch_loss = total_loss / num_samples
            logging.info(f"Epoch {epoch} loss: {epoch_loss:.4f}")

    def rl_step(self):
        logging.info(f'*****   LOOP {self.step} START   *****')

        logging.info('SAMPLE:')
        sample_list = self.sample_step()

        # sample scoring, remove failed samples, ranking and get top k samples
        logging.info('SCORE:')
        sample_list, smiles, rewards, prop_dict, _ = self.reward_step(sample_list, f'step_{self.step:0>4d}')

        # Diversity filter: penalize over-represented scaffolds so they drop out
        # of the top-k selection (raw rewards kept for long-term-memory metrics).
        filtered_rewards, df_stats = self.div_filter.update(smiles, rewards, self.step)
        if self.div_filter.enabled:
            logging.info(
                f"DIVERSITY FILTER: {df_stats['penalized']} penalized, "
                f"{df_stats['buckets']} scaffold buckets, "
                f"mean mult {df_stats['mean_multiplier']:.3f}"
            )

        log_dict = {f'{k} mean': v.mean() for k, v in prop_dict.items()}
        log_dict.update({f'{k} std': v.std() for k, v in prop_dict.items()})
        log_dict.update({'reward mean': rewards.mean(), 'reward std': rewards.std()})
        log_dict.update({
            'reward filtered_mean': filtered_rewards.mean(),
            'diversity_filter/penalized': df_stats['penalized'],
            'diversity_filter/buckets': df_stats['buckets'],
            'diversity_filter/mean_multiplier': df_stats['mean_multiplier'],
        })

        # long-term memory # TODO: checking uniqueness with smiles is not ideal, look into other ways to check uniqueness
        self.ltm.extend(smiles, rewards, self.step)
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        self.ltm.save(os.path.join(self.sample_dir, 'long_term_memory.csv'))
        logging.info(
            f'{len(self.ltm)} molecules generated so far, ' +
            f'{len(self.ltm.unique_smiles)} unique molecules' +
            f'  Burden: {metrics[0]}, Div. Ratio: {metrics[1]}.'
        )
        log_dict.update(
            {
                'molecule_num': len(self.ltm),
                'unique_molecules': len(self.ltm.unique_smiles),
                'burden': metrics[0],
                'div_ratio': metrics[1],
                'cost': self.cost,
            }
        )
        if self.logger is not None:
            self.logger.log(log_dict)

        # topk data points (ranked by diversity-filtered reward)
        sort_idx = np.argsort(filtered_rewards)[::-1]
        topk_idx = sort_idx[: int(len(filtered_rewards) * self.topk_ratio)]
        sample_topk = {'ligands': {}, 'pockets': {}}
        for key in sample_list['ligands']:
            sample_topk['ligands'][key] = [sample_list['ligands'][key][i] for i in topk_idx]
        for key in sample_list['pockets']:
            sample_topk['pockets'][key] = [sample_list['pockets'][key][i] for i in topk_idx]
        reward_topk = rewards[topk_idx]

        # experience replay # TODO: adapt replay buffer to the sbdd pipeline
        if self.replay is not None:
            pass
            # data_replay, reward_replay = self.replay.sample()
            # ft_data = sample_topk + data_replay
            # ft_reward = np.concatenate((reward_topk, reward_replay))
            # self.replay.extend(sample_topk, reward_topk)
            # logging.info(f'replay buffer size={len(self.replay)}')
            # print(f'replay rewards={reward_replay}')
            # logging.info(f'buffer reward mean={self.replay.buffer["reward"].values.mean()}')
            # print(f'buffer rewards={replay.buffer["reward"].values}')
        else:
            ft_data = sample_topk
            ft_reward = reward_topk

        # save finetuning data
        save_smiles(
            molecules=ft_data['ligands']['mols'],
            save_dir=self.sample_dir,
            filename=f'{self.start_time_str}_ft_samples.txt',
            step=self.step,
        )
        # finetuning
        logging.info('FINETUNE:')
        self.ft_step(ft_data)

        logging.info(f'*****   LOOP {self.step} FINISH   *****\n\n')

    def run_rl(self):
        logging.info('*****   RL START   *****')
        start_time = time.time()
        self.start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logging.info(f'Start time: {self.start_time_str}')

        for step in range(self.rl_epoch):
            self.step = step
            self.rl_step()
            # Save the agent weights every few iterations
            if (step + 1) % self.save_freq == 0:
                ckpt_dir = os.path.join(self.models_dir, f'loop_{step:0>4d}')
                self.model_suite.save_model(self.agent, ckpt_dir)
        # If the entire training finishes, clean up
        ckpt_dir = os.path.join(self.models_dir, 'final')
        self.model_suite.save_model(self.agent, ckpt_dir)

        logging.info('*****   RL END   *****')
        end_time = time.time()
        logging.info('Total time taken: {} s.'.format(int(end_time - start_time)))
