import logging
from typing import Tuple, List, Optional, Dict
import torch
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from functools import partial
from rdkit import Chem
from collections import defaultdict
import numpy as np

from src.data.dataset import ProcessedLigandPocketDataset
from src.data.data_utils import TensorDict, split_entity
from src.gen_utils import filter_mols
from pipeline.utils.logger import Logger
from pipeline.utils.save import save_smiles
from src.model.lightning import DrugFlow

from models.drugflow.sampling_utils import (
    get_evaluators,
    calculate_molecule_statistics,
    compute_internal_diversity,
    get_metric_score,
    prepare_input_data,
    get_size_model
)

class BaseSampler:
    def __init__(
        self,
        batch_size: int = 64,
        n_samples: int = 128,
        n_steps: Optional[int] = None,
        pocket_distance_cutoff: float = 8.0,
        molecule_size: Optional[str] = None,
        sample_with_ground_truth_size: bool = False,
        starting_point: Optional[str] = None,
        protein_path: Optional[str] = None,
        ligand_path: Optional[str] = None,
        size_checkpoint: Optional[str] = None,
        filter: bool = True,
        filter_criterion: str = 'valid_and_connected',
        metrics: List[str] = ["medchem"],
        logger: Optional[Logger] = None,
        noise_scale: float = 0.0,
        temperature: float = 1.0,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.pocket_distance_cutoff = pocket_distance_cutoff
        self.molecule_size_str = molecule_size
        self.sample_with_ground_truth_size = sample_with_ground_truth_size
        self.starting_point = starting_point
        self.protein_path = protein_path
        self.ligand_path = ligand_path
        self.size_checkpoint = size_checkpoint
        self.filter = filter
        self.filter_criterion = filter_criterion
        self.metrics = metrics
        self.logger = logger
        self.noise_scale = noise_scale
        self.temperature = temperature

        self.metric_evaluator, self.filter_evaluator = get_evaluators(metrics)
        self.device = None
        self.loader = None
        self.size_model = None
        self.molecule_size = None

    def _setup_models(self, device):
        self.device = device
        if self.size_model is None:
            self.size_model, self.molecule_size = get_size_model(
                self.molecule_size_str, self.size_checkpoint, self.device, self.batch_size
            )

    def _get_loader(self, model):
        logging.info("Preparing input...")
        ligand, pocket = prepare_input_data(
            self.protein_path, self.ligand_path, self.pocket_distance_cutoff,
            model, self.starting_point
        )
        logging.info(f"Input pocket size: {pocket['size'].item()}")

        dataset = [{'ligand': ligand, 'pocket': pocket} for _ in range(self.batch_size)]
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None),
            pin_memory=True
        )

    def _run_sampling_loop(self, model):
        smiles = set()
        samples = {
            'ligands': {'mols': [], 'generated_mols': []},
            'pockets': {}
        }

        logging.info(f"Sampling {self.n_samples} molecules with batch size {self.batch_size}")

        mols = []
        pockets = []
        all_sc_trajectories = []
        valid_mols_list = []
        valid_pockets_list = []

        while len(smiles) < self.n_samples:
            for data in self.loader:
                new_data = {
                    'ligand': TensorDict(**data['ligand']).to(self.device),
                    'pocket': TensorDict(**data['pocket']).to(self.device),
                }
                pocket_per_item = split_entity(new_data['pocket'])

                rdmols_generated, rdmols, rdpockets, _, sc_trajectory = model.sample(
                    new_data,
                    n_samples=1,
                    timesteps=self.n_steps,
                    num_nodes=self.molecule_size,
                    size_model=self.size_model,
                    sample_with_ground_truth_size=self.sample_with_ground_truth_size,
                    return_sc_ligand=True,
                    noise_scale=self.noise_scale,
                    temperature=self.temperature,
                )

                mols.extend(rdmols)
                pockets.extend(rdpockets)

                if not self.filter:
                    self._process_batch_no_filter(
                        rdmols, rdmols_generated, sc_trajectory, rdpockets, pocket_per_item,
                        smiles, samples, all_sc_trajectories, valid_mols_list, valid_pockets_list
                    )
                else:
                    self._process_batch_with_filter(
                        rdmols, rdmols_generated, sc_trajectory, rdpockets, pocket_per_item,
                        smiles, samples, all_sc_trajectories, valid_mols_list, valid_pockets_list
                    )

                if len(smiles) >= self.n_samples:
                    break

        return samples, mols, pockets, all_sc_trajectories, valid_mols_list, valid_pockets_list

    def _process_batch_no_filter(self, rdmols, rdmols_gen, sc_traj, rdpockets, pocket_items, smiles, samples, out_traj, valid_mols_list, valid_pockets_list):
        for i, mol in enumerate(rdmols):
            try:
                mol_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
            except Exception:
                mol_smiles = None

            if mol_smiles is None or mol_smiles not in smiles:
                self._add_sample(samples, pocket_items[i], mol, rdmols_gen[i])
                if sc_traj is not None:
                    out_traj.append(sc_traj[i])
                if mol_smiles is not None:
                    smiles.add(mol_smiles)
                valid_mols_list.append(mol)
                valid_pockets_list.append(rdpockets[i])

    def _process_batch_with_filter(self, rdmols, rdmols_gen, sc_traj, rdpockets, pocket_items, smiles, samples, out_traj, valid_mols_list, valid_pockets_list):
        with tempfile.TemporaryDirectory() as tmpdir:
            protein_ps = [Path(tmpdir, f'protein_{i}.pdb') for i in range(len(rdmols))]
            for receptor, pocket in zip(protein_ps, rdpockets):
                Chem.MolToPDBFile(pocket, str(receptor))

            mols_filt, _, selected_indices = filter_mols(
                rdmols, protein_ps, self.filter_evaluator, smiles=smiles,
                criterion=self.filter_criterion, preselect_unique=True
            )
            logging.info(f'Filtered {len(mols_filt)} molecules out of {len(rdmols)}')

            for i, mol in enumerate(mols_filt):
                original_idx = selected_indices[i]
                mol_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                if mol_smiles not in smiles:
                    self._add_sample(samples, pocket_items[original_idx], mol, rdmols_gen[original_idx])
                    if sc_traj is not None:
                        out_traj.append(sc_traj[original_idx])
                    smiles.add(mol_smiles)
                    valid_mols_list.append(mol)
                    valid_pockets_list.append(rdpockets[original_idx])


    def _add_sample(self, samples, pocket_item, mol, generated_mol):
        if not samples['pockets']:
            for key in pocket_item.keys():
                samples['pockets'][key] = []

        for key, value in pocket_item.items():
            if torch.is_tensor(value):
                value = value.detach().cpu()
            samples['pockets'][key].append(value)

        samples['ligands']['mols'].append(mol)
        samples['ligands']['generated_mols'].append(generated_mol)

    def _compute_and_log_metrics(self, mols, pockets, kept_mols, valid_pockets=None):
        mol_stats, valid_mols = calculate_molecule_statistics(mols, self.filter_evaluator)
        logging.info(f"Molecule Statistics: {mol_stats}")

        mol_stats['internal_diversity_unique'], mol_stats['n_diversity_failed_unique'] = \
            compute_internal_diversity(kept_mols)

        results = []
        if self.metrics:
            if valid_pockets is not None:
                results = get_metric_score(kept_mols, valid_pockets, self.metric_evaluator)
            else:
                results = get_metric_score(mols, pockets, self.metric_evaluator)
            self._log_summary(results, mol_stats)

        return valid_mols, results

    def _log_summary(self, metric_results, mol_stats):
        vals = defaultdict(list)
        for res in metric_results:
            for k, v in res.items():
                if v is None: continue
                if '.' in k:
                    prefix, base = k.split('.', 1)
                    name = f'{prefix} {base}'
                else:
                    prefix, base, name = None, k, k
                if base in ('smiles', 'time', 'relaxed_molecule'): continue
                if isinstance(v, (int, float, np.floating, np.integer)):
                    vals[name].append(float(v))

        summary = {}
        for k, v in vals.items():
            a = np.array(v, dtype=float)
            summary[f'{k}_mean'] = float(np.nanmean(a)) if a.size else float('nan')
            summary[f'{k}_std'] = float(np.nanstd(a)) if a.size else float('nan')

        log_payload = {}
        for key, value in summary.items():
            if ' ' in key:
                log_payload[key.replace(' ', '/', 1)] = value
            else:
                log_payload[key] = value

        for k, v in mol_stats.items():
            if k == 'internal_diversity_unique':
                log_payload['medchem/internal_diversity'] = v
            elif k == 'n_diversity_failed_unique':
                log_payload['medchem/n_diversity_failed'] = v
            else:
                log_payload[f'generations/{k}'] = v

        logging.info(log_payload)
        if self.logger:
            self.logger.log(log_payload)

    def _save_results(self, samples, valid_mols, **kwargs):
        save_dir = kwargs.get('save_dir')
        start_time_str = kwargs.get('start_time_str')
        step = kwargs.get('step')

        if save_dir and start_time_str and step is not None:
            try:
                save_smiles(
                    molecules=samples['ligands']['mols'],
                    save_dir=save_dir,
                    filename=f'{start_time_str}_samples.txt',
                    step=step,
                )
                if valid_mols:
                    save_smiles(
                        molecules=valid_mols,
                        save_dir=save_dir,
                        filename=f'{start_time_str}_all_samples.txt',
                        step=step,
                    )
            except Exception as e:
                logging.warning(f"Failed to save SMILES: {e}")

    def generate(self, model: DrugFlow, **kwargs) -> Tuple[List[Data], List[Chem.Mol]]:
        model.eval()
        self._setup_models(model.device)
        if self.loader is None:
            self.loader = self._get_loader(model)

        with torch.no_grad():
            samples, mols, pockets, sc_trajectories, _, valid_pockets = self._run_sampling_loop(model)

        valid_mols, metrics_results = self._compute_and_log_metrics(
            mols, pockets, samples['ligands']['mols'], valid_pockets=valid_pockets
        )
        self._save_results(samples, valid_mols, **kwargs)

        if metrics_results:
            samples['metrics'] = metrics_results

        return samples, sc_trajectories


class SBDDSampler(BaseSampler):
    """
    Standard DrugFlow Sampler.
    Returns dictionary of samples.
    """
    def generate(self, model: DrugFlow, **kwargs) -> Tuple[List[Data], List[Chem.Mol]]:
        samples, _ = super().generate(model, **kwargs)
        return samples


class SBDDNFTSampler(BaseSampler):
    """
    NFT-specific Sampler.
    Returns (samples, sc_trajectory).
    """
    def generate(self, model: DrugFlow, **kwargs) -> Tuple[List[Data], List[Chem.Mol]]:
        samples, sc_trajectory = super().generate(model, **kwargs)
        return samples, sc_trajectory
