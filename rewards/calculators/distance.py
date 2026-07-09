from typing import Any, Dict, List
import numpy as np
import torch
from rdkit import Chem

from rewards.calculators.base import Calculator
from rewards.calculators.medchem import MedChemCalc


class PocketLigandDistanceCalc(Calculator):
    def __init__(self, task: str = 'center_distance') -> None:
        super().__init__(task)

    def calc(self, samples: Any, label: str = 'tmp'):
        if not (isinstance(samples, dict) and 'ligands' in samples and 'pockets' in samples):
            n = len(samples) if isinstance(samples, list) else 0
            res = np.full(n, np.nan, dtype=float)
            return res, {'center_distance': res}

        lig_mols: List[Chem.Mol] = samples['ligands'].get('mols', [])
        lig_mols = [mol if MedChemCalc.is_valid(mol) else None for mol in lig_mols]
        n = len(lig_mols)

        pockets = samples['pockets']
        pocket_x_list = pockets.get('x', None)
        distances: List[float] = []

        for i in range(n):
            mol = lig_mols[i]
            lig_center = self._lig_center(mol)
            pocket_x = None if pocket_x_list is None else pocket_x_list[i]
            distances.append(self._pair_distance(lig_center, pocket_x))

        results = np.asarray(distances, dtype=float)

        return results

    @staticmethod
    def _lig_center(mol: Chem.Mol):
        try:
            conf = mol.GetConformer()
            pos = np.asarray(conf.GetPositions(), dtype=float)
            if pos.size == 0:
                return None
            return pos.mean(axis=0)
        except Exception:
            return None

    @staticmethod
    def _to_numpy(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _pair_distance(self, lig_center, pocket_x):
        try:
            if lig_center is None or pocket_x is None:
                return np.nan
            px = self._to_numpy(pocket_x)
            if px is None or px.ndim != 2 or px.shape[1] != 3 or px.size == 0:
                return np.nan
            pocket_center = px.mean(axis=0)
            return float(np.linalg.norm(lig_center - pocket_center))
        except Exception:
            return np.nan
