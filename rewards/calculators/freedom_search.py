import os
import multiprocessing as mp
from typing import Optional, List

import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import rdSynthonSpaceSearch

from rewards.calculators.base import Calculator

def _worker_exact_match(mol: Chem.Mol) -> bool:
    return _GLOBAL_CALC_INSTANCE._exact_match(mol)


_GLOBAL_CALC_INSTANCE: "FreedomSearchCalc" = None

class FreedomSearchCalc(Calculator):
    def __init__(
        self,
        task: str = "freedom_exact_match",
        db_path: str = None,
        txt_path: Optional[str] = None,
        max_hits: int = 5000,
        n_jobs: int = -1, #TODO: add this as a config option
        **kwargs,
    ) -> None:
        super().__init__(task)

        # Path to the SynthonSpace '.spc' DB. If not passed explicitly, fall back
        # to the 'FREEDOM_SYNTHONS' environment variable (mirrors how the Enamine
        # calculator reads 'REAL_API_KEY').
        self.db_path = db_path or os.getenv("FREEDOM_SYNTHONS")
        if not self.db_path:
            raise ValueError(
                "Freedom SynthonSpace DB path not found. Provide it via 'db_path' "
                "or set the 'FREEDOM_SYNTHONS' environment variable."
            )
        self.txt_path = txt_path
        self.max_hits = max_hits
        hw_threads = os.cpu_count() or 1
        if n_jobs <= 0:
            self.n_jobs = max(1, hw_threads + n_jobs)
        else:
            self.n_jobs = min(n_jobs, hw_threads)

        self.synthonspace = rdSynthonSpaceSearch.SynthonSpace()
        if not os.path.exists(self.db_path):
            if self.txt_path and os.path.exists(self.txt_path):
                self.synthonspace.ReadTextFile(self.txt_path)
                self.synthonspace.WriteDBFile(self.db_path)
            else:
                raise FileNotFoundError(
                    f"SynthonSpace DB not found at '{self.db_path}'. "
                    f"Provide a valid '.spc' via db_path or a valid text file via txt_path."
                )

        self.synthonspace.ReadDBFile(self.db_path)

        print(f"Number of reactions at Freedom 4.0: {self.synthonspace.GetNumReactions()}")
        print(f"Number of products at Freedom 4.0: {self.synthonspace.GetNumProducts()}")

        self.params = rdSynthonSpaceSearch.SynthonSpaceSearchParams()
        self.params.maxHits = int(self.max_hits)

    def calc(self, sample_dict: dict, label: str = "tmp") -> np.ndarray:
        samples: List[Chem.Mol] = sample_dict.get("ligands", {}).get("mols", [])
        if not samples:
            return np.array([], dtype=float)

        use_multiprocessing = self.n_jobs > 1 and len(samples) > 1
        if use_multiprocessing:
            global _GLOBAL_CALC_INSTANCE
            _GLOBAL_CALC_INSTANCE = self
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=self.n_jobs) as pool:
                bool_results = pool.map(_worker_exact_match, samples)
        else:
            bool_results = [self._exact_match(mol) for mol in samples]

        return np.array([1.0 if r else 0.0 for r in bool_results], dtype=float)

    @staticmethod
    def _canonical_smiles(mol: Chem.Mol) -> Optional[str]:
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    def _exact_match(self, rdmol: Chem.Mol) -> bool:
        if rdmol is None or rdmol.GetNumAtoms() < 1:
            return False

        try:
            query_mol = Chem.Mol(rdmol)
            Chem.SanitizeMol(query_mol)
        except Exception:
            return False

        query_smiles = self._canonical_smiles(query_mol)
        if not query_smiles:
            return False

        results = self.synthonspace.SubstructureSearch(query_mol)
        results_mols = results.GetHitMolecules()
        for mol in results_mols:
            if self._canonical_smiles(mol) == query_smiles:
                return True

        try:
            if len(results_mols) >= 1000:
                deep_results = self.synthonspace.SubstructureSearch(query_mol, self.params)
                for mol in deep_results.GetHitMolecules():
                    if self._canonical_smiles(mol) == query_smiles:
                        return True
        except Exception:
            return False

        return False