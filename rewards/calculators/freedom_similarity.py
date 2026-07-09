# rewards/calculators/freedom_similarity.py
import os
from typing import Optional, List, Tuple
import numpy as np
from multiprocessing import Pool, get_context
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdSynthonSpaceSearch
from rdkit.Chem import rdFingerprintGenerator as rFG

from rewards.calculators.base import Calculator

_GLOBAL_CALC_INSTANCE = None

def _init_worker(global_calc):
    global _GLOBAL_CALC_INSTANCE
    _GLOBAL_CALC_INSTANCE = global_calc

def _worker_similarity(args: Tuple[int, str]) -> Tuple[int, float]:
    idx, smi = args
    try:
        if not smi:
            return idx, 0.0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 1:
            return idx, 0.0
        Chem.SanitizeMol(mol)
        return idx, _GLOBAL_CALC_INSTANCE._max_similarity_against_hits(mol)
    except Exception:
        return idx, 0.0

class FreedomSimilarityCalc(Calculator):
    def __init__(
        self,
        task: str = "freedom_similarity",
        db_path: str = None,
        txt_path: Optional[str] = None,
        max_hits: int = 5000,
        mode: str = "rdkit",     # "tanimoto" or "morgan"
        radius: int = 2,            # for Morgan
        nbits: int = 2048,          # FP size
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
        self.max_hits = int(max_hits)
        self.mode = mode.lower().strip()
        if self.mode not in ("rdkit", "morgan"):
            raise ValueError("mode must be 'rdkit' or 'morgan'.")

        self.radius = int(radius)
        self.nbits = int(nbits)
        self.n_jobs = int(n_jobs)

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

        self.params = rdSynthonSpaceSearch.SynthonSpaceSearchParams()
        self.params.maxHits = self.max_hits

        self._morgan_gen = None
        self._rdkit_gen = None

    def calc(self, sample_dict: dict, label: str = "tmp") -> np.ndarray:
        mols: List[Chem.Mol] = sample_dict["ligands"]["mols"]
        return self._calc_from_mols(mols)

    def __call__(self, sample_dict_or_mols):
        if isinstance(sample_dict_or_mols, dict):
            return self.calc(sample_dict_or_mols)
        else:
            return self._calc_from_mols(sample_dict_or_mols)

    def _calc_from_mols(self, mols: List[Chem.Mol]) -> np.ndarray:
        smiles: List[str] = []
        for m in mols:
            try:
                smi = Chem.MolToSmiles(m, canonical=True) if m is not None else None
            except Exception:
                smi = None
            smiles.append(smi)

        n = len(smiles)
        out = np.zeros(n, dtype=float)
        if n == 0:
            return out

        use_parallel = (self.n_jobs is not None and self.n_jobs != 1 and n > 1)

        if use_parallel:
            start_method = "fork"
            try:
                ctx = get_context(start_method)
            except ValueError:
                ctx = get_context("spawn")

            with ctx.Pool(
                processes=(os.cpu_count() + self.n_jobs + 1 if self.n_jobs < 0 else self.n_jobs),
                initializer=_init_worker,
                initargs=(self,),
            ) as pool:
                for idx, score in pool.imap_unordered(_worker_similarity, enumerate(smiles), chunksize=16):
                    out[idx] = score
        else:
            for idx, smi in enumerate(smiles):
                try:
                    if not smi:
                        out[idx] = 0.0
                        continue
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        out[idx] = 0.0
                        continue
                    Chem.SanitizeMol(mol)
                    out[idx] = self._max_similarity_against_hits(mol)
                except Exception:
                    out[idx] = 0.0

        return out

    def _max_similarity_against_hits(self, query_mol: Chem.Mol) -> float:
        res = self.synthonspace.SubstructureSearch(query_mol)
        hit_mols = res.GetHitMolecules()

        if len(hit_mols) >= 1000:
            try:
                deep = self.synthonspace.SubstructureSearch(query_mol, self.params)
                hit_mols = deep.GetHitMolecules()
            except Exception:
                pass

        if not hit_mols:
            return 0.0

        qfp = self._fingerprint(query_mol)
        if qfp is None:
            return 0.0

        best = 0.0
        for hm in hit_mols:
            try:
                hfp = self._fingerprint(hm)
                if hfp is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(qfp, hfp)
                if sim > best:
                    best = sim
                    if best >= 1.0:
                        return 1.0
            except Exception:
                continue
        return float(best)

    def _fingerprint(self, mol: Chem.Mol):
        try:
            if self.mode == "morgan":
                if self._morgan_gen is None:
                    self._morgan_gen = rFG.GetMorganGenerator(
                        radius=self.radius,
                        fpSize=self.nbits,
                    )
                return self._morgan_gen.GetFingerprint(mol)
            else:
                if self._rdkit_gen is None:
                    self._rdkit_gen = rFG.GetRDKitFPGenerator(
                        fpSize=self.nbits,
                    )
                return self._rdkit_gen.GetFingerprint(mol)
        except Exception:
            return None