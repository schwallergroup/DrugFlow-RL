from collections import Counter, deque
from typing import Dict, List, Tuple

import logging
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


class DiversityFilter:
    """
    Diversity filter for RL, in the spirit of REINVENT's ``DiversityFilter`` but
    generalized so it can also shape a *sampling policy* (we sample from the final
    checkpoint rather than harvesting molecules during RL).

    Generated molecules are bucketed by a key (an exact molecule or a scaffold),
    an occurrence count per bucket is maintained over a configurable ``scope``,
    and the reward of over-represented buckets is discounted so the policy is
    pushed to spread its mass instead of collapsing onto a few modes.

    Three orthogonal knobs:

    ``name`` — what defines a bucket key:
        - ``"NoFilter"``: pass-through, no penalty (default).
        - ``"IdenticalMolecule"``: key = canonical SMILES. Directly targets
          *uniqueness* (penalizes re-generating the same molecule).
        - ``"IdenticalMurckoScaffold"``: key = Bemis-Murcko scaffold.
        - ``"IdenticalTopologicalScaffold"``: key = generic (graph) scaffold.
        - ``"ScaffoldSimilarity"``: cluster scaffolds within ``minsimilarity``
          Tanimoto into a shared bucket.

    ``scope`` — the window over which occurrences are counted:
        - ``"global"``: whole run (REINVENT-like novelty pressure; the penalty
          landscape is non-stationary and saturates late in training).
        - ``"window"``: only the last ``window`` steps (stationary, sharp
          differential signal in the phase that determines the final checkpoint).
        - ``"batch"``: only the current step. NOTE: samples are already
          de-duplicated within a step, so ``batch`` scope is only meaningful for
          scaffold keys (many distinct molecules sharing a scaffold in one draw);
          molecule-level uniqueness pressure needs ``window`` or ``global``.

    ``penalty`` — how occurrence count maps to a reward multiplier:
        - ``"ramp"``: 1.0 up to ``tol``, linear down to 0 at ``buff`` (AHC-style,
          https://doi.org/10.1186/s13321-022-00646-z). Has a dead zone below
          ``tol`` and a hard 0 floor above ``buff``.
        - ``"binary"``: 1.0 below ``buff``, 0.0 at/above (REINVENT hard cutoff).
        - ``"continuous"``: ``1 / (1 + count / scale)``. Penalizes from the first
          repeat, decays smoothly, never hits exactly 0 -> always keeps an ordered
          gradient among over-used modes. Single parameter (``scale``).

    Counts are true occurrence counts (a ``Counter``, not a set), so re-generating
    the same molecule across steps *does* register and get penalized.
    """

    VARIANTS = {
        "NoFilter",
        "IdenticalMolecule",
        "IdenticalMurckoScaffold",
        "IdenticalTopologicalScaffold",
        "ScaffoldSimilarity",
    }
    SCOPES = {"batch", "window", "global"}
    PENALTIES = {"ramp", "binary", "continuous"}

    def __init__(
        self,
        name: str = "NoFilter",
        scope: str = "global",
        penalty: str = "ramp",
        minscore: float = 0.4,
        tol: int = 10,
        buff: int = 25,
        scale: float = 10.0,
        window: int = 10,
        minsimilarity: float = 0.4,
        radius: int = 2,
        nbits: int = 2048,
        **kwargs,
    ) -> None:
        assert name in self.VARIANTS, (
            f"Unknown diversity filter '{name}', choose from {sorted(self.VARIANTS)}"
        )
        assert scope in self.SCOPES, f"scope must be one of {sorted(self.SCOPES)}"
        assert penalty in self.PENALTIES, f"penalty must be one of {sorted(self.PENALTIES)}"
        assert tol < buff, "tol must be < buff"
        assert scale > 0, "scale must be > 0"
        assert window >= 1, "window must be >= 1"

        self.name = name
        self.scope = scope
        self.penalty = penalty
        self.minscore = minscore
        self.tol = tol
        self.buff = buff
        self.scale = scale
        self.window = window
        self.minsimilarity = minsimilarity
        self.radius = radius
        self.nbits = nbits

        # occurrence counts per bucket key, maintained according to `scope`
        self._global_counts: Counter = Counter()               # scope="global"
        self._window_counts: deque = deque(maxlen=window)       # scope="window"
        # representative fingerprint per bucket key (ScaffoldSimilarity only);
        # cluster identities persist regardless of scope, only counts are scoped
        self._bucket_fps: Dict[str, object] = {}

        extra = ""
        if name == "ScaffoldSimilarity":
            extra += f", minsimilarity={minsimilarity}"
        if scope == "window":
            extra += f", window={window}"
        if penalty == "continuous":
            extra += f", scale={scale}"
        else:
            extra += f", tol={tol}, buff={buff}"
        logger.info(
            f"DiversityFilter: name={name}, scope={scope}, penalty={penalty}, "
            f"minscore={minscore}{extra}"
        )

    @property
    def enabled(self) -> bool:
        return self.name != "NoFilter"

    def _bucket_key(self, smile: str) -> str:
        """Map a molecule SMILES to its bucket key (or '' on failure)."""
        if self.name == "IdenticalMolecule":
            # incoming SMILES are already canonical (Chem.MolToSmiles upstream)
            return smile

        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return ""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if self.name == "IdenticalTopologicalScaffold":
                scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
        except Exception:
            return ""

        if self.name != "ScaffoldSimilarity" or scaffold_smiles == "":
            return scaffold_smiles

        # cluster near-identical scaffolds into a shared bucket
        smol = Chem.MolFromSmiles(scaffold_smiles)
        if smol is None:
            return scaffold_smiles
        fp = AllChem.GetMorganFingerprintAsBitVect(smol, self.radius, self.nbits)
        for key, ref_fp in self._bucket_fps.items():
            if DataStructs.TanimotoSimilarity(fp, ref_fp) >= self.minsimilarity:
                return key
        self._bucket_fps[scaffold_smiles] = fp
        return scaffold_smiles

    def _penalty(self, count: int) -> float:
        """Reward multiplier in [0, 1] given the prior occurrence count."""
        if self.penalty == "continuous":
            return 1.0 / (1.0 + count / self.scale)
        if self.penalty == "binary":
            return 1.0 if count < self.buff else 0.0
        # ramp
        if count <= self.tol:
            return 1.0
        if count >= self.buff:
            return 0.0
        return (self.buff - count) / (self.buff - self.tol)

    def _prior_counts(self) -> Counter:
        """Occurrence counts from history, per the configured scope."""
        if self.scope == "global":
            return self._global_counts
        if self.scope == "window":
            agg: Counter = Counter()
            for step_counts in self._window_counts:
                agg.update(step_counts)
            return agg
        return Counter()  # batch: no history

    def _commit(self, batch_counts: Counter) -> None:
        """Fold this step's counts into the scoped memory."""
        if self.scope == "global":
            self._global_counts.update(batch_counts)
        elif self.scope == "window":
            self._window_counts.append(batch_counts)
        # batch: nothing persists

    def update(
        self,
        smiles: List[str],
        rewards: np.ndarray,
        step: int,
    ) -> Tuple[np.ndarray, Dict]:
        """Penalize rewards of over-represented buckets and update memory.

        Returns the (possibly penalized) rewards and a stats dict for logging.
        Molecules below ``minscore`` and invalid/empty SMILES are left untouched
        and do not count towards any bucket.
        """
        rewards = np.asarray(rewards, dtype=float)

        if not self.enabled:
            return rewards.copy(), {
                "penalized": 0,
                "buckets": 0,
                "mean_multiplier": 1.0,
            }

        prior = self._prior_counts()
        running: Counter = Counter()  # occurrences seen so far this step
        batch_counts: Counter = Counter()

        new_rewards = rewards.copy()
        n_penalized = 0
        multipliers = []
        for i, (smile, reward) in enumerate(zip(smiles, rewards)):
            if not smile or reward < self.minscore:
                multipliers.append(1.0)
                continue
            key = self._bucket_key(smile)
            if key == "":
                multipliers.append(1.0)
                continue

            # occurrences seen before this molecule = scoped history + earlier
            # occurrences within the current step
            count = prior.get(key, 0) + running[key]
            multiplier = self._penalty(count)
            if multiplier < 1.0:
                n_penalized += 1
            new_rewards[i] = reward * multiplier
            multipliers.append(multiplier)

            running[key] += 1
            batch_counts[key] += 1

        self._commit(batch_counts)

        # number of distinct keys currently tracked in scope (post-commit)
        if self.scope == "batch":
            n_buckets = len(batch_counts)
        elif self.scope == "window":
            n_buckets = len(self._prior_counts())
        else:
            n_buckets = len(self._global_counts)

        stats = {
            "penalized": n_penalized,
            "buckets": n_buckets,
            "mean_multiplier": float(np.mean(multipliers)) if multipliers else 1.0,
        }
        return new_rewards, stats
