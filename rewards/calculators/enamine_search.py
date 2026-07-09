import os
import sys
import time
import requests
import numpy as np
from typing import Dict, List, Optional, Iterable, Any
from rdkit import Chem
import logging

from rewards.calculators.base import Calculator

# Constants adapted from scripts/analysis/enamine_search.py
SERVER_URL = "https://real.enamine.net"
EXACT_STEREO_URL = f"{SERVER_URL}/api/v1/space/real/search-structure/batch/exact-stereo"
REACTION_TYPES = [0, 1050]
REQUEST_TIMEOUT = 120


class EnamineSearchCalc(Calculator):
    def __init__(
        self,
        task: str = "enamine_exact_match",
        api_key: Optional[str] = None,
        batch_size: int = 50,
        wait_seconds: int = 300,
        **kwargs: Any,
    ) -> None:
        """Initialise the Enamine REAL calculator.

        Args:
            task: The name of the task.
            api_key: Enamine REAL API key. If None, tries to read 'REAL_API_KEY' env var.
            batch_size: Number of SMILES to submit per request.
            wait_seconds: Seconds to sleep when hitting the API rate limit.
        """
        super().__init__(task)
        self.api_key = api_key or os.getenv("REAL_API_KEY")
        
        if not self.api_key:
            raise ValueError("REAL_API_KEY not found in arguments or environment. ")

        self.batch_size = batch_size
        self.wait_seconds = wait_seconds

    def calc(self, sample_dict: Dict[str, Any], label: str = "tmp") -> np.ndarray:
        """Calculate exact-match indicators using Enamine REAL API.

        Args:
            sample_dict: Dictionary containing 'ligands' -> 'mols' (list of RDKit mols).
            label: Unused label.

        Returns:
            Numpy array of floats (1.0 for found, 0.0 for not found).
        """
        if not self.api_key:
             raise ValueError("Missing API key for EnamineSearchCalc.")

        mols: List[Chem.Mol] = sample_dict.get("ligands", {}).get("mols", [])
        if not mols:
            return np.array([], dtype=float)

        # Convert mols to canonical SMILES to ensure consistency
        smiles_list: List[str] = []
        valid_indices: List[int] = []
        
        for i, mol in enumerate(mols):
            if mol:
                try:
                    smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                    smiles_list.append(smi)
                    valid_indices.append(i)
                except Exception:
                    pass
        
        # If no valid SMILES, return all zeros
        if not smiles_list:
            return np.zeros(len(mols), dtype=float)

        # Perform the search
        results_map = self._search_enamine(smiles_list)

        # Map results back to the original order
        rewards = np.zeros(len(mols), dtype=float)
        for idx, smi in zip(valid_indices, smiles_list):
            if results_map.get(smi, False):
                rewards[idx] = 1.0

        return rewards

    def _chunked(self, items: List[str], size: int) -> Iterable[List[str]]:
        for start in range(0, len(items), size):
            yield items[start : start + size]

    def _query_batch(self, smiles_batch: List[str]) -> Dict[str, bool]:
        """Query a single batch of SMILES against the API."""
        pending = list(smiles_batch)
        results: Dict[str, bool] = {}

        while pending:
            try:
                response = requests.post(
                    EXACT_STEREO_URL,
                    json={"smiles_list": pending, "reaction_types": REACTION_TYPES},
                    headers={
                        "Content-Type": "application/json",
                        "X-API-KEY": self.api_key, # type: ignore
                    },
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.RequestException as e:
                logging.warning(f"Request failed: {e}")
                # Fail open or closed? Here we fail closed for the pending batch
                for s in pending:
                    results[s] = False
                return results

            try:
                payload = response.json()
            except ValueError:
                logging.warning(f"Non-JSON response: {response.text}")
                for s in pending:
                    results[s] = False
                return results

            if isinstance(payload, list):
                # Successful response
                for entry in payload:
                    query_smiles = entry.get("query_smiles")
                    if query_smiles:
                        results[query_smiles] = entry.get("vSynt") is not None
                return results

            if isinstance(payload, dict):
                detail = payload.get("detail", "")
                
                # Handle invalid SMILES error
                if detail.startswith("Invalid input smiles"):
                    invalid = detail.split(":", 1)[-1].strip()
                    # Mark invalid as False and remove from pending
                    results[invalid] = False
                    pending = [s for s in pending if s != invalid]
                    if not pending:
                        return results
                    continue

                # Handle rate limit
                if detail.startswith("Effective limit exceeded"):
                    logging.warning(
                        f"Rate limit exceeded, sleeping for {self.wait_seconds} seconds...",
                    )
                    time.sleep(self.wait_seconds)
                    continue

            # Fallback for unexpected responses
            logging.warning(f"Unexpected response payload: {payload}")
            for s in pending:
                results[s] = False
            return results

        return results

    def _search_enamine(self, smiles: List[str]) -> Dict[str, bool]:
        """Orchestrate the batch search."""
        summary: Dict[str, bool] = {}
        # Use a set of unique SMILES to avoid redundant queries in this batch
        unique_smiles = list(set(smiles))
        
        for batch in self._chunked(unique_smiles, self.batch_size):
            batch_results = self._query_batch(batch)
            summary.update(batch_results)
            
        return summary
