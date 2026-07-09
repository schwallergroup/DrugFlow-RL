import logging
import tempfile
import traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from src.sbdd_metrics.metrics import FullEvaluator, MedChemEvaluator, UniquenessEvaluator
from src.constants import atom_encoder, bond_encoder
from src.data.data_utils import process_raw_pair, prepare_ligand
from src.size_predictor.size_model import SizeModel
from Bio.PDB import PDBParser

ALL_EVALUATORS = [
    'representation', 'mol_props', 'posebusters', 'geometry', 'energy',
    'interactions', 'gnina', 'medchem', 'clashes', 'ring_count',
    'chembl_ring_systems', 'reos', 'strain', 'fingerprint_novelty',
    'aizynth', 'ff_relaxation',
]

def get_evaluators(metrics):
    """Initialize evaluators based on requested metrics."""
    exclude_evaluators = [e for e in ALL_EVALUATORS if e not in metrics]
    metric_evaluator = FullEvaluator(exclude_evaluators=exclude_evaluators, pb_conf='mol')
    filter_evaluator = MedChemEvaluator(connectivity_threshold=1.0)
    return metric_evaluator, filter_evaluator

def compute_internal_diversity(mols):
    """Compute diversity and return (diversity, n_failed_fingerprints)."""
    if len(mols) < 2:
        n_failed = max(0, len(mols) - 0)
        return 0.0, n_failed

    fps = []
    for m in mols:
        if m is None:
            continue
        m2 = Chem.Mol(m)
        try:
            Chem.SanitizeMol(m2)
        except Exception:
            try:
                smi = Chem.MolToSmiles(m, isomericSmiles=True)
                m2 = Chem.MolFromSmiles(smi)
                if m2 is None: continue
            except Exception:
                continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(m2, radius=2, nBits=2048)
            fps.append(fp)
        except Exception as e:
            logging.warning(f"Failed to compute fingerprint: {e}")
            continue

    n_failed = len(mols) - len(fps)
    if len(fps) < 2:
        return 0.0, n_failed

    similarities = []
    for i in range(len(fps) - 1):
        tail = fps[i+1:]
        if not tail: continue
        bulk = DataStructs.BulkTanimotoSimilarity(fps[i], tail)
        similarities.extend(list(bulk))

    if not similarities:
        return 0.0, n_failed

    avg_sim = float(np.mean(similarities))
    return 1.0 - avg_sim, n_failed

def calculate_molecule_statistics(all_mols, filter_evaluator):
    """Calculate validity, connectedness, uniqueness, and internal diversity."""
    stats = {}
    valid_mols = []

    # Validity check
    for mol in all_mols:
        try:
            if filter_evaluator.is_valid(mol):
                valid_mols.append(mol)
        except Exception:
            pass

    stats['validity'] = (len(valid_mols) / len(all_mols)) if all_mols else 0.0

    # Uniqueness
    valid_smiles = []
    for mol in valid_mols:
        try:
            valid_smiles.append(Chem.MolToSmiles(mol, kekuleSmiles=True))
        except Exception:
            continue

    try:
        uniq_eval = UniquenessEvaluator()
        uniq_res = uniq_eval(valid_smiles) if valid_smiles else {'uniqueness': 0.0}
        stats['uniqueness'] = float(uniq_res.get('uniqueness', 0.0))
    except Exception:
        stats['uniqueness'] = 0.0

    # Internal diversity
    stats['internal_diversity'], stats['n_diversity_failed'] = compute_internal_diversity(valid_mols)

    return stats, valid_mols

def get_metric_score(mols, pockets, metric_evaluator):
    """Calculate full metric scores for generated molecules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        protein_ps = [Path(tmpdir, f'protein_{i}.pdb') for i in range(len(pockets))]
        for receptor, pocket in zip(protein_ps, pockets):
            Chem.MolToPDBFile(pocket, str(receptor))

        results = []
        for mol, protein_p in zip(mols, protein_ps):
            if mol is not None:
                results.append(metric_evaluator(mol, str(protein_p)))
            else:
                results.append({
                    'preselected': False,
                    'representation.smiles': '',
                    'medchem.valid': False,
                    'medchem.connected': False
                })
        return results

def prepare_input_data(protein_path, ligand_path, pocket_distance_cutoff, model, starting_point=None):
    """Prepare ligand and pocket data for the model."""
    pdb_model = PDBParser(QUIET=True).get_structure('', protein_path)[0]
    rdmol = Chem.SDMolSupplier(str(ligand_path), sanitize=False)[0]

    ligand, pocket = process_raw_pair(
        pdb_model, rdmol,
        dist_cutoff=pocket_distance_cutoff,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True,
        nma_input=protein_path if model.dynamics.add_nma_feat else None,
    )

    if starting_point is not None:
        starting_rdmol = Chem.SDMolSupplier(str(starting_point), sanitize=False)[0]
        ligand = prepare_ligand(starting_rdmol, atom_encoder, bond_encoder)
        model.masked_modeling = True

    if 'size' not in ligand:
        ligand['size'] = ligand['mask'].sum().item()

    ligand['name'] = 'ligand'
    return ligand, pocket

def get_size_model(molecule_size, size_checkpoint, device, batch_size):
    """Initialize size prediction model or parse fixed size."""
    size_model = None
    parsed_size = None

    if molecule_size is not None:
        if molecule_size.isdigit():
            parsed_size = int(molecule_size)
            logging.info(f'Will generate molecules of size {parsed_size}')
        else:
            boundaries = [x.strip() for x in molecule_size.split(',')]
            if len(boundaries) == 2 and boundaries[0].isdigit() and boundaries[1].isdigit():
                left, right = int(boundaries[0]), int(boundaries[1])
                parsed_size = f"uniform_{left}_{right}"
                logging.info(f'Will generate linkers with numbers of atoms sampled from U({left}, {right})')
            else:
                parsed_size = "nn_prediction"
                size_model = SizeModel.load_from_checkpoint(size_checkpoint, map_location=device)
                size_model.batch_size = size_model.eval_batch_size = batch_size
                size_model.eval().to(device)
                logging.info(f'Loaded size prediction model {size_checkpoint}')

    return size_model, parsed_size

