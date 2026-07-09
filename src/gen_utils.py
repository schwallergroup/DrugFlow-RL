# This script adapted from generate.py

import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolops import GetMolFrags


SAMPLING_FILTERS = {
    'pre_filter_global': 
    (
        lambda row: (
            (row['medchem.valid'] == 1) &
            (row['medchem.connected'] == 1) &
            (row['novel'] == 1)
        ),
        ['representation','medchem', 'novel']
    ),
    'filter_final': 
    (
        lambda row: (
            (row['medchem.valid'] == 1) &
            (row['medchem.connected'] == 1) &
            (row['posebusters.all'] == 1) &
            (row['reos.PAINS'] == 1) &
            (row['reos.Dundee'] == 1) &
            (row['chembl_ring_systems.min_ring_freq_gt0_'] == 1) &
            (row['clashes.passed_clash_score_between'] == 1) &
            (row['clashes.passed_clash_score_ligands'] == 1) &
            (row['novel'] == 1) &
            (1. <= row['medchem.logp'] <= 4.) &
            (row['medchem.sa'] <= 5.)
        ),
        ['representation','medchem', 'posebusters', 'reos', 'chembl_ring_systems', 'clashes', 'novel']
    ),
    'filter_final2': 
    (
        lambda row: (
            (row['medchem.valid'] == 1) &
            (row['medchem.connected'] == 1) &
            (row['posebusters.all'] == 1) &
            (row['clashes.passed_clash_score_between'] == 1) &
            (row['clashes.passed_clash_score_ligands'] == 1) &
            (row['novel'] == 1)
        ),
        ['representation','medchem', 'posebusters', 'clashes', 'novel']
    ),
    'filter_fragments_reos': 
    (
        lambda row: (
            (row['medchem.valid'] == 1) &
            (row['medchem.connected'] == 1) &
            (row['reos.PAINS'] == 1) &
            (row['reos.Dundee'] == 1) &
            ((row['posebusters.all'] == 1) | (row['medchem.size'] <= 2.0)) &
            (row['chembl_ring_systems.min_ring_freq_gt10_'] == 1) &
            (row['clashes.passed_clash_score_between'] == 1) &
            (row['clashes.passed_clash_score_ligands'] == 1)
        ),
        ['representation', 'medchem', 'reos', 'posebusters', 'chembl_ring_systems', 'clashes']
    ),
    'filter_fragments': 
    (
        lambda row: (
            (row['medchem.valid'] == 1) &
            (row['medchem.connected'] == 1) &
            ((row['posebusters.all'] == 1) | (row['medchem.size'] <= 2.0)) &
            (row['clashes.passed_clash_score_between'] == 1) &
            (row['clashes.passed_clash_score_ligands'] == 1)
        ),
        ['representation', 'medchem', 'posebusters', 'clashes']
    ),
    'valid_and_connected':
    (
        lambda row: (
            (row['medchem.valid'] == 1) &
            (row['medchem.connected'] == 1)
        ),
        ['medchem']
    )
}

def aggregate_metrics(table, exclude=None):
    exclude = [x.strip() for x in (exclude or '').split(',')] if exclude else []
    
    for agg_col in ('posebusters', 'reos', 'chembl_ring_systems'):
        cols_to_aggregate = [c for c in table.columns if \
                                c.startswith(agg_col) and \
                                c != agg_col and \
                                c not in exclude and \
                                (agg_col != 'chembl_ring_systems' or not c.endswith('smi'))]
        table[agg_col] = table[cols_to_aggregate].fillna(0).astype(float).sum(axis=1) / len(cols_to_aggregate)
    
    return table

def preselect_mols(rdmols, smiles=None):
    smiles = set() if smiles is None else set(smiles)
    # Sanitizing molecules
    preselected = [False for _ in rdmols]
    processed_mols = [mol for mol in rdmols]
    for i, mol in enumerate(rdmols):
        if mol is None: continue
        newmol = Chem.Mol(mol)
        if Chem.SanitizeMol(newmol, catchErrors=True) != 0:
            logging.debug(f'Failed sanitizing molecule {i}')
        else:
            processed_mols[i] = newmol
            connected = Chem.GetMolFrags(newmol, asMols=True)
            if len(connected) > 1: continue
            smi = Chem.MolToSmiles(newmol)
            if smi in smiles: continue
            preselected[i] = True
            smiles.add(smi)
    return preselected, processed_mols

def filter_mols(rdmols, protein_ps, evaluator, 
                smiles=None, 
                criterion='filter_final2', 
                return_table=False, 
                preselect_unique=False):
    smiles = set() if smiles is None else set(smiles)
    if not isinstance(protein_ps, list):
        protein_ps = [protein_ps] * len(rdmols)

    if preselect_unique:
        preselection, rdmols = preselect_mols(rdmols, smiles)
    else:
        preselection = [True] * len(rdmols)

    results = [evaluator(mol, str(protein_p)) \
                    if preselected and mol is not None else 
               {'preselected': False,
                'representation.smiles': '',
                'medchem.valid': False, 
                'medchem.connected': False}
                        for mol, protein_p, preselected in 
                            zip(rdmols, protein_ps, preselection)]
    table = pd.DataFrame(results)
    table['preselected'] = preselection
    table['mol'] = rdmols
    if not 'representation.smiles' in table.columns:
        table['representation.smiles'] = ''
    table['novel'] = ~table['representation.smiles'].isin(smiles)
    table = aggregate_metrics(table)

    filter_func = None
    cols_in_filter_func = []
    if isinstance(criterion, str) and criterion in SAMPLING_FILTERS:
        filter_func = SAMPLING_FILTERS[criterion][0]
        cols_in_filter_func = SAMPLING_FILTERS[criterion][1]
    elif callable(criterion):
        filter_func = criterion
    else:
        raise ValueError(f'Unknown filter criterion: {criterion}')
    for filter_name in cols_in_filter_func:
        if filter_name not in table.columns:
            logging.debug(f'Adding missing column {filter_name} to the table.')
            table[filter_name] = True
    table['passed_filters'] = table.apply(filter_func, axis=1)

    if return_table:
        return table
    
    res_mols = []
    selected_indices = []
    for i, (_, row) in enumerate(table.iterrows()):
        if row['passed_filters']:
            mol = rdmols[i]
            smi = Chem.MolToSmiles(mol)
            if smi not in smiles:
                smiles.add(smi)
                for column in row.index:
                    if column.startswith(('medchem', 'reos','representation')):
                        mol.SetProp(column, str(row[column]))
                res_mols.append(mol)
                selected_indices.append(i)
    return res_mols, smiles, selected_indices