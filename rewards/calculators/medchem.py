import os 
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED

from rewards.calculators.base import Calculator
from src.sbdd_metrics.sascorer import calculateScore


class MedChemCalc(Calculator):
    def __init__(self, task: str = 'qed', connectivity_threshold: float = 1.0) -> None:
        super().__init__(task)
        self.connectivity_threshold = connectivity_threshold


    def calc(self, sample_dict: dict, label: str = 'tmp') -> np.ndarray[float]:
        samples = sample_dict['ligands']['mols']

        # check if the samples are valid if not replace with None
        samples = [mol if self.is_valid(mol) else None for mol in samples]

        # calculate the task
        if self.task == 'qed':
            results = np.array([self.calculate_qed(mol) for mol in samples])
        elif self.task == 'sa':
            results = np.array([self.calculate_sa(mol) for mol in samples])
        elif self.task == 'logp':
            results = np.array([self.calculate_logp(mol) for mol in samples])
        elif self.task == 'lipinski':
            results = np.array([self.calculate_lipinski(mol) for mol in samples])
        elif self.task == 'molecule_size':
            results = np.array([self.calculate_molecule_size(mol) for mol in samples])
        elif self.task == 'rotatable_bonds':
            results = np.array([self.calculate_rotatable_bonds(mol) for mol in samples])
        elif self.task == 'chiral_centers':
            results = np.array([self.calculate_chiral_centers(mol) for mol in samples])
        else:
            raise ValueError(f"Unknown task: {self.task} for MedChemCalc")

        return results

    @staticmethod
    def is_valid(rdmol):
        if rdmol is None or rdmol.GetNumAtoms() < 1:
            return False

        _mol = Chem.Mol(rdmol)
        try:
            Chem.SanitizeMol(_mol)
        except ValueError:
            return False

        return True

    def is_connected(self, rdmol):
        if rdmol.GetNumAtoms() < 1:
            return False

        try:
            mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True)
            largest_frag = max(mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms())
            return largest_frag.GetNumAtoms() / rdmol.GetNumAtoms() >= self.connectivity_threshold
        except:
            return False
    
    @staticmethod
    def calculate_qed(rdmol):
        try: return QED.qed(rdmol)
        except: return np.nan

    @staticmethod
    def calculate_sa(rdmol):
        try: return calculateScore(rdmol)
        except: return np.nan

    @staticmethod
    def calculate_logp(rdmol):
        try: return Crippen.MolLogP(rdmol)
        except: return np.nan

    @staticmethod
    def calculate_lipinski(rdmol):
        try:
            rule_1 = Descriptors.ExactMolWt(rdmol) < 500
            rule_2 = Lipinski.NumHDonors(rdmol) <= 5
            rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
            rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
            rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
            return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
        except: return np.nan
        
    @staticmethod
    def calculate_molecule_size(rdmol):
        try: return rdmol.GetNumAtoms()
        except: return np.nan
        
    @staticmethod
    def calculate_rotatable_bonds(rdmol):
        try: return Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol)
        except: return np.nan

    @staticmethod 
    def calculate_chiral_centers(rdmol):
        try:
            chiral_centers = Chem.FindMolChiralCenters(
                rdmol, 
                includeUnassigned=True, 
                includeCIP=False, 
                useLegacyImplementation=False
            )
            return len(chiral_centers)
        except: return np.nan


if __name__ == "__main__":
    task = "qed"
    medchem = MedChemCalc(task)
    
    smiless = ["CC1=CC2(C3=CN(C4=CC=C(Cl)C=C4)N=C3)CCC(C(=O)O)(CC1)CC2",
    "CC1=NC(C2=CC(C=CC3CCCCN3)=CC=C2)=C(C)N2C(N)=NN=C12",
    "CC1=CC(C2=NC3CC(=NCC4(C3)C3=C(C=CC=C3)CC4C)C2)=CN=C1",
    "CN1C2=C(C=CC=C2Cl)CC12CCC1=CN=C(NC(=O)CC3CCCC3)C=C1C2",
    "C=C1SC2C(=O)N(C3=COC(C#CC)=C3)CC2N2N=CC=C12",
    "CCCCC(=O)NCCCC(=O)N1CC2=C(C1)C(C)=CC(O)=C2",
    "CC1=CC(C2=CC=C(S(=O)(=O)NC3=C(C#N)C=CC=C3)C(F)=C2)=CC=N1",
    "CC1CN=S(=O)(C2=CC=C(NC3=NC=NC=N3)C=C2)C2=C(C1)SC=C2",
    "CCNC(=O)C1=CC=C(N2CC3=C(C=CC=C3C)CC2=O)C=C1",
    "CCC(=O)NC(=O)C1=CC=C(C2(C)C3=C(C=CC=C3)OC2C)C=C1",
    "CCCNC1=NC=C(C2(F)CC3=C4C(=CC=C3)CCN4C2=O)C=N1",
    "CNC(=O)C1=CC=C(S(=O)(=O)N2CCCC3=CC=CC=C32)C=C1",
    "CC1=CC(C)=CC(S(=O)(=O)C2=CC=C(C(=O)NC3CC3)C=C2)=C1",
    "CC1=CC=C(C2=NOC(CN3CC4C(C)CCC(C)C4C3=O)=C2)N1",
    "CC12CCC(C3=CC(C4=NC=CN=N4)=CC=C3)=CC(C1)C(CO)C2",
    "CCNC(=O)C1=CC=C(C23CCC(=O)CC2C(C)CC3)C=C1",
    "CS(=O)C1CCC(S(=O)NS(=O)(=O)CCC(C)(C)C)C1",
    "CCCC(CC(C)C)S(=O)(=O)NC1=CC=C(S(C)(=O)=O)C=C1",
    "CC(C)C1=CC=CC=C1S(=O)(=O)CCC1CCC(S(C)(=O)=O)C1",
    "CS(=O)C1=CC=CC(CC(F)CS(=O)(=O)C2CCCC3CCCC32)=C1",
    "CCC(=O)CCCCCCCS(=O)(=O)C1CCCS(=O)(=O)C2CC(F)C(C)C(C2)S(=O)(=O)C1",
    "CC(C)CCCC(=O)NC(C)CCCS(=O)(=O)CCCS(=O)(=O)CCC(C)C",
    "CCS(=O)(=O)C1CCN(S(=O)(=O)CS(=O)(=O)CC(CCC(C)(C)C)C(O)CCC(C)C)C1",
    "CC(CCCS(=O)(=O)CCC1CS(=O)(=O)CC2CCC(C)(C(C)C)C21)CC(=O)CC1CC1"]
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiless]
    sample_dict = {'ligands': {'mols': mols}}
    print(medchem.calc(sample_dict, label="qed"))
