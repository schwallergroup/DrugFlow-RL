import os

from rdkit import Chem

def save_smiles(molecules, save_dir, filename, step):
    """Save molecules as SMILES strings to a text file."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    with open(save_path, 'a') as f:
        f.write(f"# Step {step}\n")
        for i, mol in enumerate(molecules):
            if mol is not None:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except:
                    pass
                try:
                    smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                    f.write(f"{smiles}\n")
                except Exception:
                    pass
            else:
                print(f"Failed molecule {i}")
    
    return