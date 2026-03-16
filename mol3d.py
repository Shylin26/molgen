from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional

def generate_3d_data(smiles: str) -> Optional[str]:
    """
    Takes a SMILES string, generates a 3D conformation using ETKDG, 
    and returns the SDF block representing the 3D structure.
    Returns None if the generation fails.
    """
    try:
        # Create a molecule with hydrogens added
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates using ETKDG method
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res == -1:
            print(f"[mol3d] Failed to embed molecule: {smiles}")
            # Fallback or just return None if embed fails
            return None
        
        # Optional: Optimize the structure
        AllChem.MMFFOptimizeMolecule(mol)
        
        # We can remove hydrogens again if we only want heavy atoms in the viewer, 
        # but 3dmol.js handles H's well. Let's keep them for structural accuracy,
        # or remove them based on preference.
        # mol = Chem.RemoveHs(mol)
        
        return Chem.MolToMolBlock(mol)
    except Exception as e:
        print(f"[mol3d] Error generating 3D data: {e}")
        return None
