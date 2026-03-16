"""
fragment_filter.py — Subgraph-based hallucination prevention
Validates generated molecules contain real drug-like fragments
"""
from rdkit import Chem
from rdkit.Chem import BRICS, FragmentMatcher

# Common drug-like fragments (subgraphs)
VALID_FRAGMENTS = [
    "c1ccccc1",          # benzene ring
    "c1ccncc1",          # pyridine
    "C1CCNCC1",          # piperidine
    "c1ccoc1",           # furan
    "C1CCOC1",           # THF
    "c1ccc(cc1)",        # para-substituted benzene
    "C(=O)N",            # amide bond
    "C(=O)O",            # carboxylic acid/ester
    "c1cnc2ccccc2n1",    # benzimidazole
    "C1COCCN1",          # morpholine
    "c1cc[nH]c1",        # pyrrole
    "C1CCNC1",           # pyrrolidine
]

def has_valid_fragment(smiles: str) -> bool:
    """
    Check if molecule contains at least one known drug-like subgraph.
    Pure English words like 'These' contain no valid chemical fragments.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    for frag_smiles in VALID_FRAGMENTS:
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag is None:
            continue
        if mol.HasSubstructMatch(frag):
            return True
    
    return False

def fragment_score(smiles: str) -> float:
    """
    Score how many known drug fragments a molecule contains.
    Higher = more drug-like substructure composition.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    
    count = 0
    for frag_smiles in VALID_FRAGMENTS:
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag and mol.HasSubstructMatch(frag):
            count += 1
    
    return count / len(VALID_FRAGMENTS)

def brics_decompose(smiles: str):
    """
    Decompose molecule into BRICS fragments.
    BRICS = Breaking Retrosynthetically Interesting Chemical Substructures
    Used by pharma to identify synthetically accessible building blocks.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    frags = BRICS.BRICSDecompose(mol)
    return list(frags)

if __name__ == "__main__":
    # Test on your generated molecules
    test_smiles = [
        "COc1ccc(NC(=O)[C@@H]2CCCN(C(=O)c3ccc(C)o3)C2)cc1C",  # valid
        "These",           # hallucination
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "Inhibits",        # hallucination
    ]
    
    print("Fragment validation test:")
    for smi in test_smiles:
        valid = has_valid_fragment(smi)
        score = fragment_score(smi) if valid else 0.0
        frags = brics_decompose(smi) if valid else []
        print(f"\n  SMILES: {smi}")
        print(f"  Has valid fragment: {valid}")
        print(f"  Fragment score: {score:.3f}")
        if frags:
            print(f"  BRICS fragments: {list(frags)[:3]}")
