import torch, os

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# Paths
ZINC_CSV        = "zinc250k.csv"
PROCESSED_CSV   = "zinc_processed.csv"
REWARD_CKPT     = "checkpoints/reward_model.pt"
MOLT5_CKPT_BEST = "checkpoints/molt5_best.pt"
MOLT5_CKPT_LAST = "checkpoints/molt5_last.pt"
TRAIN_LOG       = "checkpoints/train_log.json"
OUTPUTS_DIR     = "outputs"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Dataset
MAX_MOLS     = 50000
MAX_SEQ_LEN  = 128

# Reward model
REWARD_HIDDEN  = 256
REWARD_LAYERS  = 4
REWARD_EPOCHS  = 10
REWARD_BATCH   = 32  # Reduced from 64
REWARD_LR      = 1e-3

# MolT5
MOLT5_MODEL  = "laituan245/molt5-small"
MOLT5_EPOCHS = 5
MOLT5_BATCH  = 8   # Reduced from 16
MOLT5_LR     = 3e-4
GRAD_CLIP    = 1.0

# Generation
GEN_N_CANDIDATES = 50
GEN_BEAMS        = 1
GEN_TEMPERATURE  = 1.1
GEN_TOP_P        = 0.95

# Atom/bond vocab
ATOM_TYPES = ['C','N','O','F','P','S','Cl','Br','I','other']
NODE_DIM   = len(ATOM_TYPES) + 3

from rdkit.Chem import rdchem
BOND_TYPES = [
    rdchem.BondType.SINGLE,       # <-- removed stray "bn" here
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
EDGE_DIM = len(BOND_TYPES)
