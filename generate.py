import os, random, torch, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch_geometric.data import Batch

from config import (
    DEVICE, MOLT5_MODEL, MOLT5_CKPT_BEST, REWARD_CKPT,
    GEN_N_CANDIDATES, GEN_TEMPERATURE, GEN_TOP_P, OUTPUTS_DIR
)
from reward_model import RewardModel, smiles_to_graph

RESULTS_CSV = os.path.join(OUTPUTS_DIR, "generated_molecules.csv")

def validate_smiles(smi: str):
    """Checks if a molecule is chemically 'sane'."""
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None: return None
        # Sanitize
        canon = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol = Chem.MolFromSmiles(canon)
        
        return {
            "smiles"  : canon,
            "qed"     : round(QED.qed(mol), 4),
            "logp"    : round(Descriptors.MolLogP(mol), 4),
            "tpsa"    : round(rdMolDescriptors.CalcTPSA(mol), 2),
            "mw"      : round(Descriptors.MolWt(mol), 2),
            "lipinski": int(Descriptors.MolWt(mol) <= 500),
        }
    except: return None

def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MOLT5_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(MOLT5_MODEL).to(DEVICE)
    if os.path.exists(MOLT5_CKPT_BEST):
        model.load_state_dict(torch.load(MOLT5_CKPT_BEST, map_location=DEVICE), strict=False)
    
    reward = RewardModel().to(DEVICE)
    if os.path.exists(REWARD_CKPT):
        reward.load_state_dict(torch.load(REWARD_CKPT, map_location=DEVICE))
    
    model.eval(); reward.eval()
    return model, tokenizer, reward


# ── Score molecules with GNN reward model ─────────────────────────────────────

def score_with_reward(reward_model, molecules: list,
                      target_qed=0.8, target_logp=2.5, target_tpsa=80.0, target_mw=350.0):
    """Add reward_score to each molecule dict."""
    scored = []
    for m in molecules:
        try:
            g = smiles_to_graph(m["smiles"])
            if g is None:
                continue
            batch = Batch.from_data_list([g]).to(DEVICE)
            with torch.no_grad():
                pred_qed, pred_logp, pred_tpsa = reward_model(batch)

            r = (
                0.5 * (1.0 - abs(pred_qed.item() - target_qed))  # Reward closeness to target QED
                - 0.2 * abs(pred_logp.item() - target_logp)      # Penalize LogP deviation
                - 0.001 * abs(pred_tpsa.item() * 200 - target_tpsa)  # Penalize TPSA deviation
                - 0.01 * abs(m["mw"] - target_mw) / 100          # Penalize MW deviation (normalized)
            )
            m["reward_score"] = round(r, 5)
            scored.append(m)
        except Exception:
            m["reward_score"] = 0.0
            scored.append(m)
    return scored


def generate_molecules(model, tokenizer, reward,
                       prompt: str,
                       n_generate: int = GEN_N_CANDIDATES,
                       target_qed:  float = 0.8,
                       target_logp: float = 2.5,
                       target_tpsa: float = 80.0,
                       target_mw:   float = 350.0):
    """Generate n_generate candidates, validate, score with GNN, return ranked list."""

    # ── GIBBERISH GUARD ─────────────────────────────────────────────────────
    if not re.search(r'\d', prompt):
        print("[gen] Gibberish prompt detected, skipping generation.")
        return []

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    ).to(DEVICE)

    all_smiles = []
    batch_size = 10                            # pure sampling batch
    n_batches  = max(1, n_generate // batch_size)

    print(f"[gen] Generating {n_batches * batch_size} candidates...")

    for i in tqdm(range(n_batches), desc="  Generating"):

        # ✅ FIX 1: unique random seed per batch — breaks determinism
        seed = random.randint(0, 99999) + i * 1000
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        with torch.no_grad():
            out = model.generate(
                input_ids            = enc["input_ids"],
                attention_mask       = enc["attention_mask"],
                max_new_tokens       = 128,
                num_beams            = 1,          # ✅ FIX 2: pure sampling, no beam convergence
                num_return_sequences = batch_size,
                do_sample            = True,        # ✅ stochastic sampling
                temperature          = GEN_TEMPERATURE,
                top_p                = GEN_TOP_P,
                top_k                = 50,          # ✅ FIX 3: nucleus vocab pruning
                repetition_penalty   = 1.3,         # ✅ FIX 4: stop repeating tokens
                early_stopping       = False,        # ✅ FIX 5: was killing sequence variety
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        all_smiles.extend(decoded)

    # ── Validate & deduplicate ─────────────────────────────────────────────────
    print(f"[gen] Validating {len(all_smiles)} candidates...")
    valid, seen = [], set()
    for smi in all_smiles:
        props = validate_smiles(smi)
        if props and props["smiles"] not in seen:
            seen.add(props["smiles"])
            valid.append(props)

    pct = len(valid) / max(1, len(all_smiles)) * 100
    print(f"[gen] Valid: {len(valid)}/{len(all_smiles)} ({pct:.1f}%)")

    if not valid:
        print("[gen] No valid molecules generated.")
        print("      Try: more finetune epochs, or lower --n value.")
        return []

    # ── Score with GNN ─────────────────────────────────────────────────────────
    scored = score_with_reward(reward, valid,
                                target_qed=target_qed,
                                target_logp=target_logp,
                                target_tpsa=target_tpsa,
                                target_mw=target_mw)

    scored.sort(key=lambda x: x["reward_score"], reverse=True)
    return scored

def save_results(molecules, path=RESULTS_CSV):
    pd.DataFrame(molecules).to_csv(path, index=False)