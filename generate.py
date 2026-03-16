"""
generate.py — MolT5 + GNN Reward Model
Usage:
    python3 generate.py
    python3 generate.py --qed 0.8 --logp 2.5 --tpsa 80 --mw 350
"""

import os, argparse, re
import torch
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch_geometric.data import Batch

from config import (
    DEVICE, MOLT5_MODEL, MOLT5_CKPT_BEST, REWARD_CKPT,
    GEN_N_CANDIDATES, GEN_BEAMS, GEN_TEMPERATURE, GEN_TOP_P, OUTPUTS_DIR
)
from dataset import make_custom_prompt
from reward_model import RewardModel, smiles_to_graph
from fragment_filter import has_valid_fragment, fragment_score

RESULTS_CSV = os.path.join(OUTPUTS_DIR, "generated_molecules.csv")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Gibberish Guard ────────────────────────────────────────────────────────────

def is_valid_prompt(prompt: str) -> bool:
    """Returns False if prompt looks like gibberish."""
    if not prompt or len(prompt.strip()) < 5:
        return False
    # Must contain letters
    if not re.search("[a-zA-Z]", prompt):
        return False
    # Must contain at least one chemistry keyword
    keywords = [
        "drug", "molecule", "molecular", "weight", "solubility",
        "lipinski", "polar", "logp", "tpsa", "qed", "high", "low",
        "medium", "moderate", "generate", "small", "large",
        "satisfy", "violat", "lipophil", "antiviral", "cancer",
        "brain", "oral", "antibiotic", "covid", "binding"
    ]
    text = prompt.lower()
    return any(kw in text for kw in keywords)


# ── SMILES Validation ──────────────────────────────────────────────────────────

def validate_smiles(smi: str):
    """Return property dict if SMILES is chemically valid, else None."""
    try:
        smi = smi.strip()
        # Reject obvious non-SMILES (plain English words)
        if re.match(r'^[A-Z][a-z]+$', smi):
            return None
        if len(smi) < 2:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        canon = Chem.MolToSmiles(mol)
        mol2  = Chem.MolFromSmiles(canon)
        if mol2 is None:
            return None
            
        # Reject molecules that don't contain any valid drug-like subgraphs (hallucination prevention)
        if not has_valid_fragment(canon):
            return None

        qed_  = QED.qed(mol2)
        logp_ = Descriptors.MolLogP(mol2)
        tpsa_ = rdMolDescriptors.CalcTPSA(mol2)
        mw_   = Descriptors.MolWt(mol2)
        hbd   = rdMolDescriptors.CalcNumHBD(mol2)
        hba   = rdMolDescriptors.CalcNumHBA(mol2)
        lip   = int(mw_ <= 500 and logp_ <= 5 and hbd <= 5 and hba <= 10)
        frag  = fragment_score(canon)
        return {
            "smiles"  : canon,
            "qed"     : round(qed_,  4),
            "logp"    : round(logp_, 4),
            "tpsa"    : round(tpsa_, 2),
            "mw"      : round(mw_,   2),
            "lipinski": lip,
            "fragment_score": round(frag, 3),
        }
    except Exception:
        return None


# ── Load Models ────────────────────────────────────────────────────────────────

def load_models():
    print("[gen] Loading MolT5...")
    tokenizer = AutoTokenizer.from_pretrained(MOLT5_MODEL)
    model     = T5ForConditionalGeneration.from_pretrained(MOLT5_MODEL).to(DEVICE)
    if os.path.exists(MOLT5_CKPT_BEST):
        ckpt  = torch.load(MOLT5_CKPT_BEST, map_location=DEVICE)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        print(f"[gen] Loaded fine-tuned MolT5 from {MOLT5_CKPT_BEST}")
    else:
        print(f"[gen] WARNING: no checkpoint at {MOLT5_CKPT_BEST}")
    model.eval()

    print("[gen] Loading reward model...")
    reward = RewardModel().to(DEVICE)
    if os.path.exists(REWARD_CKPT):
        reward.load_state_dict(torch.load(REWARD_CKPT, map_location=DEVICE))
        print(f"[gen] Loaded reward model from {REWARD_CKPT}")
    else:
        print(f"[gen] WARNING: no reward model at {REWARD_CKPT}")
    reward.eval()

    return model, tokenizer, reward


# ── GNN Scoring ────────────────────────────────────────────────────────────────

def score_with_reward(reward_model, molecules: list,
                      target_qed=0.8, target_logp=2.5, target_tpsa=80.0):
    """Score each molecule with GNN and add reward_score field."""
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
                0.5   * pred_qed.item()
                - 0.2  * abs(pred_logp.item() - target_logp)
                - 0.001 * abs(pred_tpsa.item() * 200 - target_tpsa)
            )
            m["reward_score"] = round(r, 5)
            scored.append(m)
        except Exception:
            m["reward_score"] = 0.0
            scored.append(m)
    return scored


# ── Core Generation ────────────────────────────────────────────────────────────

def generate_molecules(model, tokenizer, reward,
                       prompt: str,
                       n_generate: int = GEN_N_CANDIDATES,
                       target_qed: float = 0.8,
                       target_logp: float = 2.5,
                       target_tpsa: float = 80.0,
                       target_mw: float = 350.0):
    """Generate candidates, validate, score with GNN, return ranked list."""

    # Gibberish guard
    if not is_valid_prompt(prompt):
        print("[gen] Invalid prompt detected. Rejected.")
        return []

    enc = tokenizer(
        prompt, return_tensors="pt",
        max_length=128, truncation=True, padding=True
    ).to(DEVICE)

    all_smiles = []
    n_batches  = max(1, n_generate // GEN_BEAMS)

    print(f"[gen] Generating {n_batches * GEN_BEAMS} candidates...")
    for _ in tqdm(range(n_batches), desc="  Generating"):
        with torch.no_grad():
            out = model.generate(
                input_ids            = enc["input_ids"],
                attention_mask       = enc["attention_mask"],
                max_new_tokens       = 128,
                num_beams            = GEN_BEAMS,
                num_return_sequences = GEN_BEAMS,
                do_sample            = True,
                temperature          = GEN_TEMPERATURE,
                top_p                = GEN_TOP_P,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        all_smiles.extend(decoded)

    # Validate
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
        return []

    # Score and rank
    scored = score_with_reward(
        reward, valid,
        target_qed=target_qed,
        target_logp=target_logp,
        target_tpsa=target_tpsa
    )
    scored.sort(key=lambda x: x["reward_score"], reverse=True)
    return scored


# ── Save Results ───────────────────────────────────────────────────────────────

def save_results(molecules: list, path=RESULTS_CSV) -> pd.DataFrame:
    df = pd.DataFrame(molecules)
    df.to_csv(path, index=False)
    print(f"[gen] Saved {len(df)} molecules → {path}")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate drug-like molecules")
    parser.add_argument("--qed",  type=float, default=0.80, help="Target QED (0-1)")
    parser.add_argument("--logp", type=float, default=2.50, help="Target LogP")
    parser.add_argument("--tpsa", type=float, default=80.0, help="Target TPSA")
    parser.add_argument("--mw",   type=float, default=350.0, help="Target MW")
    parser.add_argument("--n",    type=int,   default=GEN_N_CANDIDATES,
                        help="Candidates to generate")
    args = parser.parse_args()

    model, tokenizer, reward = load_models()

    prompt = make_custom_prompt(
        qed=args.qed, logp=args.logp,
        tpsa=args.tpsa, mw=args.mw
    )
    print(f"\n[gen] Prompt: {prompt}\n")

    molecules = generate_molecules(
        model, tokenizer, reward, prompt,
        n_generate  = args.n,
        target_qed  = args.qed,
        target_logp = args.logp,
        target_tpsa = args.tpsa,
        target_mw   = args.mw,
    )

    if molecules:
        df = save_results(molecules)
        print("\n── Top 5 Results ───────────────────────────────")
        for i, row in df.head(5).iterrows():
            lip = "✓" if row["lipinski"] else "✗"
            print(f"\n  {i+1}. {row['smiles']}")
            print(f"     QED={row['qed']:.3f}  LogP={row['logp']:.2f}  "
                  f"TPSA={row['tpsa']:.1f}  MW={row['mw']:.1f}  "
                  f"Lipinski={lip}  FragScore={row.get('fragment_score', 0):.3f}  Score={row['reward_score']:.4f}")
        print(f"\n[gen] Full results → {RESULTS_CSV}")
    else:
        print("[gen] Generation failed. Check checkpoints and try again.")