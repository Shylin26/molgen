import os, requests
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, FilterCatalog
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from config import ZINC_CSV, PROCESSED_CSV, MAX_MOLS

ZINC_URL = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"

def download_zinc(path=ZINC_CSV):
    if os.path.exists(path):
        print(f"[data] {path} already exists.")
        return
    print("[data] Downloading ZINC250k...")
    r = requests.get(ZINC_URL, timeout=180)
    r.raise_for_status()
    open(path, "wb").write(r.content)
    print("[data] Done.")

def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        canonical = Chem.MolToSmiles(mol)
        qed  = float(QED.qed(mol))
        logp = float(Descriptors.MolLogP(mol))
        mw   = float(Descriptors.MolWt(mol))
        tpsa = float(CalcTPSA(mol))
        hbd  = int(Descriptors.NumHDonors(mol))
        hba  = int(Descriptors.NumHAcceptors(mol))
        lip  = int(mw<=500 and logp<=5 and hbd<=5 and hba<=10)
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        cat = FilterCatalog.FilterCatalog(params)
        pains = int(cat.HasMatch(mol))
        return dict(smiles=canonical, qed=round(qed,3), logp=round(logp,3),
                    mw=round(mw,2), tpsa=round(tpsa,2), hbd=hbd, hba=hba,
                    lipinski=lip, pains=pains)
    except Exception:
        return None

def _bqed(v):  return "high" if v>=0.7 else ("medium" if v>=0.4 else "low")
def _blogp(v): return "low"  if v<=2   else ("moderate" if v<=4  else "high")
def _btpsa(v): return "low"  if v<=60  else ("moderate" if v<=120 else "high")

def make_prompt(row):
    return (
        f"Generate a molecule with {_bqed(row['qed'])} drug-likeness, "
        f"{_blogp(row['logp'])} lipophilicity, "
        f"{_btpsa(row['tpsa'])} polarity, "
        f"molecular weight {int(row['mw'])}, "
        f"{'satisfying' if row['lipinski'] else 'violating'} Lipinski rules."
    )

def make_custom_prompt(qed=0.8, logp=2.5, tpsa=80.0, mw=350, lipinski=True):
    # Guard against invalid / non-numeric input (e.g. "ggg") that could
    # still be converted into a prompt and then hallucinated by the model.
    for name, value in dict(qed=qed, logp=logp, tpsa=tpsa, mw=mw).items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Invalid numeric value for {name}: {value!r}")
        if value != value:  # NaN check
            raise ValueError(f"Invalid numeric value for {name}: NaN")

    row = dict(qed=qed, logp=logp, tpsa=tpsa, mw=mw, lipinski=int(lipinski))
    return make_prompt(row)

def build_dataset(csv_path=ZINC_CSV, cache=PROCESSED_CSV, max_mols=MAX_MOLS):
    if os.path.exists(cache):
        print(f"[data] Loading cache: {cache}")
        return pd.read_csv(cache)
    download_zinc(csv_path)
    raw = pd.read_csv(csv_path)
    col = "smiles" if "smiles" in raw.columns else raw.columns[0]
    smiles_list = raw[col].dropna().tolist()[:max_mols]
    rows = []
    for smi in tqdm(smiles_list, desc="Computing properties"):
        p = compute_properties(smi)
        if p:
            rows.append(p)
    df = pd.DataFrame(rows)
    df["prompt"] = df.apply(make_prompt, axis=1)
    df.to_csv(cache, index=False)
    print(f"[data] Saved {len(df)} molecules.")
    return df

class MoleculeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df  = df.reset_index(drop=True)
        self.tok = tokenizer
        self.mlen = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tok(str(row["prompt"]), max_length=self.mlen,
                       padding="max_length", truncation=True, return_tensors="pt")
        tgt = self.tok(text_target=str(row["smiles"]), max_length=self.mlen,
                       padding="max_length", truncation=True, return_tensors="pt")
        labels = tgt["input_ids"].squeeze().clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids"      : enc["input_ids"].squeeze(),
            "attention_mask" : enc["attention_mask"].squeeze(),
            "labels"         : labels,
        }

if __name__ == "__main__":
    df = build_dataset()
    print(f"Total: {len(df)} | Mean QED: {df['qed'].mean():.3f}")
    print(f"Sample: {df['prompt'].iloc[0]}")
