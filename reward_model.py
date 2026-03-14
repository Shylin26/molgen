import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import pandas as pd
from tqdm import tqdm
from config import (DEVICE, REWARD_HIDDEN, REWARD_LAYERS, REWARD_EPOCHS,
                    REWARD_BATCH, REWARD_LR, REWARD_CKPT,
                    ATOM_TYPES, BOND_TYPES, NODE_DIM, EDGE_DIM, PROCESSED_CSV)

def _atom_features(atom):
    sym = atom.GetSymbol()
    oh  = [0.0] * len(ATOM_TYPES)
    idx = ATOM_TYPES.index(sym) if sym in ATOM_TYPES else len(ATOM_TYPES)-1
    oh[idx] = 1.0
    return oh + [atom.GetFormalCharge()/4.0, float(atom.IsInRing()), float(atom.GetIsAromatic())]

def _bond_features(bond):
    bt  = bond.GetBondType()
    oh  = [0.0] * len(BOND_TYPES)
    oh[BOND_TYPES.index(bt) if bt in BOND_TYPES else 0] = 1.0
    return oh

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    except Exception:
        return None
    rows, cols, ef = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = _bond_features(bond)
        rows += [i, j]; cols += [j, i]; ef += [f, f]
    if not rows:
        return None
    return Data(x=x,
                edge_index=torch.tensor([rows, cols], dtype=torch.long),
                edge_attr=torch.tensor(ef, dtype=torch.float))

class MPNNLayer(MessagePassing):
    def __init__(self, hidden):
        super().__init__(aggr="sum")
        self.msg_net = nn.Sequential(
            nn.Linear(hidden*2+EDGE_DIM, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.upd_net = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(x + self.upd_net(torch.cat([x, agg], dim=-1)))

    def message(self, x_i, x_j, edge_attr):
        return self.msg_net(torch.cat([x_i, x_j, edge_attr], dim=-1))

class RewardModel(nn.Module):
    def __init__(self, hidden=REWARD_HIDDEN, n_layers=REWARD_LAYERS):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(NODE_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.mpnn_layers = nn.ModuleList([MPNNLayer(hidden) for _ in range(n_layers)])

        def _head(sigmoid=False):
            layers = [nn.Linear(hidden, hidden//2), nn.SiLU(),
                      nn.Dropout(0.1), nn.Linear(hidden//2, 1)]
            if sigmoid:
                layers.append(nn.Sigmoid())
            return nn.Sequential(*layers)

        self.head_qed  = _head(sigmoid=True)
        self.head_logp = _head(sigmoid=False)
        self.head_tpsa = _head(sigmoid=True)

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.embed(x)
        for layer in self.mpnn_layers:
            h = layer(h, ei, ea)
        g = global_mean_pool(h, batch)
        return self.head_qed(g).squeeze(-1), self.head_logp(g).squeeze(-1), self.head_tpsa(g).squeeze(-1)

    @torch.no_grad()
    def predict_smiles(self, smiles, device=None):
        device = device or DEVICE
        g = smiles_to_graph(smiles)
        if g is None:
            return None
        self.eval()
        bat = Batch.from_data_list([g.to(device)])
        q, l, t = self(bat)
        return {"qed": round(float(q[0]),3), "logp": round(float(l[0]),3), "tpsa": round(float(t[0])*200,2)}

    @torch.no_grad()
    def predict_batch(self, smiles_list, device=None):
        device = device or DEVICE
        graphs = [smiles_to_graph(s) for s in smiles_list]
        valid  = [(i,g) for i,g in enumerate(graphs) if g is not None]
        results = [None]*len(smiles_list)
        if not valid:
            return results
        idxs, gs = zip(*valid)
        bat = Batch.from_data_list(list(gs)).to(device)
        self.eval()
        q, l, t = self(bat)
        for rank, orig_i in enumerate(idxs):
            results[orig_i] = {"qed": round(float(q[rank]),3),
                               "logp": round(float(l[rank]),3),
                               "tpsa": round(float(t[rank])*200,2)}
        return results

    @staticmethod
    def composite_score(props, target_qed=0.75, target_logp=2.5, target_tpsa=80.0):
        if props is None:
            return 0.0
        r_qed  = props["qed"]
        r_logp = max(0.0, 1.0 - abs(props["logp"] - target_logp) / 5.0)
        r_tpsa = max(0.0, 1.0 - abs(props["tpsa"] - target_tpsa) / 100.0)
        return round((r_qed + r_logp + r_tpsa) / 3.0, 4)

class RewardLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pq, pl, pt, tq, tl, tt):
        lq = self.mse(pq, tq)
        ll = self.mse(pl, tl/10.0)
        lt = self.mse(pt, tt/200.0)
        return lq+ll+lt, {"qed": lq.item(), "logp": ll.item(), "tpsa": lt.item()}

def train_reward_model(df, epochs=REWARD_EPOCHS, batch_size=REWARD_BATCH,
                       lr=REWARD_LR, save_path=REWARD_CKPT, device=DEVICE):
    print(f"[reward] Device: {device}")
    print("[reward] Building graphs...")
    graphs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SMILES->graph"):
        g = smiles_to_graph(row["smiles"])
        if g is None:
            continue
        g.y_qed  = torch.tensor([row["qed"]],  dtype=torch.float)
        g.y_logp = torch.tensor([row["logp"]], dtype=torch.float)
        g.y_tpsa = torch.tensor([row["tpsa"]], dtype=torch.float)
        graphs.append(g)
    print(f"[reward] {len(graphs)} graphs ready.")
    n_val = max(500, int(len(graphs)*0.05))
    train_g, val_g = graphs[:-n_val], graphs[-n_val:]
    train_loader = PyGLoader(train_g, batch_size=batch_size, shuffle=True)
    val_loader   = PyGLoader(val_g,   batch_size=batch_size, shuffle=False)
    model = RewardModel().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = RewardLoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    best_val = float("inf")
    for epoch in range(1, epochs+1):
        model.train()
        t_loss = 0.0
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(device)
            pq, pl, pt = model(batch)
            loss, _ = crit(pq, pl, pt,
                           batch.y_qed.squeeze(), batch.y_logp.squeeze(), batch.y_tpsa.squeeze())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item()
        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pq, pl, pt = model(batch)
                loss, _ = crit(pq, pl, pt,
                               batch.y_qed.squeeze(), batch.y_logp.squeeze(), batch.y_tpsa.squeeze())
                v_loss += loss.item()
        avg_t = t_loss/len(train_loader)
        avg_v = v_loss/len(val_loader)
        sched.step(avg_v)
        print(f"  Epoch {epoch:02d}  train={avg_t:.4f}  val={avg_v:.4f}")
        if avg_v < best_val:
            best_val = avg_v
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved (val={best_val:.4f})")
    return model

def load_reward_model(path=REWARD_CKPT, device=DEVICE):
    model = RewardModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    g = smiles_to_graph(smi)
    print(f"Aspirin: {g.x.shape[0]} atoms, {g.edge_index.shape[1]//2} bonds")
    m = RewardModel()
    q,l,t = m(Batch.from_data_list([g]))
    print(f"QED={q.item():.3f} LogP={l.item():.3f} TPSA={t.item()*200:.1f}")
    from dataset import build_dataset
    df = build_dataset()
    train_reward_model(df)
