import os, json, argparse
import torch
from torch.utils.data import DataLoader, random_split
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
from config import (DEVICE, MOLT5_MODEL, MOLT5_EPOCHS, MOLT5_BATCH, MOLT5_LR,
                    MOLT5_CKPT_BEST, MOLT5_CKPT_LAST, TRAIN_LOG,
                    MAX_MOLS, MAX_SEQ_LEN, GRAD_CLIP)
from dataset import build_dataset, MoleculeDataset

def save_ckpt(path, model, optimizer, epoch, best_val):
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "best_val": best_val}, path)

def load_ckpt(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt.get("best_val", float("inf"))

def append_log(path, entry):
    log = []
    if os.path.exists(path):
        with open(path) as f:
            log = json.load(f)
    log.append(entry)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)

def train(resume=False, epochs=MOLT5_EPOCHS, max_mols=MAX_MOLS,
          batch_size=MOLT5_BATCH, lr=MOLT5_LR):
    print(f"\n{'='*55}\n  MolT5 Fine-tuning\n  Device: {DEVICE}\n{'='*55}\n")
    print("[finetune] Loading MolT5...")
    tokenizer = AutoTokenizer.from_pretrained(MOLT5_MODEL)
    model     = T5ForConditionalGeneration.from_pretrained(MOLT5_MODEL).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[finetune] {n_params/1e6:.1f}M params")
    print("[finetune] Loading dataset...")
    df       = build_dataset(max_mols=max_mols)
    full_ds  = MoleculeDataset(df, tokenizer, max_len=MAX_SEQ_LEN)
    n_val    = max(500, int(len(full_ds)*0.05))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[finetune] Train: {n_train}  Val: {n_val}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs
    warmup_steps = min(200, total_steps//10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        return max(0.05, 0.5*(1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_epoch = 1
    best_val    = float("inf")
    if resume and os.path.exists(MOLT5_CKPT_LAST):
        start_epoch, best_val = load_ckpt(MOLT5_CKPT_LAST, model, optimizer, DEVICE)
        start_epoch += 1
        print(f"[finetune] Resumed from epoch {start_epoch-1}")
    for epoch in range(start_epoch, epochs+1):
        print(f"\n── Epoch {epoch}/{epochs} ──────────────────────────")
        model.train()
        t_loss = 0
        pbar = tqdm(train_loader, desc="  train")
        for batch in pbar:
            out  = model(input_ids=batch["input_ids"].to(DEVICE),
                         attention_mask=batch["attention_mask"].to(DEVICE),
                         labels=batch["labels"].to(DEVICE))
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            t_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  val  "):
                out = model(input_ids=batch["input_ids"].to(DEVICE),
                            attention_mask=batch["attention_mask"].to(DEVICE),
                            labels=batch["labels"].to(DEVICE))
                v_loss += out.loss.item()
        avg_t = t_loss/len(train_loader)
        avg_v = v_loss/len(val_loader)
        print(f"  train={avg_t:.4f}  val={avg_v:.4f}")
        save_ckpt(MOLT5_CKPT_LAST, model, optimizer, epoch, best_val)
        print(f"  ✓ Checkpoint saved (safe to sleep)")
        if avg_v < best_val:
            best_val = avg_v
            save_ckpt(MOLT5_CKPT_BEST, model, optimizer, epoch, best_val)
            print(f"  ★ Best model saved (val={best_val:.4f})")
        append_log(TRAIN_LOG, {"epoch": epoch, "train_loss": round(avg_t,5), "val_loss": round(avg_v,5)})
    print(f"\n[finetune] Done. Best val={best_val:.4f}")
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=MOLT5_EPOCHS)
    parser.add_argument("--max",    type=int, default=MAX_MOLS)
    parser.add_argument("--batch",  type=int, default=MOLT5_BATCH)
    args = parser.parse_args()
    train(resume=args.resume, epochs=args.epochs, max_mols=args.max, batch_size=args.batch)
