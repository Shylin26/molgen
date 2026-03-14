"""
api.py — Final Optimized FastAPI backend for MolGen
Includes Natural Language Parsing + Gibberish Guard
"""

import os
import torch
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

# ── Import ML pipeline ────────────────────────────────────────────────────
from config import OUTPUTS_DIR, DEVICE
from dataset import make_custom_prompt
from generate import load_models, generate_molecules, save_results
from prompt_parser import parse_natural_language  # <--- YOUR NEW FILE

# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="MolGen API",
    description="AI Drug Molecule Generation — MolT5 + GNN Reward Model",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models at startup ────────────────────────────────────────────────
print("[api] Loading models at startup...")
MODEL, TOKENIZER, REWARD = load_models()
print("[api] Models ready.")

# ── Request / Response schemas ────────────────────────────────────────────

class GenerateRequest(BaseModel):
    # Optional text prompt for "Natural Language Mode"
    text_prompt: str = Field(default=None, description="Human text like 'low solubility drug'")
    
    # Numeric targets for "Expert Mode" (Defaults used if text_prompt is empty)
    qed:  float = Field(default=0.80, ge=0.0, le=1.0)
    logp: float = Field(default=2.50, ge=-5.0, le=10.0)
    tpsa: float = Field(default=80.0, ge=0.0, le=200.0)
    mw:   float = Field(default=350.0, ge=100.0, le=600.0)
    n:    int   = Field(default=5,   ge=1, le=50)

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "models": "loaded"}

@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        # 1. DECIDE MODE: Natural Language or Numeric Sliders
        if req.text_prompt:
            print(f"[api] Natural Language Mode: '{req.text_prompt}'")
            targets = parse_natural_language(req.text_prompt)
            
            # GIBBERISH GUARD: parse_natural_language returns None for "ggg"
            if targets is None:
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid prompt. Please use chemical terms (e.g., 'small molecule', 'polar')."
                )
            
            # Map parsed values to our variables
            target_qed  = targets['qed']
            target_logp = targets['logp']
            target_tpsa = targets['tpsa']
            target_mw   = targets['mw']
        else:
            # Expert Mode: Use slider values directly
            target_qed  = req.qed
            target_logp = req.logp
            target_tpsa = req.tpsa
            target_mw   = req.mw

        # 2. CREATE THE SCIENTIFIC PROMPT
        prompt = make_custom_prompt(
            qed=target_qed, 
            logp=target_logp, 
            tpsa=target_tpsa, 
            mw=target_mw
        )

        print(f"[api] Final Model Prompt: {prompt}")

        # 3. RUN GENERATION
        molecules = generate_molecules(
            MODEL, TOKENIZER, REWARD,
            prompt,
            n_generate  = req.n,
            target_qed  = target_qed,
            target_logp = target_logp,
            target_tpsa = target_tpsa,
            target_mw   = target_mw
        )

        # 4. FINAL VALIDATION
        if not molecules:
            raise HTTPException(
                status_code=422, 
                detail="Chemistry error: Model failed to generate a valid structure for these targets."
            )

        save_results(molecules)

        return {
            "status": "success",
            "mode": "natural_language" if req.text_prompt else "expert_sliders",
            "targets": {"qed": target_qed, "mw": target_mw, "logp": target_logp, "tpsa": target_tpsa},
            "molecules": molecules[:5] 
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[SYSTEM ERROR] {e}")
        raise HTTPException(status_code=500, detail="Internal AI Processing Error")

@app.get("/results")
def get_results():
    path = os.path.join(OUTPUTS_DIR, "generated_molecules.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No previous results found.")
    df = pd.read_csv(path)
    return {"molecules": df.head(10).to_dict(orient="records")}

# ── Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # Listening on Port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)