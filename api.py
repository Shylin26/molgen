"""
api.py — FastAPI backend for MolGen
Run: python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd

from config import OUTPUTS_DIR, DEVICE
from dataset import make_custom_prompt
from generate import load_models, generate_molecules, save_results
from prompt_parser import parse_natural_language
from mol3d import generate_3d_data

app = FastAPI(
    title="MolGen API",
    description="AI Drug Molecule Generation — MolT5 + GNN Reward Model",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models ONCE at startup
print("[api] Loading models at startup...")
MODEL, TOKENIZER, REWARD = load_models()
print("[api] Models ready.")


# ── Schemas ────────────────────────────────────────────────────────────────────

class Constraints(BaseModel):
    qed:  float = Field(default=0.80, ge=0.0,   le=1.0,   description="Drug-likeness (0-1)")
    logp: float = Field(default=2.50, ge=-5.0,  le=10.0,  description="Lipophilicity")
    tpsa: float = Field(default=80.0, ge=0.0,   le=200.0, description="Polar surface area")
    mw:   float = Field(default=350.0, ge=100.0, le=600.0, description="Molecular weight")

class WebGenerateRequest(BaseModel):
    prompt: Optional[str] = Field(default=None, description="Plain English prompt")
    constraints: Constraints
    n: int = Field(default=10, ge=1, le=50, description="Candidates to generate")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if API and models are running."""
    return {
        "status" : "ok",
        "device" : str(DEVICE),
        "models" : "loaded"
    }


@app.post("/generate")
def generate(req: WebGenerateRequest):
    """
    Generate drug-like molecules.
    Expects prompt and constraints object.
    """
    try:
        # Mode 1: Natural language
        if req.prompt and req.prompt.strip():
            print(f"[api] Natural language mode: '{req.prompt}'")
            targets = parse_natural_language(req.prompt)

            if targets is None:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid prompt. Use chemistry terms like 'antiviral', 'small', 'polar', 'brain', 'cancer'."
                )

            target_qed  = targets["qed"]
            target_logp = targets["logp"]
            target_tpsa = targets["tpsa"]
            target_mw   = targets["mw"]

        # Mode 2: Use provided constraints directly if no prompt or prompt parsing failed
        else:
            print(f"[api] Using constraints: {req.constraints}")
            target_qed  = req.constraints.qed
            target_logp = req.constraints.logp
            target_tpsa = req.constraints.tpsa
            target_mw   = req.constraints.mw

        # Build scientific prompt
        prompt = make_custom_prompt(
            qed=target_qed, logp=target_logp,
            tpsa=target_tpsa, mw=target_mw
        )
        print(f"[api] Prompt: {prompt}")

        # Run generation
        molecules = generate_molecules(
            MODEL, TOKENIZER, REWARD,
            prompt,
            n_generate  = req.n,
            target_qed  = target_qed,
            target_logp = target_logp,
            target_tpsa = target_tpsa,
            target_mw   = target_mw,
        )

        if not molecules:
            raise HTTPException(
                status_code=422,
                detail="Model could not generate valid molecules for these targets. Try adjusting parameters."
            )

        save_results(molecules)
        
        # Format the response with 3D data and top matches
        # Sort molecules by reward score securely
        molecules.sort(key=lambda x: x.get("reward_score", 0), reverse=True)
        
        primary_molecule = molecules[0]
        # Generate 3D Data for primary
        td_data = generate_3d_data(primary_molecule["smiles"])
        
        primary_response = {
            "smiles"        : primary_molecule["smiles"],
            "3d_data"       : td_data,
            "score"         : primary_molecule.get("reward_score", 0),
            "qed"           : primary_molecule.get("qed"),
            "logp"          : primary_molecule.get("logp"),
            "tpsa"          : primary_molecule.get("tpsa"),
            "mw"            : primary_molecule.get("mw"),
            "lipinski"      : primary_molecule.get("lipinski"),
            "fragment_score": primary_molecule.get("fragment_score"),
        }
        
        # Get up to 5 alternatives
        alternatives = []
        for m in molecules[1:6]:
            alternatives.append({
                "smiles"        : m["smiles"],
                "score"         : m.get("reward_score", 0),
                "qed"           : m.get("qed"),
                "logp"          : m.get("logp"),
                "tpsa"          : m.get("tpsa"),
                "mw"            : m.get("mw"),
                "lipinski"      : m.get("lipinski"),
                "fragment_score": m.get("fragment_score"),
            })

        return {
            "primary": primary_response,
            "alternatives": alternatives
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[api] ERROR: {e}")
        raise HTTPException(status_code=500, detail="Internal generation error.")


@app.get("/results")
def get_results():
    """Return last 10 generated molecules from CSV."""
    path = os.path.join(OUTPUTS_DIR, "generated_molecules.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No results yet. Call /generate first.")
    df = pd.read_csv(path)
    return {"molecules": df.head(10).to_dict(orient="records")}


class ValidateRequest(BaseModel):
    smiles: str

@app.post("/validate")
def validate_smiles_endpoint(req: ValidateRequest):
    """Validate a SMILES string and return its properties."""
    from generate import validate_smiles
    result = validate_smiles(req.smiles)
    if result is None:
        raise HTTPException(status_code=422, detail="Invalid or non-drug-like SMILES.")
    td_data = generate_3d_data(req.smiles)
    result["3d_data"] = td_data
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)