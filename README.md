# MolGen: AI Drug Molecule Generation

An AI-powered system for generating novel drug-like molecules using MolT5 transformer and GNN reward model.

## Features

- **Text-to-Molecule Generation**: Input natural language descriptions (e.g., "high solubility drug") or precise property targets
- **Property Optimization**: GNN-based reward model ensures generated molecules match desired QED, LogP, TPSA, and MW
- **Validation**: Built-in chemical validity checks and gibberish protection
- **REST API**: FastAPI backend for easy integration

## Architecture

- **MolT5**: Fine-tuned T5 transformer for SMILES generation
- **GNN Reward Model**: Graph Neural Network for molecular property prediction
- **Natural Language Parser**: Converts human descriptions to numeric targets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/molgen.git
cd molgen
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers torch-geometric rdkit fastapi uvicorn pandas tqdm
```

3. Download model checkpoints (if available) or train your own.

## Usage

### Training

1. Train reward model:
```bash
python run.py --step reward
```

2. Fine-tune MolT5:
```bash
python run.py --step finetune
```

### Generation

Generate molecules via CLI:
```bash
python generate.py --qed 0.8 --logp 2.5 --tpsa 80 --mw 350 --n 10
```

### API

Start the server:
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Example request:
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"text_prompt": "high solubility drug", "n": 5}'
```

## Project Structure

```
molgen/
├── api.py              # FastAPI server
├── generate.py         # Molecule generation pipeline
├── reward_model.py     # GNN property predictor
├── finetune.py         # MolT5 training
├── dataset.py          # Data processing
├── prompt_parser.py    # Natural language parsing
├── config.py           # Configuration
├── run.py              # Training orchestration
└── checkpoints/        # Model weights (not included)
```

## License

MIT License

## Citation

If you use this code, please cite our work.