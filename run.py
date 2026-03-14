import argparse

def step_data():
    print("\n=== STEP 1: Dataset ===")
    from dataset import build_dataset
    df = build_dataset()
    print(f"✓ {len(df)} molecules ready")
    return df

def step_reward(df=None):
    print("\n=== STEP 2: Reward Model (GNN from scratch) ===")
    from dataset import build_dataset
    from reward_model import train_reward_model
    if df is None:
        df = build_dataset()
    train_reward_model(df)
    print("✓ Reward model trained")

def step_finetune(resume=False):
    print("\n=== STEP 3: Fine-tune MolT5 ===")
    from finetune import train
    train(resume=resume)
    print("✓ MolT5 fine-tuned")

def step_generate():
    print("\n=== STEP 4: Generate Molecules ===")
    from generate import load_models, generate_molecules, save_results
    from dataset import make_custom_prompt
    model, tokenizer, reward = load_models()
    prompt = make_custom_prompt(qed=0.8, logp=2.5, tpsa=80.0, mw=350.0)
    mols = generate_molecules(model, tokenizer, reward, prompt)
    if mols:
        save_results(mols)
        print(f"✓ {len(mols)} molecules generated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all",
                        choices=["all","data","reward","finetune","generate"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.step == "data":
        step_data()
    elif args.step == "reward":
        step_reward()
    elif args.step == "finetune":
        step_finetune(resume=args.resume)
    elif args.step == "generate":
        step_generate()
    elif args.step == "all":
        df = step_data()
        step_reward(df)
        step_finetune(resume=args.resume)
        step_generate()
        print("\n🏆 DONE. Check outputs/ folder.")
