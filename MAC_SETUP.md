# OmniscientLLM — Mac M1 Pro Setup Steps

Step-by-step instructions to clone, set up, and run the full training pipeline on a Mac M1 Pro (MLX on Apple Silicon).

---

## Step 1: Clone and Set Up Environment

```bash
git clone https://github.com/vipurkumar/omni-llm.git
cd omni-llm

python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies (core + MLX training + dev tools)
pip install -e ".[train,dev]"
```

This installs: `mlx`, `datasets`, `wandb`, `fastapi`, `uvicorn`, `tokenizers`, `aiosqlite`, `pytest`, `ruff`, `httpx`.

## Step 2: Verify Installation

```bash
# Confirm MLX works on Apple Silicon
python3 -c "import mlx.core as mx; print(f'MLX OK — device: {mx.default_device()}')"

# Run tests to confirm everything is wired up
make test
```

All 29 tests should pass.

## Step 3: Prepare Training Data

```bash
mkdir -p data/raw data/processed

# Download pretraining datasets (code + NL mix)
# Requires HF_TOKEN for gated datasets
export HF_TOKEN="your_huggingface_token"
python3 -m data.download --output-dir data/raw --datasets all --max-tokens 20000000000
```

**Start smaller first** (recommended for first run):
```bash
python3 -m data.download --output-dir data/raw --datasets code --max-tokens 100000000
```

The data lands in `data/raw/` as JSONL shards.

## Step 4: Train the Tokenizer

```bash
python3 -m tokenizer.train_tokenizer --data-dir data/raw --vocab-size 32768

# Verify
ls -la tokenizer/omniscient-tokenizer.json
```

Produces `tokenizer/omniscient-tokenizer.json` (~2MB).

## Step 5: Pretrain the Model

**Quick smoke test first** (verify the training loop works before committing to days):
```bash
python3 -m training.pretrain \
    --data-dir data/raw \
    --output-dir checkpoints/test \
    --max-steps 100 \
    --batch-size 2 \
    --grad-accum 4 \
    --checkpoint-every 50
```

**Full pretraining** (~6-8 days on M1 Pro for 100K steps):
```bash
python3 -m training.pretrain \
    --data-dir data/raw \
    --output-dir checkpoints/pretrain \
    --max-steps 100000 \
    --batch-size 2 \
    --grad-accum 16 \
    --max-lr 3e-4 \
    --warmup-steps 2000 \
    --checkpoint-every 5000
```

Optional: add `--wandb` to monitor with Weights & Biases.

Checkpoints save to `checkpoints/pretrain/step_XXXXX/`.

## Step 6: Supervised Fine-Tuning (SFT)

Prepare SFT data as JSONL — each line:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Place at `data/sft_train.jsonl`, then run:
```bash
python3 -m training.sft \
    --checkpoint checkpoints/pretrain/step_100000 \
    --data data/sft_train.jsonl \
    --output-dir checkpoints/sft \
    --epochs 3 \
    --lr 2e-5 \
    --warmup-steps 100
```

~12-18 hours for 235K examples.

## Step 7: Generate Synthetic CoT Data (Optional but Recommended)

```bash
python3 -m data.generate_cot \
    --input data/coding_problems.jsonl \
    --output data/cot_examples.jsonl \
    --api-url https://api.openai.com/v1/chat/completions \
    --model gpt-4 \
    --max-workers 4
```

Then merge `data/cot_examples.jsonl` into your SFT data and re-run Step 6.

## Step 8: FIM Training

```bash
python3 -m training.fim \
    --checkpoint checkpoints/sft \
    --data-dir data/raw \
    --output-dir checkpoints/fim
```

~1-2 days.

## Step 9: DPO Alignment

Prepare preference data as JSONL — each line:
```json
{"prompt": "...", "chosen": "good response", "rejected": "bad response"}
```

Place at `data/dpo_pairs.jsonl`, then run:
```bash
python3 -m training.dpo \
    --checkpoint checkpoints/fim \
    --data data/dpo_pairs.jsonl \
    --output-dir checkpoints/dpo \
    --beta 0.1 \
    --lr 5e-7 \
    --max-steps 3000
```

~4-8 hours.

## Step 10: Serve the Model

```bash
# Copy final checkpoint to expected location
cp -r checkpoints/dpo checkpoints/latest

# Save config alongside weights
python3 -c "
from model.config import OmniscientConfig
OmniscientConfig().save('checkpoints/latest/config.json')
"

# Start the API server
make serve
```

Test it:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Write a hello world in Python"}], "stream": false}'
```

## Step 11: VS Code Extension (Optional)

```bash
cd extension
npm install
npm run compile

# Package and install
npx vsce package
code --install-extension omniscient-llm-0.1.0.vsix
```

Open VS Code, press Ctrl+Shift+O (Cmd+Shift+O on Mac) to open the chat panel.

## Step 12: Run Evaluations

```bash
make eval

# Or specific benchmarks:
python3 -m evals.run_eval --benchmark humaneval --checkpoint checkpoints/latest
python3 -m evals.run_eval --benchmark mt_bench --checkpoint checkpoints/latest
```

---

## Quick Reference

| Step | Command | Time |
|------|---------|------|
| Install | `pip install -e ".[train,dev]"` | 2 min |
| Test | `make test` | 1 sec |
| Tokenizer | `make train-tokenizer` | 10-30 min |
| Pretrain | `make pretrain` | 6-8 days |
| SFT | `make sft` | 12-18 hrs |
| FIM | `make fim` | 1-2 days |
| DPO | `make dpo` | 4-8 hrs |
| Serve | `make serve` | instant |
| Eval | `make eval` | varies |

**Total training time: ~10-12 days end-to-end on M1 Pro.**
