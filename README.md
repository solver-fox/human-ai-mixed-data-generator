# Human–AI Mixed Data Generator

Generate word-level labeled datasets for training models to detect human vs AI text. Human text comes from [MiniPile](https://huggingface.co/datasets/JeanKaddour/minipile); AI segments are produced via [OpenRouter](https://openrouter.ai/) using a random model per sample.

Output is saved as **PyTorch `.pt` chunk files** (default 1000 samples per file), ready for `DataLoader` training.

## Generation modes

Two styles are supported via `--mode`:

### `sandwitch` (default) — human + AI + human

Implemented in `sandwitch_sequence.py`.

1. Take a random prefix (3–10 words) and suffix (3–10 words) from the human text.
2. Ask an LLM to write bridge text connecting start → end.
3. Concatenate: **human start + AI bridge + human end**.
4. Labels: `0` = human, `1` = AI.

```
[human start ...] [AI bridge ...] [human end ...]
 0 0 0 0 0 0 0     1 1 1 1 1       0 0 0 0 0 0
```

### `append` — human + AI

Implemented in `mixed_sequence.py`.

1. Take a random prefix of the human text (3–100 words).
2. Ask an LLM to continue writing naturally.
3. Concatenate: **human prefix + AI continuation**.
4. Labels: `0` = human, `1` = AI.

```
[human prefix ...] [AI continuation ...]
 0 0 0 0 0 0 0       1 1 1 1 1 1
```

## Setup

```bash
pip install -r requirements.txt
cp .env.ex .env
# Edit .env and set OPENROUTER_API_KEY
```

### MiniPile data

Place MiniPile parquet files under `dataset/minipile/data/`:

```
dataset/minipile/data/
  train-*.parquet      # 1,000,000 texts (12 shards)
  validation-*.parquet # 500 texts
  test-*.parquet       # 10,000 texts
```

Download from Hugging Face:

```bash
huggingface-cli download JeanKaddour/minipile --repo-type dataset \
  --local-dir dataset/minipile
```

Then move or symlink parquet files into `dataset/minipile/data/`.

## Usage

Sandwitch mode (default):

```bash
python main.py --from 0 --to 1000
```

Append mode:

```bash
python main.py --mode append --from 0 --to 1000
```

Process a specific index range for train, validation, and test:

```bash
python main.py --from 0 --to 2500 --splits train,validation,test
```

Only the train split, indices 5000–6999:

```bash
python main.py --splits train --from 5000 --to 7000
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `sandwitch` | `append` (human+AI) or `sandwitch` (human+AI+human) |
| `--data-dir` | `dataset/minipile/data` | Input parquet directory |
| `--output-dir` | `output` | Output directory for `.pt` chunks |
| `--splits` | `train,validation,test` | Comma-separated splits to process |
| `--from` | `0` | Start index in each split (inclusive) |
| `--to` | end of split | End index in each split (exclusive) |
| `--chunk-size` | `1000` | Samples per output file |

### Environment variables

All CLI options can also be set in `.env` (see `.env.ex`):

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key (required) |
| `GENERATION_MODE` | `append` or `sandwitch` (default: `sandwitch`) |
| `OPENROUTER_CONCURRENCY` | Parallel API requests (default: `32`) |
| `OPENROUTER_MODELS` | Optional comma-separated model list; if unset, models are chosen from the filtered pool |
| `OPENROUTER_FREE_ONLY` | If `true`, only free models (`:free` or $0 pricing) |
| `OPENROUTER_MAX_PROMPT_PRICE` | Max input price per token (USD) |
| `OPENROUTER_MAX_COMPLETION_PRICE` | Max output price per token (USD) |
| `OPENROUTER_AI_WORDS_MIN` / `MAX` | AI word count range (append default 10–100; sandwitch default 5–100) |
| `MINIPILE_DATA_DIR` | Input data directory |
| `MINIPILE_OUTPUT_DIR` | Output directory |
| `MINIPILE_SPLITS` | Splits to process |
| `MINIPILE_FROM` | Start index |
| `MINIPILE_TO` | End index |
| `MINIPILE_CHUNK_SIZE` | Chunk file size |

## Output format

Files are written under `{output_dir}/{split}/` with inclusive index ranges in the filename:

```
output/
  train/
    train_0_999.pt
    train_1000_1999.pt
  validation/
    validation_0_499.pt
  test/
    test_0_999.pt
```

Each `.pt` file contains:

```python
{
    "texts": list[str],           # full mixed text
    "labels": list[Tensor],       # word-level labels (LongTensor, variable length)
    "models": list[str],          # OpenRouter model ID used per sample
    "indices": Tensor,            # original MiniPile index for each sample
}
```

## Loading in PyTorch

Use `load_pt.py` or the notebook `load_pt.ipynb`:

```python
from torch.utils.data import DataLoader
from load_pt import load_split, make_dataloader

# All chunks for a split
train_ds = load_split("output", "train")

# Index range only
train_ds = load_split("output", "train", from_idx=0, to_idx=100)

# DataLoader with padded labels (-1 = padding)
loader = make_dataloader("output", "train", batch_size=32, shuffle=True)
batch = next(iter(loader))
```

Inspect chunks from the command line:

```bash
python load_pt.py output/train/train_0_999.pt --sample 2
python load_pt.py --split train --output-dir output
```

Because label sequences vary in length, use a custom `collate_fn` if you need padded batches manually:

```python
from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    texts = [item["text"] for item in batch]
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-1)
    return {"text": texts, "labels": labels}
```

## MiniPile split sizes

| Split | Samples |
|-------|---------|
| train | 1,000,000 |
| validation | 500 |
| test | 10,000 |

Use `--from` / `--to` to process manageable batches. The train split is large; start small (e.g. `--from 0 --to 100`) before running bigger jobs.

## Saving OpenRouter balance

At ~$1.50 per 100 texts with expensive models, processing all **1M train** samples would cost on the order of **$15,000**. The main reasons:

1. **Random expensive models** — the default pool includes models like `o3-pro` and Claude Opus that cost far more than small/free models.
2. **Every text is a unique request** — each sample sends a different prompt, so OpenRouter cannot reuse one cached prompt across the whole run.
3. **Long outputs** — the script asks for many AI words per sample (input + output tokens are both billed).
4. **Retries** — failed models trigger extra API calls.

### Recommended `.env` for large runs

```env
OPENROUTER_FREE_ONLY=true
OPENROUTER_AI_WORDS_MIN=10
OPENROUTER_AI_WORDS_MAX=40
GENERATION_MODE=sandwitch
```

Or pin a small set of cheap models:

```env
OPENROUTER_MODELS=deepseek/deepseek-v4-flash:free,google/gemma-4-26b-a4b-it:free,openrouter/free
```

Optional price caps (USD per token, from the OpenRouter models API):

```env
OPENROUTER_MAX_PROMPT_PRICE=0.0000005
OPENROUTER_MAX_COMPLETION_PRICE=0.000001
```

The script also passes `max_tokens` to cap generation length per request.

### Rough cost math

```
cost ≈ (input_tokens × prompt_price) + (output_tokens × completion_price)
```

With free models, marginal cost is **$0** (subject to rate limits). With cheap paid models, 1M samples at ~150 input + ~50 output tokens might be tens of dollars instead of thousands.

### Other strategies

- Process in batches (`--from` / `--to`) and monitor spend on the [OpenRouter dashboard](https://openrouter.ai/activity).
- Lower concurrency if you hit rate limits (retries cost money).
- For maximum savings, run a local model (Ollama, vLLM) instead of OpenRouter for the AI continuation step.

## Notes

- Progress is printed as `Progress: N/M (X%)` while API requests run in parallel.
- If a model fails (404, 500, rate limit, empty response, etc.), another random model is tried automatically (up to 10 retries).
- Deprecated OpenRouter models and models with an `expiration_date` are excluded from the random pool.
- API usage is billed through your OpenRouter account; cost depends on models selected and text volume.

## Project layout

```
main.py                # CLI entry point; orchestrates generation and saving
utils.py               # OpenRouter client, data I/O, retries, parallel runner
mixed_sequence.py      # append mode: human + AI
sandwitch_sequence.py  # sandwitch mode: human + AI + human
load_pt.py             # Load and inspect .pt chunk files
load_pt.ipynb          # Notebook for exploring chunks vs original text
pytorch_dataset.py     # Backward-compatible re-exports from load_pt.py
requirements.txt
.env.ex                # Environment variable template
dataset/minipile/      # MiniPile source data (not committed)
output/                # Generated .pt chunks (not committed)
```


# Checkpoint 2 on full test set
python validate_token_model.py --model-dir output/dactyl_token_finetuned/2

# Checkpoint 1 on validation split
python validate_token_model.py \
  --model-dir output/dactyl_token_finetuned/1 \
  --data pickles/validation_samples.pkl

# Quick smoke test
python validate_token_model.py \
  --model-dir output/dactyl_token_finetuned/2 \
  --max-samples 200


# Easiest — auto-detects foundation + .pth
python validate_token_model.py --model-dir llm-detection/models

# Also works
python validate_token_model.py --model-dir llm-detection/models/deberta-v3-large-hf-weights

# Explicit
python validate_token_model.py \
  --model-dir llm-detection/models/deberta-v3-large-hf-weights \
  --weights-path llm-detection/models/deberta-large-ls03-ctx1024.pth