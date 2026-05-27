# Human–AI Mixed Data Generator

Generate word-level labeled datasets for training models to detect where human-written text ends and AI-generated text begins. Human text comes from [MiniPile](https://huggingface.co/datasets/JeanKaddour/minipile); AI continuations are produced via [OpenRouter](https://openrouter.ai/) using a random model per sample.

## How it works

For each input text:

1. Take a random prefix of the human text (3–100 words).
2. Ask a randomly chosen LLM to continue writing naturally.
3. Concatenate human prefix + AI continuation.
4. Label each word: `0` = human, `1` = AI.

Output is saved as **PyTorch `.pt` chunk files** (default 1000 samples per file), ready for `DataLoader` training.

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

```bash
python main.py --from 0 --to 1000
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
| `OPENROUTER_CONCURRENCY` | Parallel API requests (default: `32`) |
| `OPENROUTER_MODELS` | Optional comma-separated model list; if unset, models are chosen from the filtered pool |
| `OPENROUTER_FREE_ONLY` | If `true`, only free models (`:free` or $0 pricing) |
| `OPENROUTER_MAX_PROMPT_PRICE` | Max input price per token (USD) |
| `OPENROUTER_MAX_COMPLETION_PRICE` | Max output price per token (USD) |
| `OPENROUTER_AI_WORDS_MIN` / `MAX` | AI continuation length range (default 10–100) |
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

Use the included `pytorch_dataset.py` helper:

```python
from torch.utils.data import DataLoader
from pytorch_dataset import load_split, MixedTextChunkDataset

# Load all chunks for a split
train_ds = load_split("output", "train")

# Or load a single chunk
chunk_ds = MixedTextChunkDataset("output/train/train_0_999.pt")

loader = DataLoader(train_ds, batch_size=32, shuffle=True)

for batch in loader:
    texts = batch["text"]      # list[str]
    labels = batch["labels"]   # list[Tensor], one per sample (variable length)
    models = batch["model"]      # list[str]
    indices = batch["index"]   # Tensor of original dataset indices
```

Because label sequences vary in length, use a custom `collate_fn` if you need padded batches:

```python
from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    texts = [item["text"] for item in batch]
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-1)
    return {"text": texts, "labels": labels}

loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
```

## MiniPile split sizes

| Split | Samples |
|-------|---------|
| train | 1,000,000 |
| validation | 500 |
| test | 10,000 |

Use `--from` / `--to` to process manageable batches. The train split is large; start small (e.g. `--from 0 --to 100`) before running bigger jobs.

## Saving OpenRouter balance

At ~$1.50 per 100 texts with the default settings, processing all **1M train** samples would cost on the order of **$15,000**. The main reasons:

1. **Random expensive models** — the default pool includes models like `o3-pro` and Claude Opus that cost far more than small/free models.
2. **Every text is a unique request** — each sample sends a different human prefix, so OpenRouter cannot reuse one cached prompt across the whole run.
3. **Long outputs** — the script asks for 10–100 AI words per sample (input + output tokens are both billed).
4. **Retries** — failed models trigger extra API calls.

### Recommended `.env` for large runs

```env
OPENROUTER_FREE_ONLY=true
OPENROUTER_AI_WORDS_MIN=10
OPENROUTER_AI_WORDS_MAX=40
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
- If a model fails (404, rate limit, etc.), another random model is tried automatically (up to 10 retries).
- Deprecated OpenRouter models and models with an `expiration_date` are excluded from the random pool.
- API usage is billed through your OpenRouter account; cost depends on models selected and text volume.

## Project layout

```
main.py              # Generator script
pytorch_dataset.py   # PyTorch Dataset helpers
requirements.txt
.env.ex              # Environment variable template
dataset/minipile/    # MiniPile source data (not committed)
output/              # Generated .pt chunks (not committed)
```
