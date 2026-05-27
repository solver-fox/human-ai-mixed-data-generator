import asyncio

from dotenv import load_dotenv

from mixed_sequence import create_mixed_sequences
from sandwitch_sequence import MIN_SANDWITCH_WORDS, create_sandwitch_sequences
from utils import (
    get_api_key,
    get_client,
    get_concurrency,
    get_models,
    load_split_texts,
    parse_args,
)

load_dotenv()

SEQUENCE_CREATORS = {
    "append": create_mixed_sequences,
    "sandwitch": create_sandwitch_sequences,
}


async def main():
    args = parse_args()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not splits:
        raise ValueError("At least one split must be specified.")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1.")

    create_sequences = SEQUENCE_CREATORS[args.mode]
    min_words = MIN_SANDWITCH_WORDS if args.mode == "sandwitch" else None

    api_key = get_api_key()
    models = get_models(api_key)
    print(f"Mode: {args.mode}")
    print(f"Using {len(models)} OpenRouter chat models (random per entry).")
    range_label = (
        f"[{args.from_idx}, {args.to_idx})"
        if args.to_idx is not None
        else f"[{args.from_idx}, end)"
    )
    print(f"Index range per split: {range_label}, chunk size: {args.chunk_size}")

    for split in splits:
        human_entries = load_split_texts(
            args.data_dir,
            split,
            start=args.from_idx,
            end=args.to_idx,
            min_words=min_words or 3,
        )
        concurrency = get_concurrency(len(human_entries))
        client = get_client(api_key, concurrency)

        print(f"\n{split}: {len(human_entries)} entries, {concurrency} parallel requests")
        try:
            await create_sequences(
                client,
                human_entries,
                models,
                concurrency,
                split=split,
                output_dir=args.output_dir,
                from_idx=args.from_idx,
                chunk_size=args.chunk_size,
            )
        finally:
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
