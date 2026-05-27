import asyncio
import time

from dotenv import load_dotenv

from generator.mixed_sequence import create_mixed_sequences
from generator.mixed_v2 import MIN_MIXED_V2_WORDS, create_mixed_v2_sequences
from generator.sandwitch_sequence import MIN_SANDWITCH_WORDS, create_sandwitch_sequences
from generator.sandwitch_v2 import MIN_SANDWITCH_V2_WORDS, create_sandwitch_v2_sequences
from generator.utils import (
    get_api_key,
    get_client,
    get_concurrency,
    get_models,
    load_split_texts,
    log_timing,
    log_timing_msg,
    parse_args,
)

load_dotenv()

SEQUENCE_CREATORS = {
    "append": create_mixed_sequences,
    "mixed_v2": create_mixed_v2_sequences,
    "sandwitch": create_sandwitch_sequences,
    "sandwitch_v2": create_sandwitch_v2_sequences,
}


async def main():
    with log_timing("main (total)"):
        args = parse_args()
        splits = [split.strip() for split in args.splits.split(",") if split.strip()]
        if not splits:
            raise ValueError("At least one split must be specified.")
        if args.chunk_size < 1:
            raise ValueError("--chunk-size must be at least 1.")

        create_sequences = SEQUENCE_CREATORS[args.mode]
        if args.mode == "sandwitch":
            min_words = MIN_SANDWITCH_WORDS
        elif args.mode == "sandwitch_v2":
            min_words = MIN_SANDWITCH_V2_WORDS
        elif args.mode == "mixed_v2":
            min_words = MIN_MIXED_V2_WORDS
        else:
            min_words = None

        api_key = get_api_key()
        with log_timing("get_models"):
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
            with log_timing(f"load_split_texts ({split})"):
                human_entries = load_split_texts(
                    args.data_dir,
                    split,
                    start=args.from_idx,
                    end=args.to_idx,
                    min_words=min_words or 3,
                )
            concurrency = get_concurrency(len(human_entries))
            with log_timing("get_client (httpx AsyncClient)"):
                client = get_client(api_key, concurrency)

            print(f"\n{split}: {len(human_entries)} entries, {concurrency} parallel requests")
            try:
                saved_count = await create_sequences(
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
                close_start = time.perf_counter()
                await client.close()
                log_timing_msg("client.close", time.perf_counter() - close_start)
            if saved_count == 0:
                raise SystemExit(
                    f"No samples saved for split '{split}'. "
                    "Check logs for skipped/failed entries."
                )


if __name__ == "__main__":
    asyncio.run(main())
