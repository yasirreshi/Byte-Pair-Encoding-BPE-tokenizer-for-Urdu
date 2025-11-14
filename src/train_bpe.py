"""Train an Urdu BPE tokenizer.

This version REMOVES the enforced <5000 vocab cap entirely so you can
experiment freely with larger vocab sizes. A simple internal config block
near the top of the file lets you tweak values and just run:

    python train_bpe.py

Pipeline
1. Normalize (NFKC) + whitespace & punctuation pre-tokenization.
2. Train BPE with requested vocab size (or several if auto_tune=True).
3. Auto-tune probes larger vocab sizes to find a balance of compression & size.
4. Save tokenizer JSON + stats JSON.

Compression ratio = total_chars / total_tokens.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

# Root for relative paths (clean_structure/)
CLEAN_ROOT = Path(__file__).resolve().parents[1]

# INTERNAL DEFAULTS (base fallback)
INTERNAL_DEFAULTS = {
    "input": str(CLEAN_ROOT / "data" / "urdu_corpus_consolidated.txt"),
    "vocab_size": 5500,            # default starting requested vocab
    "model_prefix": str(CLEAN_ROOT / "models" / "urdu_bpe_experiment"),
    "min_compression": 3.2,
    "auto_tune": False,
}

# USER CONFIG: Edit these to experiment (overrides INTERNAL_DEFAULTS)
# Leave a key commented out to fall back to INTERNAL_DEFAULTS.
USER_CONFIG = {
    # "input": str(CLEAN_ROOT / "data" / "urdu_corpus_consolidated.txt"),
    # "vocab_size": 12000,
    # "model_prefix": str(CLEAN_ROOT / "models" / "urdu_bpe_large"),
    # "min_compression": 3.2,
    # "auto_tune": True,
}


def _build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation(behavior="isolated"),
    ])
    return tokenizer


def _train_on_corpus(corpus_path: str, requested_vocab: int) -> Tokenizer:
    tokenizer = _build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=requested_vocab,
        min_frequency=1,
        special_tokens=["[UNK]", "[PAD]"],
        show_progress=False,
    )
    tokenizer.train(files=[corpus_path], trainer=trainer)
    return tokenizer


def train_bpe_model(
    corpus_path: str,
    vocab_size: int,
    model_prefix: str,
    min_compression: float = 3.2,
    auto_tune: bool = True,
):
    """Train BPE model (uncapped vocab). Auto-tune if requested.

    Auto-tune strategy (uncapped): probe progressively larger requested sizes
    and select by priority: compression >= target first, then highest vocab.
    """

    print("Starting training...")
    assert os.path.exists(corpus_path), f"Corpus file not found: {corpus_path}"

    initial_requested = vocab_size
    requested_list = [vocab_size]
    if auto_tune:
        probe = [max(2000, vocab_size), 6000, 8000, 10000, 16000, 24000, 32000]
        requested_list = list(dict.fromkeys([vocab_size] + probe))

    best = {
        "tokenizer": None,
        "stats": None,
        "requested_vocab": None,
        "actual_vocab": -1,
        "compression": -1.0,
    }

    last_actual = -1
    plateau_count = 0
    for req in requested_list:
        print(f"- Training with requested vocab={req}...")
        tok = _train_on_corpus(corpus_path, req)
        stats = calculate_stats(corpus_path, tok)
        actual = stats["vocab_size"]
        comp = stats["compression_ratio"]
        print(f"  -> actual_vocab={actual}, compression={comp:.3f}")

        # Track plateau to avoid expensive extra probes on tiny corpora
        if actual == last_actual:
            plateau_count += 1
        else:
            plateau_count = 0
        last_actual = actual

        # Candidate selection: prefer those meeting compression target, then highest vocab
        def is_better(a, b):
            if a is None:
                return True
            a_meets = a["compression"] >= min_compression
            b_meets = b["compression"] >= min_compression
            if a_meets and not b_meets:
                return True
            if not a_meets and b_meets:
                return False
            # Both meet or both miss: prefer higher compression then vocab size
            if a["compression"] != b["compression"]:
                return a["compression"] > b["compression"]
            return a["actual_vocab"] > b["actual_vocab"]

        current = {
            "tokenizer": tok,
            "stats": stats,
            "requested_vocab": req,
            "actual_vocab": actual,
            "compression": comp,
        }

        if is_better(current, best):
            best = current

        # If we've plateaued twice and we're near the top, stop early
        if plateau_count >= 2 and req >= 4900:
            print("  -> Plateau detected at high requested vocab. Stopping early.")
            break

    assert best["tokenizer"] is not None, "Training failed to produce a model"

    # Save best model and stats
    model_path = f"{model_prefix}.json"
    best["tokenizer"].save(model_path)

    # Enrich stats with requirement checks and metadata
    stats = best["stats"]
    stats.update({
        "initial_requested_vocab": initial_requested,
        "requested_vocab": best["requested_vocab"],
        "meets_compression_target": stats["compression_ratio"] >= min_compression,
        "compression_target": min_compression,
        "vocab_cap_mode": "UNCAPPED",
    })

    stats_file = f"{model_prefix}_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nBest Model Statistics:")
    print(f"Initial Requested Vocab: {stats['initial_requested_vocab']}")
    print(f"Best Candidate Requested Vocab: {stats['requested_vocab']}")
    print(f"Vocabulary Size: {stats['vocab_size']} tokens")
    print(f"Compression Ratio: {stats['compression_ratio']:.3f} chars/token")
    print("Requirement Checks:")
    print("- Vocab cap: UNCAPPED (no check)")
    print(
        f"- Compression >= {min_compression}: "
        f"{'PASS' if stats['meets_compression_target'] else 'FAIL'}"
    )


def calculate_stats(corpus_path: str, tokenizer_or_path) -> dict:
    tokenizer = tokenizer_or_path if isinstance(tokenizer_or_path, Tokenizer) else Tokenizer.from_file(tokenizer_or_path)
    total_chars = 0
    total_tokens = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_chars += len(line)
            encoding = tokenizer.encode(line)
            total_tokens += len(encoding.ids)
    vocab_size = tokenizer.get_vocab_size()
    return {
        "vocab_size": vocab_size,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "compression_ratio": (total_chars / total_tokens) if total_tokens > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train BPE model (uncapped vocab)")
    parser.add_argument("--config", help="Optional JSON config file to override internal settings")
    parser.add_argument("--input", required=False, help="Input corpus file (highest precedence)")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Requested vocabulary size (will not exceed 4999). Overrides config if provided",
    )
    parser.add_argument("--model-prefix", required=False, help="Model prefix (highest precedence)")
    parser.add_argument(
        "--min-compression",
        type=float,
        default=None,
        help="Minimum required compression ratio (chars/token). Overrides config",
    )
    parser.add_argument(
        "--no-auto-tune",
        action="store_true",
        help="Disable auto-tuning of vocab size",
    )
    # --no-vocab-cap retained for backward compatibility; now always uncapped
    parser.add_argument("--no-vocab-cap", action="store_true", help="(Deprecated) Vocab already uncapped")
    args = parser.parse_args()

    # Load config if provided
    cfg = {}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"Error reading config {args.config}: {e}")
            sys.exit(1)

    # Resolve effective parameters with precedence: CLI > Config JSON > Internal defaults
    effective = dict(INTERNAL_DEFAULTS)
    effective.update(USER_CONFIG)
    effective.update(cfg)

    corpus_path = args.input if args.input else effective.get("input")
    vocab_size = args.vocab_size if args.vocab_size is not None else effective.get("vocab_size")
    model_prefix = args.model_prefix if args.model_prefix else effective.get("model_prefix")
    min_compression = (
        args.min_compression if args.min_compression is not None else effective.get("min_compression", 3.2)
    )
    auto_tune = effective.get("auto_tune", True)
    if args.no_auto_tune:
        auto_tune = False
    enforce_vocab_cap = False  # permanently uncapped

    missing = []
    if not corpus_path:
        missing.append("--input or config['input']")
    if not model_prefix:
        missing.append("--model-prefix or config['model_prefix']")
    if missing:
        print("Missing required parameters: " + ", ".join(missing))
        parser.print_help()
        sys.exit(1)

    # No cap check
    
    print("Using configuration:")
    print(json.dumps({
        "input": corpus_path,
        "vocab_size": int(vocab_size),
        "model_prefix": model_prefix,
        "min_compression": float(min_compression),
        "auto_tune": bool(auto_tune),
        "vocab_cap_mode": "UNCAPPED"
    }, ensure_ascii=False, indent=2))

    train_bpe_model(
        corpus_path=corpus_path,
        vocab_size=int(vocab_size),
        model_prefix=model_prefix,
        min_compression=float(min_compression),
        auto_tune=bool(auto_tune),
    )


if __name__ == "__main__":
    main()