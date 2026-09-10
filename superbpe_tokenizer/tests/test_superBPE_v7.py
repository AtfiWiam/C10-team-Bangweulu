# -*- coding: utf-8 -*-
"""test_superBPE_v7.py

Tokenization Benchmark for SuperBPE v7:
----------------------------------------
-Automated evaluation and validation suite for the SuperBPE tokenizer pipeline.

Evaluation Metrics & Guarantees:
--------------------------------
1. Compression Ratio:
   - Atomic Characters per Token: Measures true linguistic compression where each
     Unicode marker ([U+XXXX NAME]) counts as exactly 1 character.
   - Words per Token (Inverse Fertility): Measures lexical absorption capacity.
2. Throughput Performance:
   - Encoding Speed: Evaluated in words/second and atomic characters/second
     using Backward DAG Shortest-Path Dynamic Programming.
   - Decoding Speed: Evaluated in words/second using direct array table lookups.
3. Lossless Reconstruction Guarantee:
   - Verifies 100% exact equality: decode(encode(text)) == text across all
     cases, whitespace runs, punctuation marks, and non-ASCII Unicode markers.
"""

import os
import sys
import glob
import time
import argparse
from typing import Optional, List, Dict, Any

from tokenizer import Tokenizer
from superbpe_v7 import (
    atomic_length,
    preprocess_text,
    load_file,
)


def find_benchmark_txt_files(
    target_path: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> List[str]:
    """Resolves target .txt files across candidate directories for benchmarking.

    Discovery Strategy:
    1. If target_path is an existing file, returns it directly as a single-element list.
    2. If target_path is a directory, discovers all .txt files inside it.
    3. If target_path is a filename or glob pattern, searches across base_dir,
       script directory, current working directory, and parent directory.
    4. If target_path is None, automatically discovers all .txt files located
       in the search paths.

    Args:
        target_path: Optional file path, directory path, or glob pattern.
        base_dir: Optional base directory to prioritize during resolution.

    Returns:
        Sorted list of unique absolute paths to valid .txt benchmark files.
    """
    
    # 1. Establish Ordered, Deduplicated Search Directories
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    search_dirs = [
        base_dir if base_dir and os.path.isdir(base_dir) else None,
        script_dir,
        os.getcwd(),
        os.path.dirname(script_dir),
    ]
    search_dirs = [os.path.abspath(d) for d in search_dirs if d and os.path.isdir(d)]
    
    # Deduplicate search dirs preserving order
    seen_dirs = set()
    unique_dirs = []
    for d in search_dirs:
        if d not in seen_dirs:
            seen_dirs.add(d)
            unique_dirs.append(d)

    # 2. Resolve Explicit Target (File, Directory, or Pattern)
    if target_path:
        # Check if direct file path exists
        if os.path.isfile(target_path):
            return [os.path.abspath(target_path)]

        # Check if directory path exists
        if os.path.isdir(target_path):
            txts = [
                os.path.abspath(os.path.join(target_path, f))
                for f in sorted(os.listdir(target_path))
                if f.lower().endswith(".txt") and os.path.isfile(os.path.join(target_path, f))
            ]
            if txts:
                return txts

        # Search by filename or glob pattern in search dirs
        for s_dir in unique_dirs:
            candidate = os.path.join(s_dir, target_path)
            if os.path.isfile(candidate):
                return [os.path.abspath(candidate)]
            # Try glob
            matched = glob.glob(os.path.join(s_dir, target_path))
            matched_txts = [os.path.abspath(f) for f in matched if f.lower().endswith(".txt") and os.path.isfile(f)]
            if matched_txts:
                return sorted(matched_txts)

    # 3. Fallback: Discover All .txt Files in Search Directories
    found_files = []
    seen_fpaths = set()
    for s_dir in unique_dirs:
        try:
            for fname in sorted(os.listdir(s_dir)):
                if fname.lower().endswith(".txt"):
                    fpath = os.path.abspath(os.path.join(s_dir, fname))
                    if os.path.isfile(fpath) and fpath not in seen_fpaths:
                        seen_fpaths.add(fpath)
                        found_files.append(fpath)
        except (OSError, PermissionError):
            continue

    return found_files


def benchmark_single_file(
    tok: Tokenizer,
    file_path: str,
    repo_dir: str,
    sample_ratio: float = 1.0,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """Benchmarks tokenization performance and lossless recovery on a text file.

    Evaluates:
    - Preprocessed sentence chunk ingestion.
    - Timed encoding using Backward DAG Shortest-Path DP.
    - Timed decoding using dense array lookup tables.
    - Strict lossless reconstruction: decoded string must match original normalized input.
    - Metrics: characters/token, words/token, words/sec, and characters/sec.

    Args:
        tok: Initialized Tokenizer instance with loaded vocabulary.
        file_path: Absolute path to the .txt file being evaluated.
        repo_dir: Repository root path used to compute relative display names.
        sample_ratio: Fraction of lines to sample (between 0.0 and 1.0).
        seed: Random seed for deterministic subsampling when sample_ratio < 1.0.

    Returns:
        Dictionary containing granular performance metrics and test status,
        or None if the file is empty or contains no valid text lines.
    """

    rel_name = os.path.relpath(file_path, repo_dir)
    all_chunks = load_file(file_path)
    if not all_chunks:
        return None

    # Apply standard competition normalization pipeline
    prep_chunks = [preprocess_text(c) for c in all_chunks]
    prep_chunks = [p for p in prep_chunks if p]
    num_chunks = len(prep_chunks)
    if num_chunks == 0:
        return None

    # Optional deterministic subsampling for quick local sanity checks
    if sample_ratio < 1.0:
        import random
        random.seed(seed)
        sample_size = max(1, int(num_chunks * sample_ratio))
        eval_chunks = random.sample(prep_chunks, sample_size)
    else:
        sample_size = num_chunks
        eval_chunks = prep_chunks

    # Timed Encoding
    t0_enc = time.perf_counter()
    encoded_batch = tok.encode(eval_chunks)
    t1_enc = time.perf_counter()

    # Timed Decoding
    t0_dec = time.perf_counter()
    decoded_batch = tok.decode(encoded_batch)
    t1_dec = time.perf_counter()

    enc_time = max(0.0001, t1_enc - t0_enc)
    dec_time = max(0.0001, t1_dec - t0_dec)

    # Lossless Verification & Metric Aggregation
    total_words = sum(len(c.split()) for c in eval_chunks)
    total_chars_in = 0
    total_chars_out = 0
    total_tokens = 0
    matches = 0

    for orig, enc, dec in zip(eval_chunks, encoded_batch, decoded_batch):
        c_in = atomic_length(orig)
        c_out = atomic_length(dec)
        t_count = len(enc)

        total_chars_in += c_in
        total_chars_out += c_out
        total_tokens += t_count

        if orig == dec:
            matches += 1

    passed = (matches == sample_size) and (total_chars_in == total_chars_out)
    chars_per_token = total_chars_in / max(1, total_tokens)
    words_per_token = total_words / max(1, total_tokens)
    encode_wps = total_words / enc_time
    encode_cps = total_chars_in / enc_time
    decode_wps = total_words / dec_time

    return {
        "file_path": file_path,
        "name": rel_name,
        "sentences": sample_size,
        "total_words": total_words,
        "chars_in": total_chars_in,
        "chars_out": total_chars_out,
        "tokens": total_tokens,
        "chars_per_token": chars_per_token,
        "words_per_token": words_per_token,
        "encode_time": enc_time,
        "decode_time": dec_time,
        "encode_wps": encode_wps,
        "encode_cps": encode_cps,
        "decode_wps": decode_wps,
        "passed": passed,
    }


def run_benchmark(
    tokenizer_path: str = "tokenizer.json",
    target_file: Optional[str] = None,
    data_dir: Optional[str] = None,
    sample_ratio: float = 1.0,
) -> None:
    """Executes the complete benchmark suite across target corpora.

    Workflow:
    1. Loads vocabulary from tokenizer.json.
    2. Identifies all target text corpora.
    3. Benchmarks each corpus individually (compression ratio, speed, lossless round-trip).
    4. Outputs individual diagnostics and an aligned summary table.
    5. Aggregates overall throughput, compression, and verification status.

    Args:
        tokenizer_path: Path to the tokenizer JSON vocabulary file.
        target_file: Optional path or pattern identifying specific benchmark files.
        data_dir: Optional directory to search for .txt files.
        sample_ratio: Fraction of sentences to evaluate (default: 1.0).
    """
    
    print("\n" + "=" * 92)
    header_target = target_file if target_file else "ALL DISCOVERED .TXT FILES"
    print(f"  SUPERBPE V5 — TOKENIZATION BENCHMARK [{header_target}]")
    print("=" * 92)

    # 1. Initialize Tokenizer Model
    if not os.path.exists(tokenizer_path):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
        candidate = os.path.join(script_dir, tokenizer_path)
        if os.path.exists(candidate):
            tokenizer_path = candidate
        else:
            print(f"[ERROR] Tokenizer JSON file not found at: {tokenizer_path}")
            print("Please train tokenizer with 'python superbpe_v5.py' first.")
            return

    tok = Tokenizer(tokenizer_path)
    print(f"Loaded Tokenizer: {tokenizer_path} (Vocabulary Size: {len(tok.token_to_id):,} tokens)")

    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    repo_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "SuperBPE" else script_dir

    # 2. Identify Benchmark Target Corpora
    target_files = find_benchmark_txt_files(target_path=target_file, base_dir=data_dir)
    if not target_files:
        print(f"[WARNING] No .txt files found to benchmark.")
        return

    print(f"Benchmark Target File(s) ({len(target_files)}):")
    for f in target_files:
        print(f"  - {os.path.relpath(f, repo_dir)}")
    print("-" * 92)

    # 3. Execute Benchmarking Across Files
    overall_words = 0
    overall_chars = 0
    overall_tokens = 0
    overall_enc_time = 0.0
    overall_dec_time = 0.0
    all_passed = True
    results = []

    for fpath in target_files:
        res = benchmark_single_file(
            tok=tok,
            file_path=fpath,
            repo_dir=repo_dir,
            sample_ratio=sample_ratio,
        )
        if not res:
            continue

        results.append(res)
        overall_words += res["total_words"]
        overall_chars += res["chars_in"]
        overall_tokens += res["tokens"]
        overall_enc_time += res["encode_time"]
        overall_dec_time += res["decode_time"]
        if not res["passed"]:
            all_passed = False

        status_tag = "PASSED" if res["passed"] else "FAILED"
        print(f"[{status_tag}] {res['name']}")
        print(f"       Corpus: {res['sentences']:,} sentences | {res['total_words']:,} words | {res['chars_in']:,} atomic chars")
        print(f"       Tokens Emitted: {res['tokens']:,} tokens")
        print(f"       Compression:    {res['chars_per_token']:.2f} chars/token ({res['words_per_token']:.2f} words/token)")
        print(f"       Encode Speed:   {res['encode_wps']:,.0f} words/sec ({res['encode_cps']:,.0f} chars/sec) in {res['encode_time']:.3f}s")
        print(f"       Decode Speed:   {res['decode_wps']:,.0f} words/sec in {res['decode_time']:.3f}s\n")

    if not results:
        print("[WARNING] No valid sentences processed.")
        return

    # 4. Formatted Tabular Summary & Aggregate Results
    print("=" * 92)
    print(f"{'Dataset File':<38} {'Words':>10} {'Chars':>12} {'Tokens':>10} {'Char/Tok':>10} {'Speed (w/s)':>14}")
    print("-" * 92)
    for res in results:
        print(f"{res['name']:<38} {res['total_words']:>10,d} {res['chars_in']:>12,d} {res['tokens']:>10,d} {res['chars_per_token']:>10.2f} {res['encode_wps']:>14,.0f}")
    print("-" * 92)

    overall_cpt = overall_chars / max(1, overall_tokens)
    overall_wpt = overall_words / max(1, overall_tokens)
    overall_wps = overall_words / max(0.0001, overall_enc_time)
    overall_cps = overall_chars / max(0.0001, overall_enc_time)

    print(f"\nTOTAL BENCHMARK RESULTS:")
    print(f"  Total Processed Words:       {overall_words:,} words")
    print(f"  Total Atomic Characters:     {overall_chars:,} chars (1 Unicode marker = 1 character)")
    print(f"  Total Tokens Generated:      {overall_tokens:,} tokens")
    print(f"  Overall Compression Ratio:   {overall_cpt:.2f} chars/token ({overall_wpt:.2f} words/token)")
    print(f"  Overall Encoding Throughput: {overall_wps:,.0f} words/sec ({overall_cps:,.0f} chars/sec)")
    print(f"  Reconstruction Accuracy:     {'100% EXACT LOSSLESS ROUND-TRIP (ALL PASSED)' if all_passed else 'FAILED'}")
    print("=" * 92)


def main() -> None:
    """CLI entry point for running the SuperBPE tokenization benchmark."""
    parser = argparse.ArgumentParser(
        description="SuperBPE v5 Tokenization Benchmark — Evaluates Compression Ratio, Speed, and Lossless Reconstruction"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Optional .txt file name, path, or pattern to benchmark (e.g. 'languages.txt' or 'data.txt'). If omitted, benchmarks all .txt files found in directory.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer.json",
        help="Path to tokenizer JSON vocabulary file (default: tokenizer.json)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory to search for .txt files",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Fraction of sentences to benchmark (default: 1.0 for entire corpus)",
    )
    args = parser.parse_args()

    run_benchmark(
        tokenizer_path=args.tokenizer,
        target_file=args.target,
        data_dir=args.dir,
        sample_ratio=args.sample_ratio,
    )


if __name__ == "__main__":
    main()
