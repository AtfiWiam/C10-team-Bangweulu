# -*- coding: utf-8 -*-
"""superbpe_v7.py

Production SuperBPE Tokenizer Trainer for African Languages.

Architecture Overview:
----------------------
1. Linguistic Typology Balancing:
   - Partitions vocabulary into three distinct morphological classes:
     * Agglutinative (Bantu, Cushitic): 80% Stage 1 subwords / 20% Stage 2 superwords.
     * Fusional (Semitic, Chadic, Romance): 65% Stage 1 subwords / 35% Stage 2 superwords.
     * Analytic (Kwa, Mande, Pidgin): 40% Stage 1 subwords / 60% Stage 2 superwords.

2. Canonical Two-Stage SuperBPE Curriculum (Liu et al., 2025):
   - Stage 1 (Intra-Word BPE): Whitespace pre-tokenization restricts merges
     to word boundaries, constructing productive morphemes, affixes, and stems.
   - Stage 2 (Cross-Word SuperBPE): Whitespace pre-tokenization barrier is lifted,
     allowing iterative merges across spaces to capture multi-word idioms and compounds.

3. Strict Invariant Guarantees:
   - Unicode Marker Atomicity: Non-ASCII markers ([U+XXXX NAME]) are treated
     as indivisible single alphabet symbols (atomic length = 1).
   - Zero Casing Waste: Canonical lowercasing index ensures exactly one case
     variant is retained per multi-word collocation.
   - Zero Spacing Waste: Superwords are strictly learned with a leading space
     (' w1 w2'), preventing duplicate unspaced vocabulary pollution.
"""

import os
import sys
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict

# ============================================================================
# 1. CONFIGURATION & ATOMIC SYMBOLS
# ============================================================================

DEFAULT_VOCAB_SIZE = 20_000
ASCII_START = 32
ASCII_END = 127
INITIAL_ASCII_VOCAB_SIZE = ASCII_END - ASCII_START

# Morphological Typology Budget Partitions (Sum = 1.0)
TYPOLOGY_BUDGET_RATIOS = {
    "agglutination": 0.35,  # Bantu, Cushitic, etc.
    "fusional": 0.35,       # Semitic, Chadic, Romance, etc.
    "analytic": 0.30,       # Kwa, Mande, Pidgin, Wolof, etc.
}

# Two-Stage SuperBPE Curriculum Ratio: Stage 1 (Intra-word) vs Stage 2 (Cross-word)
TYPOLOGY_STAGE1_RATIOS = {
    "agglutination": 0.80,  # 80% subwords/morphemes, 20% Cross-words
    "fusional":      0.65,  # 65% subwords/stems, 35% Cross-words
    "analytic":      0.40,  # 40% subwords, 60% Cross-words (compounds & phrases)
}

# Maximum phrase length (in whitespace-delimited words) per typology
TYPOLOGY_MAX_WORDS = {
    "agglutination": 2,
    "fusional": 3,
    "analytic": 4,
}

# Transliteration mappings for non-ASCII quotes, dashes, and Ethiopic/Arabic punctuation
PUNCTUATION_MAPPINGS = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2026": "...",
    "\u1361": " ", "\u1362": ".", "\u1363": ",", "\u1364": ";",
    "\u1365": ":", "\u1366": ":", "\u1367": "?", "\u1368": ".",
    "\u060C": ",", "\u061B": ";", "\u061F": "?", "\u06D4": ".", "\u0640": "",
}
PUNCTUATION_MAP = str.maketrans(PUNCTUATION_MAPPINGS)

# Preprocessing and token extraction regexes
URL_CLEAN_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_HANDLE_REGEX = re.compile(r"@\w+")
HASHTAG_REGEX = re.compile(r"#\w+")
WHITESPACE_REGEX = re.compile(r"\s+")
MARKER_PATTERN = re.compile(r"\[U\+([0-9A-F]{4,6})\s+[^\]]+\]")
PLACEHOLDER_REGEX = re.compile(r"__U_([0-9A-F]{4,6})__")
CLAUSE_SPLIT_REGEX = re.compile(r"[\r\n\t]+|[.!?…:;\"«»—–]+|(?<=\s)['’]|['’](?=\s)")
BAD_PUNCT_REGEX = re.compile(r"""[\"""«».,:;!?(){}\[\]\\/`~@#$%^&*+=<>|]""")

# Atomic unit regex: matches complete placeholder OR complete marker OR single character
ATOMIC_UNIT_REGEX = re.compile(r"__U_[0-9A-F]{4,6}__|\[U\+[0-9A-F]{4,6}\s+[^\]]+\]|.")

# Global cache for single-character Unicode marker transliteration
_UNICODE_MARKER_CACHE = {}

# ============================================================================
# 2. STRING PREPROCESSING & ATOMIC UNIT UTILITIES
# ============================================================================
def unicode_marker(ch: str) -> str:
    """
    Returns the standardized ASCII marker representation for a Unicode character.

    Args:
        ch: A single Unicode character.

    Returns:
        Formatted ASCII string representation, e.g., '[U+00E9 LATIN SMALL LETTER E WITH ACUTE]'.
    """
    marker = _UNICODE_MARKER_CACHE.get(ch)
    if marker is None:
        cp = ord(ch)
        name = unicodedata.name(ch, "UNKNOWN")
        marker = f"[U+{cp:04X} {name}]"
        _UNICODE_MARKER_CACHE[ch] = marker
    return marker


def extract_atomic_units(text: str) -> list[str]:
    """
    Tokenizes a string into indivisible atomic symbols.

    Ensures that Unicode marker placeholders (e.g. '__U_00E9__') or markers
    ('[U+00E9 ...]') are never sliced across their internal characters.

    Args:
        text: Input string.

    Returns:
        List of atomic string tokens, where each marker or ASCII character is 1 unit.
    """
    return ATOMIC_UNIT_REGEX.findall(text)


def atomic_length(text: str) -> int:
    """
    Computes the linguistic character length of a string.

    Guarantees that each Unicode marker counts as exactly 1 unit of length.

    Args:
        text: Input string.

    Returns:
        Total number of atomic units.
    """
    return len(extract_atomic_units(text))


def preprocess_text(text: str) -> str:
    """
    Normalizes raw corpus text.

    Operations:
    1. Removes URLs, @handles, and hashtags.
    2. Applies Unicode NFKC normalization.
    3. Translates non-ASCII punctuation (curly quotes, dashes, Ethiopic/Arabic dots).
    4. Converts remaining non-ASCII characters to explicit ASCII markers.
    5. Collapses whitespace runs to a single ASCII space and strips boundaries.

    Args:
        text: Raw text string.

    Returns:
        Cleaned, fully ASCII-compliant string.
    """
    if not isinstance(text, str):
        return ""
    text = URL_CLEAN_REGEX.sub(" ", text)
    text = USER_HANDLE_REGEX.sub(" ", text)
    text = HASHTAG_REGEX.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(PUNCTUATION_MAP)
    if not text.isascii():
        out = []
        for ch in text:
            if ord(ch) < ASCII_END:
                out.append(ch)
            else:
                out.append(unicode_marker(ch))
        text = "".join(out)
    text = WHITESPACE_REGEX.sub(" ", text)
    return text.strip()

# ============================================================================
# 3. MARKER MANAGER CLASS (PROTECTION & UNMASKING)
# ============================================================================
class MarkerManager:
    """
    Protects Unicode markers during BPE training by substituting space-free placeholders.

    Replaces verbose tokens like '[U+00E9 LATIN SMALL LETTER E WITH ACUTE]'
    with atomic tokens like '__U_00E9__' during training so that internal spaces
    do not interfere with whitespace pre-tokenization. Restores canonical strings
    when writing the final tokenizer vocabulary.
    """
    def __init__(self):
        """Initializes empty bidirectional placeholder mapping dictionaries."""
        self.marker_to_placeholder = {}
        self.placeholder_to_marker = {}

    def register_and_mask(self, text: str) -> str:
        """
        Replaces canonical markers in text with atomic placeholders.

        Args:
            text: Preprocessed text containing '[U+XXXX NAME]' markers.

        Returns:
            Text containing compact '__U_XXXX__' placeholders.
        """
        def repl(match):
            m = match.group(0)
            hex_code = match.group(1)
            ph = f"__U_{hex_code}__"
            self.marker_to_placeholder[m] = ph
            self.placeholder_to_marker[ph] = m
            return ph
        return MARKER_PATTERN.sub(repl, text)

    def mask_texts(self, texts: list[str]) -> list[str]:
        """
        Batch masks an entire collection of text documents.

        Args:
            texts: List of preprocessed strings.

        Returns:
            List of masked strings.
        """
        return [self.register_and_mask(t) for t in texts]

    def unmask(self, token: str) -> str:
        """
        Converts an atomic placeholder back to its canonical Unicode marker.

        Args:
            token: Learned token string (e.g., ' __U_00E9__cole').

        Returns:
            Fully unmasked string (e.g., ' [U+00E9 LATIN SMALL LETTER E WITH ACUTE]cole').
        """
        for ph, m in self.placeholder_to_marker.items():
            if ph in token:
                token = token.replace(ph, m)
        return token

    def unmask_vocab(self, vocab: dict[int, str]) -> dict[int, str]:
        """
        Unmasks all placeholder tokens in the learned vocabulary.

        Args:
            vocab: Mapping of token ID to masked token string.

        Returns:
            Mapping of token ID to canonical unmasked token string.
        """
        return {tid: self.unmask(tok) for tid, tok in vocab.items()}


# ============================================================================
# 4. TYPOLOGY CLASSIFICATION
# ============================================================================
TYPOLOGY_KEYWORDS = {
    "agglutination": [
        "agglutin", "bantu", "swahili", "kiswahili", "zulu", "xhosa", "shona",
        "luganda", "kinyarwanda", "kirundi", "sotho", "tswana", "chewa", "oromo", "somali"
    ],
    "fusional": [
        "fusion", "semitic", "amharic", "tigrinya", "arabic", "hausa", "french", "english", "portuguese"
    ],
    "analytic": [
        "analytic", "isolating", "yoruba", "igbo", "ewe", "akan", "twi", "pidgin", "wolof", "bambara"
    ],
}


def classify_dataset_name(filename: str) -> str:
    """
    Infers the language typology of a dataset from its filename.

    Args:
        filename: Name or path of the text file.

    Returns:
        Typology key: 'agglutination', 'fusional', or 'analytic'.
    """
    fn = filename.lower().replace("-", "_")
    for typo, kws in TYPOLOGY_KEYWORDS.items():
        if any(kw in fn for kw in kws):
            return typo
    return "fusional"


def classify_line_typology(text: str) -> str:
    """
    Infers typology dynamically from line content when using mixed corpora.

    Args:
        text: Masked line of text.

    Returns:
        Typology key: 'agglutination', 'fusional', or 'analytic'.
    """
    if any(k in text for k in ["__U_12", "__U_13", "__U_06", "[U+12", "[U+13", "[U+06"]):
        return "fusional"
    words = set(text.lower().split())
    if any(w in words for w in {"kuma", "wannan", "mutane", "akan", "zuwa", "tare", "the", "and", "les", "des"}):
        return "fusional"
    if any(k in text for k in ["__U_1E", "__U_01", "__U_02", "[U+1E", "[U+01", "[U+02"]):
        return "analytic"
    if any(w in words for w in {"awon", "ninu", "lati", "igbo", "ndi", "onye", "nke", "dey", "wetin", "fit", "ci", "bi", "mu"}):
        return "analytic"
    return "agglutination"


# ============================================================================
# 5. CORE BPE ENGINE (INVERTED INDEX / GREEDY MERGE)
# ============================================================================
def run_bpe(
    tokenized_sequences: dict[tuple[str, ...], int],
    target_merges: int,
    existing_vocab: set[str],
    is_stage2: bool = False,
    max_words: int = 3,
) -> list[str]:
    """
    Executes iterative Byte Pair Encoding using an inverted sequence index.

    Algorithm:
    1. Builds an inverted index mapping adjacent token pairs (A, B) -> sequence IDs.
    2. Maintains a frequency table of all adjacent pairs across the weighted corpus.
    3. Greedily selects the pair with the maximum token count reduction (ΔTokens = freq).
    4. Validates anti-redundancy rules (casing duplication guard, max word limit).
    5. Updates affected sequences in-place and increments/decrements pair frequencies.

    Args:
        tokenized_sequences: Mapping of token tuples to corpus occurrence frequencies.
        target_merges: Number of merge operations to perform.
        existing_vocab: Global set of registered vocabulary tokens (mutated in-place).
        is_stage2: True if running Stage 2 (SuperBPE across whitespace).
        max_words: Maximum word count permitted for multi-word superwords.

    Returns:
        List of newly created tokens in order of merge execution.
    """
    if target_merges <= 0 or not tokenized_sequences:
        return []

    # Inverted index: pair -> list of sequence IDs
    pair_counts = defaultdict(int)
    pair_to_seqs = defaultdict(set)

    seq_list = list(tokenized_sequences.keys())
    weights = list(tokenized_sequences.values())
    num_seqs = len(seq_list)

    # Populate initial inverted index
    for s_idx in range(num_seqs):
        seq = seq_list[s_idx]
        w = weights[s_idx]
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            pair_counts[p] += w
            pair_to_seqs[p].add(s_idx)

    learned_tokens = []
    seen_lower_phrases = set()

    # Pre-populate seen lower phrases from existing vocab
    for tok in existing_vocab:
        if " " in tok.strip():
            seen_lower_phrases.add(tok.strip().lower())

    while len(learned_tokens) < target_merges and pair_counts:
        # Find the pair that yields the largest absolute reduction in token count
        best_pair = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]

        if best_count < 2:
            break

        p0, p1 = best_pair
        del pair_counts[best_pair]

        cand_token = p0 + p1

        # Validation Rule 1: Skip tokens that already exist in vocabulary
        if cand_token in existing_vocab:
            continue

        # Validation Rule 2: Disallow trailing whitespace or pure numeric digit tokens
        if cand_token.endswith(" ") or cand_token.strip().isdigit():
            continue
        
        # Stage 2 (SuperBPE) Guards
        if is_stage2:
            words = cand_token.strip().split(" ")
            # Reject phrases exceeding typology word limit
            if len(words) > max_words:
                continue

            # Reject phrases containing isolated punctuation marks
            if BAD_PUNCT_REGEX.search(cand_token):
                continue
            
            # Zero Casing Duplication: Strictly 1 casing variant per multi-word Cross-word
            cand_lower = cand_token.strip().lower()
            if cand_lower in seen_lower_phrases:
                continue
            seen_lower_phrases.add(cand_lower)

        # Register valid token
        existing_vocab.add(cand_token)
        learned_tokens.append(cand_token)

        # In-place sequence update for affected corpus lines
        affected = list(pair_to_seqs.pop(best_pair, ()))
        for s_idx in affected:
            seq = seq_list[s_idx]
            w = weights[s_idx]
            n = len(seq)
            if n < 2:
                continue

            # Check if pair is actually in sequence
            has_pair = any(seq[i] == p0 and seq[i + 1] == p1 for i in range(n - 1))
            if not has_pair:
                continue

            # Decrement frequencies of old adjacent pairs
            for i in range(n - 1):
                p = (seq[i], seq[i + 1])
                pair_counts[p] -= w
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)

            # Reconstruct sequence with newly merged token
            new_seq = []
            i = 0
            while i < n:
                if i < n - 1 and seq[i] == p0 and seq[i + 1] == p1:
                    new_seq.append(cand_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq_list[s_idx] = tuple(new_seq)

            # Increment frequencies of new adjacent pairs
            m = len(new_seq)
            for i in range(m - 1):
                p = (new_seq[i], new_seq[i + 1])
                pair_counts[p] += w
                pair_to_seqs[p].add(s_idx)

    return learned_tokens


# ============================================================================
# 6. SUPERBPE TRAINER CLASS
# ============================================================================
class SuperBPETrainer:
    """Manages multi-typology corpus partitioning, training, and export."""

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE):
        """
        Initializes the trainer.

        Args:
            vocab_size: Maximum total vocabulary slots (default: 20,000).
        """
        self.vocab_size = vocab_size
        self.marker_mgr = MarkerManager()
        self.vocab = {}

    def _init_base_vocab(self, texts: list[str]) -> set[str]:
        """
        Initializes base symbols: printable ASCII [32..126] + frequent Unicode markers.

        Infrequent markers (freq < 3) decompose to base ASCII at encode time,
        freeing hundreds of vocabulary slots for productive linguistic units.

        Args:
            texts: List of masked corpus strings.

        Returns:
            Set of initial base alphabet tokens.
        """
        base = {chr(c) for c in range(ASCII_START, ASCII_END)}

        # Count frequencies of all Unicode marker placeholders
        all_text = " ".join(texts)
        found_phs = PLACEHOLDER_REGEX.findall(all_text)
        ph_counts = Counter(f"__U_{h}__" for h in found_phs)

        # Only register markers that appear >= 3 times
        for ph, cnt in ph_counts.items():
            if cnt >= 3:
                base.add(ph)

        print(f"[BASE] {INITIAL_ASCII_VOCAB_SIZE} ASCII chars + {len(base) - INITIAL_ASCII_VOCAB_SIZE} frequent markers.")
        return base

    def train_typology(
        self,
        typology: str,
        texts: list[str],
        total_budget: int,
        global_vocab: set[str],
    ) -> list[str]:
        
        """
        Executes the two-stage SuperBPE curriculum for a specific language family.

        Args:
            typology: Key indicating 'agglutination', 'fusional', or 'analytic'.
            texts: List of masked text documents belonging to this typology.
            total_budget: Vocabulary slot quota allocated to this typology.
            global_vocab: Shared set of learned tokens across all typologies.

        Returns:
            List of learned tokens for this typology.
        """

        if total_budget <= 0 or not texts:
            return []

        s1_ratio = TYPOLOGY_STAGE1_RATIOS[typology]
        stage1_budget = int(round(total_budget * s1_ratio))
        stage2_budget = total_budget - stage1_budget
        max_words = TYPOLOGY_MAX_WORDS[typology]

        print()
        print(f"=== TRAINING TYPOLOGY: [{typology.upper()}] ===")
        print(f"Total Budget: {total_budget:,} slots | Corpus: {len(texts):,} lines")
        print(f"  * Stage 1 (Intra-word BPE / Morphemes & Stems): {stage1_budget:,} merges ({s1_ratio*100:.0f}%)")
        print(f"  * Stage 2 (SuperBPE / Cross-word Cross-words):  {stage2_budget:,} merges ({(1-s1_ratio)*100:.0f}%, Max words: {max_words})")

        # --------------------------------------------------------------------
        # STAGE 1: INTRA-WORD BPE (Subwords, Morphemes, Lexical Words)
        # --------------------------------------------------------------------
        # Whitespace pre-tokenization enforces that merges stay INSIDE words
        word_freqs = Counter()
        for text in texts:
            for clause in CLAUSE_SPLIT_REGEX.split(text):
                words = clause.strip().split(" ")
                for idx, w in enumerate(words):
                    if not w or w.isdigit():
                        continue
                    # Attach leading space to represent mid-sentence words
                    surf = (" " + w) if idx > 0 else w
                    word_freqs[surf] += 1

        # Break words into atomic units (preserving Unicode markers intact)
        tokenized_words = {}
        for surf, freq in word_freqs.items():
            units = tuple(extract_atomic_units(surf))
            if len(units) >= 2:
                tokenized_words[units] = freq

        t0 = time.perf_counter()
        stage1_tokens = run_bpe(
            tokenized_sequences=tokenized_words,
            target_merges=stage1_budget,
            existing_vocab=global_vocab,
            is_stage2=False,
        )
        t1 = time.perf_counter()
        print(f"[{typology}] Stage 1 finished: {len(stage1_tokens):,}/{stage1_budget:,} subwords/words learned ({t1 - t0:.2f}s).")

        # --------------------------------------------------------------------
        # STAGE 2: SUPERBPE (Cross-Word Cross-words Bridging Whitespace)
        # --------------------------------------------------------------------
        # The whitespace barrier is LIFTED: tokenize text with Stage 1 vocabulary
        # and merge adjacent tokens across whitespace!
        
        def segment_clause(clause_str: str) -> list[str]:
            """Fast greedy segmentation of clauses with current Stage 1 vocabulary."""
            words = clause_str.strip().split(" ")
            tokens = []
            for idx, w in enumerate(words):
                if not w or w.isdigit():
                    continue
                surf = (" " + w) if idx > 0 else w
                units = extract_atomic_units(surf)
                curr = list(units)
                changed = True
                while changed:
                    changed = False
                    for i in range(len(curr) - 1):
                        cand = curr[i] + curr[i + 1]
                        if cand in global_vocab:
                            curr[i] = cand
                            del curr[i + 1]
                            changed = True
                            break
                tokens.extend(curr)
            return tokens

        clause_freqs = Counter()
        for text in texts:
            for clause in CLAUSE_SPLIT_REGEX.split(text):
                if len(clause.split(" ")) < 2:
                    continue
                c_toks = tuple(segment_clause(clause))
                if len(c_toks) >= 2:
                    clause_freqs[c_toks] += 1

        # Filter low frequency clauses for fast Stage 2 training
        filtered_clauses = {c: f for c, f in clause_freqs.items() if f >= 2}

        t2 = time.perf_counter()
        stage2_tokens = run_bpe(
            tokenized_sequences=filtered_clauses,
            target_merges=stage2_budget,
            existing_vocab=global_vocab,
            is_stage2=True,
            max_words=max_words,
        )
        t3 = time.perf_counter()
        print(f"[{typology}] Stage 2 finished: {len(stage2_tokens):,}/{stage2_budget:,} Cross-words learned ({t3 - t2:.2f}s).")

        return stage1_tokens + stage2_tokens

    def train(self, corpus_by_file: dict[str, list[str]]) -> None:
        """
        Coordinates multi-file ingestion, budget quotas, and vocabulary assembly.

        Args:
            corpus_by_file: Dictionary mapping filename to raw text lines.
        """

        print("=" * 80)
        print("SUPERBPE V7: CURRICULUM-DRIVEN MULTILINGUAL TOKENIZATION")
        print("=" * 80)

        typology_texts = {"agglutination": [], "fusional": [], "analytic": []}
        all_texts = []

        # 1. Ingest and preprocess all documents
        for fname, lines in corpus_by_file.items():
            prep_lines = [preprocess_text(line) for line in lines if line.strip()]
            prep_lines = self.marker_mgr.mask_texts(prep_lines)
            if not prep_lines:
                continue

            all_texts.extend(prep_lines)
            assigned_typo = classify_dataset_name(fname)

            # Route lines by content if single file or multi-language dataset
            if len(corpus_by_file) == 1 or "all" in fname.lower() or "corpus" in fname.lower():
                print(f"[INGEST] Partitioning mixed dataset '{fname}' by line typology...")
                for line in prep_lines:
                    t = classify_line_typology(line)
                    typology_texts[t].append(line)
            else:
                print(f"[INGEST] File '{fname}' -> Assigned to [{assigned_typo}] ({len(prep_lines):,} lines)")
                typology_texts[assigned_typo].extend(prep_lines)

        # Fallback safeguard for empty categories
        for t in typology_texts:
            if not typology_texts[t]:
                typology_texts[t] = list(all_texts)

        # 2. Initialize base alphabet & calculate typology slot budgets
        base_tokens = self._init_base_vocab(all_texts)
        total_learned_budget = self.vocab_size - len(base_tokens)

        budgets = {}
        alloc_sum = 0
        typologies = list(TYPOLOGY_BUDGET_RATIOS.keys())
        for t in typologies[:-1]:
            q = int(round(total_learned_budget * TYPOLOGY_BUDGET_RATIOS[t]))
            budgets[t] = q
            alloc_sum += q
        budgets[typologies[-1]] = max(0, total_learned_budget - alloc_sum)

        global_vocab = set(base_tokens)
        master_learned = []

        # 3. Train each typology curriculum
        for t in ["agglutination", "fusional", "analytic"]:
            learned = self.train_typology(t, typology_texts[t], budgets[t], global_vocab)
            master_learned.extend(learned)

        # 4. Sorting Invariant: Longest by ATOMIC length -> Smallest Token ID (ID 0)
        # Enables the Trie DAG shortest-path DP to explore longest paths first
        sorted_learned = sorted(
            list(dict.fromkeys(master_learned)),
            key=lambda x: (-atomic_length(x), -len(x), x)
        )

        final_vocab = {}
        curr_id = 0

        # Assign learned tokens to lower IDs
        for tok in sorted_learned[:total_learned_budget]:
            final_vocab[curr_id] = tok
            curr_id += 1

        # Assign base Unicode markers to middle IDs
        markers = sorted([t for t in base_tokens if t.startswith("__U_")])
        for m in markers:
            if m not in final_vocab.values() and curr_id < self.vocab_size:
                final_vocab[curr_id] = m
                curr_id += 1

        # Assign base Printable ASCII [32..126] to final IDs (fallback guarantee)
        for c in range(ASCII_START, ASCII_END):
            ch = chr(c)
            if ch not in final_vocab.values() and curr_id < self.vocab_size:
                final_vocab[curr_id] = ch
                curr_id += 1

        self.vocab = final_vocab
        print()
        print("=" * 80)
        print(f"SUPERBPE VOCABULARY READY: {len(self.vocab):,} / {self.vocab_size:,} slots saturated.")
        print("=" * 80)

    def save(self, path: str = "tokenizer.json") -> None:
        """Restores canonical Unicode markers and exports JSON mapping.

        Format: {"token_string": token_id} matching the contract of tokenizer.py.

        Args:
            path: Destination file path (default: 'tokenizer.json').
        """

        unmasked = self.marker_mgr.unmask_vocab(self.vocab)
        vocab_dict = {tok: tid for tid, tok in unmasked.items()}
        with open(path, "w", encoding="ascii") as f:
            json.dump(vocab_dict, f, ensure_ascii=True, indent=2)
        print(f"[SUCCESS] Saved SuperBPE tokenizer JSON to: {path}")


# ============================================================================
# 7. DIRECTORY SCANNER & COMMAND LINE ENTRY POINT
# ============================================================================
def discover_txt_files(data_dir: str = None) -> dict[str, str]:
    """Scans the specified directory and current working directory for .txt files.

    Args:
        data_dir: Optional explicit directory path to search.

    Returns:
        Dictionary mapping filename to absolute file path.
    """

    search_dirs = []
    if data_dir and os.path.isdir(data_dir):
        search_dirs.append(os.path.abspath(data_dir))
    search_dirs.append(os.getcwd())

    files = {}
    for d in search_dirs:
        try:
            for fname in sorted(os.listdir(d)):
                if fname.lower().endswith(".txt"):
                    fpath = os.path.join(d, fname)
                    if os.path.isfile(fpath) and fname not in files:
                        files[fname] = fpath
        except Exception:
            continue
    return files


def load_file(path: str) -> list[str]:
    """Safely reads a UTF-8 text file, replacing invalid byte sequences.

    Args:
        path: Path to the target text file.

    Returns:
        List of non-empty stripped lines.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    """CLI Entry point for SuperBPE model training."""
    import argparse
    parser = argparse.ArgumentParser(description="SuperBPE v7 Trainer")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--output", type=str, default="tokenizer.json")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing .txt datasets")
    args = parser.parse_args()

    txt_files = discover_txt_files(args.data_dir)
    if not txt_files:
        print("[ERROR] No .txt files found in directory.")
        sys.exit(1)

    print(f"[FILES] Discovered {len(txt_files)} dataset file(s): {list(txt_files.keys())}")
    corpus = {}
    for fname, fpath in txt_files.items():
        lines = load_file(fpath)
        if lines:
            corpus[fname] = lines

    trainer = SuperBPETrainer(vocab_size=args.vocab_size)
    t0 = time.perf_counter()
    trainer.train(corpus)
    t1 = time.perf_counter()
    print(f"[BENCHMARK] SuperBPE training completed in {t1 - t0:.2f} seconds.")
    trainer.save(args.output)


if __name__ == "__main__":
    main()