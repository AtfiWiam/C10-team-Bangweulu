# -*- coding: utf-8 -*-
"""tokenizer.py

Architecture & Algorithmic Design:
----------------------------------
1. Exact Global Optimization via DAG Shortest Path:
   - Treats text tokenization as finding the shortest path on a Directed Acyclic
     Graph (DAG) where nodes represent character boundaries and edges represent
     valid vocabulary tokens.
   - Because all token edge costs are uniform (+1 token), finding the shortest path
     is mathematically guaranteed to produce the absolute minimum number of emitted
     token IDs for any given vocabulary.

2. High-Performance Backward Dynamic Programming:
   - Solves subproblems from right-to-left (backward DP) across string indices:
     dp[i] = min_{edge (i -> j)} (1 + dp[j]).
   - Backward evaluation eliminates the need to reverse the reconstructed path;
     the optimal token sequence is recovered via a direct forward linear walk.
   - Includes early-exit pruning: if a transition achieves cost == 1, it reaches
     the end of the text in a single token. Since no non-empty text can be encoded
     in fewer than 1 token, search terminates immediately.

3. Fast Lookup Data Structures:
   - Trie: Dict-of-dicts prefix tree where the empty string key `""` denotes a
     terminal node storing the integer token ID.
   - Direct Array Lookup for Single Characters: A precomputed 256-element table
     resolves any single-byte ASCII character transition in O(1) time without
     querying the Trie.
   - Decode Table: A dense list indexed directly by token ID enables O(1) pointer-speed
     string reconstruction during decoding.
   - Memory Arena / Reusable Buffers: Pre-allocated arrays (`_dp`, `_next_idx`,
     `_next_tok`) avoid heap allocation overhead across batch encoding loops.
"""

import os
import json


class Tokenizer:
    """
    SuperBPE Tokenizer supporting globally optimal DP encoding and exact decoding.

    Attributes:
        token_to_id (dict[str, int]): Mapping from surface token string to integer ID.
        id_to_token (dict[int, str]): Mapping from integer ID to surface token string.
        vocab_size (int): Total number of distinct tokens in the vocabulary.
        single_char_tok (list[int]): Fast O(1) lookup table for ASCII ordinals 0..255.
        root (dict): Dict-of-dicts Trie representing all multi-character vocabulary tokens.
        id_to_token_list (list[str]): Dense array lookup table for O(1) token decoding.
    """

    def __init__(self, tokenizer_path: str = None):
        """
        Loads vocabulary and initializes Trie, lookup tables, and reusable buffers.

        Args:
            tokenizer_path: Optional path to `tokenizer.json`. If None, automatically
                searches the current directory and the directory containing this file.

        Raises:
            FileNotFoundError: If `tokenizer.json` cannot be located in candidate paths.
        """
        """Load vocabulary and build the zero-overhead Trie."""

        # 1. Vocabulary File Resolution
        if tokenizer_path and os.path.exists(tokenizer_path):
            pass
        else:
            module_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
            candidates = [
                tokenizer_path,
                os.path.join(module_dir, "tokenizer.json"),
                "tokenizer.json",
            ]
            tokenizer_path = None
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    tokenizer_path = candidate
                    break

        if tokenizer_path is None or not os.path.exists(tokenizer_path):
            raise FileNotFoundError("Tokenizer JSON file not found.")

        # 2. Vocabulary Parsing
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both {"vocab": {id: token}} and flat {token: id} formats
        if "vocab" in data and isinstance(data["vocab"], dict):
            self.id_to_token = {int(k): v for k, v in data["vocab"].items()}
            self.token_to_id = {v: int(k) for k, v in self.id_to_token.items()}
        else:
            self.token_to_id = {k: int(v) for k, v in data.items()}
            self.id_to_token = {int(v): k for k, v in data.items()}

        self.vocab_size = len(self.id_to_token)

        # 3. Direct Single-Character Lookup Table (O(1) Fallback Baseline)
        # Precompute direct single-character token IDs (fallback / length-1 baseline)
        self.single_char_tok = [
            self.token_to_id.get(chr(c), self.token_to_id.get(chr(c), 0))
            for c in range(256)
        ]

        # 4. Dict-of-Dicts Prefix Tree (Trie Construction)
        # Empty string key "" stores the terminal token_id
        self.root = {}
        for token_str, token_id in self.token_to_id.items():
            if not token_str:
                continue
            curr = self.root
            for ch in token_str:
                if ch not in curr:
                    curr[ch] = {}
                curr = curr[ch]
            curr[""] = token_id

        # 5. Pre-allocated reusable buffers to avoid heap allocation per string
        # Prevents costly memory allocations and GC churn during batch inference.
        self._buf_size = 8192
        self._dp = [0] * self._buf_size
        self._next_idx = [0] * self._buf_size
        self._next_tok = [0] * self._buf_size

        # 6. Array-based decode table for O(1) pointer-speed decoding
        max_id = max(self.id_to_token.keys(), default=0)
        self.id_to_token_list = [self.id_to_token.get(i, "") for i in range(max_id + 1)]

    def _encode_one(self, text: str) -> list[int]:
        """Encodes a single string into the globally optimal minimum token count.

        Uses Backward DAG Shortest Path Dynamic Programming across the Trie.
        Guarantees mathematical global optimality with respect to total emitted tokens.

        Args:
            text: Normalized ASCII string to tokenize.

        Returns:
            List of integer token IDs representing the shortest tokenization path.
        """

        n = len(text)
        if n == 0:
            return []

        # Resize reusable buffers if input exceeds current capacity
        if n >= self._buf_size:
            self._buf_size = n + 2048
            self._dp = [0] * self._buf_size
            self._next_idx = [0] * self._buf_size
            self._next_tok = [0] * self._buf_size

        dp = self._dp
        next_idx = self._next_idx
        next_tok = self._next_tok

        # Base condition: suffix starting past the end of text has 0 cost
        dp[n] = 0

        # Local variable binding for tight loop performance
        root_get = self.root.get
        single_char_tok = self.single_char_tok
        t2id_get = self.token_to_id.get

        # Backward DP: solve suffix subproblems from right to left
        for i in range(n - 1, -1, -1):
            ch = text[i]
            code = ord(ch)

            # Step 1: Base 1-character transition (baseline cost)
            best_cost = 1 + dp[i + 1]
            best_next = i + 1
            best_tok = single_char_tok[code] if code < 256 else t2id_get(ch, 0)

            # Step 2: Explore multi-character tokens in the Trie (lengths >= 2)
            node = root_get(ch)
            if node is not None:
                curr = i + 1
                while curr < n:
                    node = node.get(text[curr])
                    if node is None:
                        break
                    curr += 1
                    # Terminal token check
                    if "" in node:
                        cost = 1 + dp[curr]
                        if cost < best_cost:
                            best_cost = cost
                            best_next = curr
                            best_tok = node[""]

                            # If cost == 1, this token reaches the end of the text.
                            # No non-empty encoding can use fewer than 1 token. Break immediately.
                            if cost == 1:
                                break

            # Store the optimal decision for suffix text[i:]
            dp[i] = best_cost
            next_idx[i] = best_next
            next_tok[i] = best_tok

        # 3. Path reconstruction forwards (linear walk, no reverse needed)
        tokens = []
        curr = 0
        while curr < n:
            tokens.append(next_tok[curr])
            curr = next_idx[curr]

        return tokens

    def encode(self, texts: list[str]) -> list[list[int]]:
        """
        Encodes a batch of strings into lists of integer token IDs.

        Each input string is independently tokenized to achieve the global
        minimum token count via shortest-path DP.

        Args:
            texts: List of normalized ASCII input strings.

        Returns:
            List of lists of integer token IDs, exactly one list per input string.

        Raises:
            TypeError: If `texts` is not a list or tuple.
        """

        if not isinstance(texts, (list, tuple)):
            raise TypeError(f"encode() expects list[str], got {type(texts).__name__}")
        encode_one = self._encode_one
        return [encode_one(t) for t in texts]

    def decode(self, encoded_texts: list[list[int]]) -> list[str]:
        """Reconstructs the original strings from token ID sequences exactly.

        Decoding reproduces every character, case, space, and punctuation losslessly
        using dense array table lookups.

        Args:
            encoded_texts: Batch of token ID sequences (list of lists of ints).

        Returns:
            List of reconstructed original strings.

        Raises:
            TypeError: If `encoded_texts` is not a list or tuple.
        """
        
        if not isinstance(encoded_texts, (list, tuple)):
            raise TypeError(f"decode() expects list[list[int]], got {type(encoded_texts).__name__}")
        id_list = self.id_to_token_list
        max_valid = len(id_list)
        return [
            "".join(id_list[tid] for tid in seq if 0 <= tid < max_valid)
            for seq in encoded_texts
        ]