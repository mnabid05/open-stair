# SPDX-License-Identifier: Apache-2.0
"""Token trie used by constrained decoding; independent of ML dependencies."""

from __future__ import annotations

from collections.abc import Iterable


class TokenTrie:
    def __init__(self, sequences: Iterable[list[int]], eos_token_id: int) -> None:
        self.root: dict = {}
        self.eos = eos_token_id
        self.max_length = 0
        count = 0
        for sequence in sequences:
            if not sequence or eos_token_id in sequence:
                raise ValueError("Label tokens must be nonempty and cannot contain EOS")
            node = self.root
            for token in [*sequence, eos_token_id]:
                node = node.setdefault(token, {})
            self.max_length = max(self.max_length, len(sequence) + 1)
            count += 1
        if not count:
            raise ValueError("A trie needs at least one label")

    def allowed(self, prefix: list[int]) -> list[int]:
        node = self.root
        for index, token in enumerate(prefix):
            if token not in node:
                return []
            node = node[token]
            if token == self.eos:
                # Transformers keeps finished beams padded in later decoding steps.
                return [self.eos] if all(t == self.eos for t in prefix[index:]) else []
        return sorted(node)
