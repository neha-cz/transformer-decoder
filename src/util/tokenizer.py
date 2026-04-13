"""Tokenizer builder and loader for DEM prompt parsing.

The tokenizer converts DEM node prompts (e.g. ``"E(3,4)[1]: 2.89 <ans>"``)
into integer token sequences that an AI decoder can consume.

Vocabulary layout (by ID order)::

    0   <unk>     unknown / OOV fallback
    1   <pad>     padding
    2   <ans>     appended to every node prompt; marks generation boundary
    3-7           node-type prefixes and structural punctuation
    8-12          delimiters & signs
    13-112        integer literals "0" … "99"
    113-118       system-level keywords

Two entry points:

* ``build_tokenizer`` — (re)creates a word-level tokenizer from vocabulary
  constants and optional user extensions.  Requires the ``tokenizers`` package.
* ``load_tokenizer`` — loads a pre-built ``tokenizer.json`` as a HuggingFace
  ``PreTrainedTokenizerFast``.  Requires the ``transformers`` package.

Both heavy dependencies are imported lazily so the core package stays lightweight.
"""

import os
from typing import Dict, List, Optional, Set, Union

# ---------------------------------------------------------------------------
# Default vocabulary components
# ---------------------------------------------------------------------------

#: Special and structural tokens shared by all DEM prompts.
#: <ans> — appended to each node prompt; signals "produce your answer now".
#:         The node representation is extracted from the <ans> position via
#:         causal attention (sees the full prompt).
DEFAULT_STRUCTURAL_TOKENS: List[str] = [
    '<unk>', '<pad>', '<ans>', '<syn>',
    'E', 'D', 'L',             # node-type prefixes
    '(', ')', '[', ']',        # grouping
    ',', ':', '.', '+', '-',   # delimiters & signs
]

#: Keyword tokens used in system-level prompts.
DEFAULT_KEYWORDS: List[str] = [
    'code', 'type', 'distance', 'layers',
    'surface', 'repetition',
]

#: How many integer literals ``"0"`` … ``"num_range-1"`` to include.
DEFAULT_NUM_RANGE: int = 100


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_vocab(
    extra_tokens: Optional[List[str]] = None,
    num_range: int = DEFAULT_NUM_RANGE,
) -> Dict[str, int]:
    """Assemble the vocabulary dict ``{token: id}`` from components.

    Order:  structural → numerals 0..(num_range-1) → keywords → extra_tokens.
    Duplicates are silently skipped (first occurrence wins).

    Args:
        extra_tokens: Additional tokens appended after the defaults
                      (e.g. ``['toric', 'color']`` for new code families).
        num_range: Integer range ``[0, num_range)`` included as tokens.

    Returns:
        Ordered ``{token_str: token_id}`` mapping.

    Examples:
        >>> vocab = build_vocab()
        >>> vocab['<ans>'], vocab['0'], vocab['1']
        (2, 13, 14)
        >>> len(vocab)
        121
    """
    tokens: List[str] = list(DEFAULT_STRUCTURAL_TOKENS)
    tokens.extend(str(i) for i in range(num_range))
    tokens.extend(DEFAULT_KEYWORDS)
    if extra_tokens:
        tokens.extend(extra_tokens)

    # De-duplicate while preserving order
    seen: Set[str] = set()
    vocab: Dict[str, int] = {}
    for tok in tokens:
        if tok not in seen:
            vocab[tok] = len(vocab)
            seen.add(tok)
    return vocab


def build_tokenizer(
    extra_tokens: Optional[List[str]] = None,
    num_range: int = DEFAULT_NUM_RANGE,
    save_path: Optional[Union[str, os.PathLike]] = None,
):
    """Build a word-level tokenizer for DEM prompts.

    Special tokens (``<ans>``, ``<pad>``, ``<unk>``) are registered
    via ``add_special_tokens`` so the pre-tokenizer never splits them.

    Args:
        extra_tokens: Additional tokens to append to the default vocabulary
                      (e.g. ``['toric', 'color']``).
        num_range: Integer range ``[0, num_range)`` to include as tokens.
        save_path: If given, persist the tokenizer JSON to this path.

    Returns:
        A ``tokenizers.Tokenizer`` instance.

    Raises:
        ImportError: If the ``tokenizers`` package is not installed.

    Examples:
        >>> tok = build_tokenizer(save_path='tokenizer.json')
        >>> tok.encode('E(3,4)[1]: 2.89 <ans>').ids
        [3, 6, 16, 10, 17, 7, 8, 15, 12, ...]
    """
    try:
        from tokenizers import Tokenizer, AddedToken
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation
    except ImportError:
        raise ImportError(
            "The 'tokenizers' package is required to build the tokenizer. "
            "Install it with:  pip install tokenizers"
        )

    vocab = build_vocab(extra_tokens=extra_tokens, num_range=num_range)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = Sequence([
        Whitespace(),
        Punctuation(behavior="isolated"),
    ])

    # Register angle-bracket tokens as special so the pre-tokenizer
    # treats them as atomic units (not split on '<', '>').
    special_tokens = [
        tok for tok in vocab if tok.startswith('<') and tok.endswith('>')
    ]
    tokenizer.add_special_tokens([
        AddedToken(tok, special=True) for tok in special_tokens
    ])

    if save_path is not None:
        save_path = str(save_path)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        tokenizer.save(save_path)

    return tokenizer


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_tokenizer(path: Optional[Union[str, os.PathLike]] = None):
    """Load the tokenizer as a HuggingFace ``PreTrainedTokenizerFast``.

    Args:
        path: Path to ``tokenizer.json``.  Defaults to the bundled file
              shipped with this package (``util/tokenizer.json``).

    Returns:
        A ``PreTrainedTokenizerFast`` with ``pad_token`` already set.

    Raises:
        ImportError: If the ``transformers`` package is not installed.
        FileNotFoundError: If the tokenizer file does not exist.

    Examples:
        >>> tokenizer = load_tokenizer()
        >>> tokenizer('E(0): 2.20 <ans>')['input_ids']
        [3, 6, 13, 7, 11, ...]
    """
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        raise ImportError(
            "The 'transformers' package is required to load the tokenizer. "
            "Install it with:  pip install transformers"
        )

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")
    path = str(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Tokenizer file not found: {path}\n"
            "You can regenerate it with:  build_tokenizer(save_path=...)"
        )

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
    fast_tokenizer.add_special_tokens({'pad_token': '<pad>'})
    return fast_tokenizer
