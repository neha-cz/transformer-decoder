"""Utility files and resources."""

import os as _os

UTIL_DIR = _os.path.dirname(_os.path.abspath(__file__))


def get_tokenizer_path() -> str:
    """Return the absolute path to the bundled tokenizer.json."""
    return _os.path.join(UTIL_DIR, "tokenizer.json")


from .tokenizer import build_vocab, build_tokenizer, load_tokenizer

__all__ = [
    'UTIL_DIR',
    'get_tokenizer_path',
    'build_vocab',
    'build_tokenizer',
    'load_tokenizer',
]
