"""Backward-compatible SciFact entry point; shared BEIR implementation lives in beir_real."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from standalone.beir_real import (
    ARCHIVE_BYTES,
    ARCHIVE_SHA256,
    ROOT,
    URL,
    english_stopwords,
    evaluate_quality,
    main,
    prepare,
    ranking_metrics,
    read,
    read_source,
    run,
    tokenize,
    validate_data,
)

__all__ = [
    'ARCHIVE_BYTES',
    'ARCHIVE_SHA256',
    'ROOT',
    'URL',
    'english_stopwords',
    'evaluate_quality',
    'main',
    'prepare',
    'ranking_metrics',
    'read',
    'read_source',
    'run',
    'tokenize',
    'validate_data',
]

if __name__ == '__main__':
    main()
