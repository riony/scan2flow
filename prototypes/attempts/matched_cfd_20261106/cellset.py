"""Saved v2412 binary-header cellSet parser, including compact empty 0()."""

import re

import numpy as np

ARCH = b"LSB;label=32;scalar=64"


def binary(path):
    data = path.read_bytes()
    assert ARCH in data[:1200] and re.search(rb"format\s+binary\s*;", data[:1200]), path
    return data


def cell_set(path, allow_missing_zero=False):
    if not path.exists():
        assert allow_missing_zero, f"Missing saved set: {path}"
        return np.empty(0, dtype=np.int64)
    data = binary(path)
    assert re.search(rb"class\s+cellSet\s*;", data[:1200])
    # Empty cell sets may use compact ``0()`` or whitespace-separated syntax.
    match = re.search(rb"\n(\d+)\s*\(", data)
    assert match is not None, path
    count = int(match[1])
    # OpenFOAM labelHashSet writes textual labels even with a binary FoamFile.
    payload_bytes, closing, _ = data[match.end() :].partition(b")")
    assert closing == b")", "Unterminated cellSet payload"
    payload = payload_bytes.decode("ascii")
    assert re.fullmatch(r"[\s\d]*", payload), "Unsupported cellSet payload"
    tokens = payload.split()
    saved = np.asarray([int(token) for token in tokens], dtype=np.int64)
    assert len(saved) == count and len(np.unique(saved)) == count
    return np.sort(saved)


def cell_set_before_compact_zero_fix(path, allow_missing_zero=False):
    if not path.exists():
        assert allow_missing_zero, f"Missing saved set: {path}"
        return np.empty(0, dtype=np.int64)
    data = binary(path)
    assert re.search(rb"class\s+cellSet\s*;", data[:1200])
    match = re.search(rb"\n(\d+)\s*\n\(", data)
    assert match is not None, path
    count = int(match[1])
    # OpenFOAM labelHashSet writes textual labels even with a binary FoamFile.
    payload = data[match.end() :].split(b")", 1)[0].decode("ascii")
    assert re.fullmatch(r"[\s\d]*", payload), "Unsupported cellSet payload"
    saved = np.fromstring(payload, sep=" ", dtype=np.int64)
    assert len(saved) == count and len(np.unique(saved)) == count
    return np.sort(saved)
