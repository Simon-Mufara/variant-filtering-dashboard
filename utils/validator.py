"""VCF input validation — called before parsing to surface clear errors."""
from __future__ import annotations
import re
from typing import Tuple

MAX_FILE_SIZE_MB = 500
_VCF_HEADER_RE = re.compile(r"^##fileformat=VCFv", re.IGNORECASE)
_CHROM_HEADER_RE = re.compile(r"^#CHROM", re.IGNORECASE)


def validate_vcf(vcf_file) -> Tuple[bool, str]:
    """Validate an uploaded or path-based VCF file before full parsing.

    Returns:
        (True, "")            — file looks valid, proceed
        (False, error_msg)    — describe the problem to the user
    """
    try:
        # ── 1. File size ──────────────────────────────────────────────────────
        if hasattr(vcf_file, "size"):
            size_mb = vcf_file.size / (1024 ** 2)
            if size_mb > MAX_FILE_SIZE_MB:
                return False, (
                    f"File is {size_mb:.0f} MB — maximum allowed is {MAX_FILE_SIZE_MB} MB. "
                    "Split the VCF by chromosome for large datasets."
                )

        # ── 2. Read first 64 KB to inspect headers ────────────────────────────
        raw = _peek(vcf_file, 65_536)
        lines = raw.splitlines()

        if not lines:
            return False, "File appears to be empty."

        first_non_empty = next((l for l in lines if l.strip()), "")
        if not first_non_empty.startswith("##"):
            return False, (
                "File does not start with VCF meta-information lines (##). "
                "Make sure this is a valid VCF file."
            )

        has_fileformat = any(_VCF_HEADER_RE.match(l) for l in lines[:20])
        if not has_fileformat:
            return False, (
                "Missing '##fileformat=VCFv4.x' header. "
                "The file may not be a valid VCF."
            )

        has_chrom = any(_CHROM_HEADER_RE.match(l) for l in lines)
        if not has_chrom:
            return False, (
                "No '#CHROM' column header found. "
                "The VCF is missing its column-header line."
            )

        # ── 3. Spot-check first data line ─────────────────────────────────────
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                parts = line.split()
            if len(parts) < 5:
                return False, (
                    f"First data line has only {len(parts)} columns "
                    "(expected ≥8 for a valid VCF)."
                )
            break

        return True, ""

    except Exception as exc:
        return False, f"Could not read file: {exc}"


# ── helpers ───────────────────────────────────────────────────────────────────

def _peek(vcf_file, n_bytes: int) -> str:
    """Read the first n_bytes as a decoded string without consuming the stream."""
    import gzip, io

    if hasattr(vcf_file, "read"):
        raw = vcf_file.read(n_bytes)
        # Rewind so the full parser can re-read from the start
        if hasattr(vcf_file, "seek"):
            vcf_file.seek(0)
        elif hasattr(vcf_file, "getvalue"):
            pass  # BytesIO — already peeked, caller will getvalue()
        try:
            return gzip.decompress(raw).decode("utf-8", errors="replace")
        except (OSError, EOFError):
            return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

    path = str(vcf_file)
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read(n_bytes)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(n_bytes)
