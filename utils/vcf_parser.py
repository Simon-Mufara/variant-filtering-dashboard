import pandas as pd


# Known chromosome aliases to normalise to a canonical name
_CHROM_ALIASES = {
    "chrmt": "MT", "chrm": "MT", "mt": "MT", "m": "MT",
    "chrun": "Un", "chrUn": "Un",
}

# Standard human chromosomes in display order
STANDARD_CHROMS = (
    [str(i) for i in range(1, 23)] +
    ["X", "Y", "MT"] +
    [f"chr{i}" for i in range(1, 23)] +
    ["chrX", "chrY", "chrMT", "chrM"]
)


def normalise_chrom(chrom: str) -> str:
    """Return a consistent chromosome name.

    Accepts any of:  1, chr1, CHR1, chrX, chrMT, chrM, MT, chrUn_gl000220, etc.
    Strips leading/trailing whitespace and standardises case for the prefix only.
    """
    chrom = chrom.strip()
    lower = chrom.lower()
    if lower in _CHROM_ALIASES:
        return _CHROM_ALIASES[lower]
    return chrom


def load_vcf(vcf_file) -> pd.DataFrame:
    """Load a VCF file (path or Streamlit UploadedFile) into a DataFrame.

    Robust pure-Python parser:
    - Handles any chromosome name (wildtype, non-standard, MT, Un, etc.)
    - Handles missing QUAL / DP (. fields)
    - Handles multi-allelic ALT (expands each allele to its own row)
    - Handles both tab- and space-separated files
    - Handles gzipped VCF if passed as a path (*.vcf.gz)
    """
    lines = _read_lines(vcf_file)

    header = None
    samples = []
    records = []

    for line in lines:
        line = line.rstrip("\n\r")
        if not line:
            continue
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM") or line.startswith("#chrom"):
            cols = line.lstrip("#").split("\t")
            if len(cols) == 1:          # fallback: space-separated
                cols = line.lstrip("#").split()
            header = [c.upper() for c in cols]
            samples = cols[9:] if len(cols) > 9 else []
            continue
        if header is None:
            continue

        parts = line.split("\t")
        if len(parts) == 1:
            parts = line.split()     # fallback for space-separated VCFs
        if len(parts) < 8:
            continue

        chrom = normalise_chrom(parts[0])
        pos_raw, _, ref, alt_field, qual_raw, filt, info_raw = parts[1:8]

        fmt_keys = parts[8].split(":") if len(parts) > 8 else []
        sample_values = parts[9:] if len(parts) > 9 else []

        # Parse QUAL — '.' means missing
        quality = _parse_float(qual_raw)

        # Parse INFO
        info_dict = _parse_info(info_raw)
        depth = _parse_int(info_dict.get("DP", None), default=0)
        af_raw = info_dict.get("AF", None)

        # Expand multi-allelic ALT (e.g. "A,T,G" → 3 rows)
        alts = [a.strip() for a in alt_field.split(",") if a.strip() not in (".", "*", "<NON_REF>")]
        if not alts:
            alts = [alt_field]

        for alt in alts:
            af = _parse_float(af_raw.split(",")[0] if af_raw else None)
            variant_type = _classify(ref, alt)

            row = {
                "chrom": chrom,
                "position": _parse_int(pos_raw, default=0),
                "ref": ref,
                "alt": alt,
                "quality": quality,
                "depth": depth,
                "af": af,
                "filter": filt,
                "variant_type": variant_type,
                "info_raw": info_raw,  # kept for SnpEff/ClinVar downstream parsing
            }

            # SV-specific fields
            if variant_type == "SV":
                row["svtype"] = info_dict.get("SVTYPE", alt.strip("<>") if alt.startswith("<") else "")
                row["svlen"] = _parse_int(info_dict.get("SVLEN", None), default=0)
                row["sv_end"] = _parse_int(info_dict.get("END", None), default=0)

            # Genotype columns
            for i, sample in enumerate(samples):
                gt = "."
                if i < len(sample_values) and fmt_keys:
                    vals = sample_values[i].split(":")
                    if "GT" in fmt_keys:
                        gt_idx = fmt_keys.index("GT")
                        gt = vals[gt_idx] if gt_idx < len(vals) else "."
                row[f"sample_{sample}_GT"] = gt

            records.append(row)

    if not records:
        return pd.DataFrame(columns=[
            "chrom", "position", "ref", "alt", "quality",
            "depth", "af", "filter", "variant_type",
        ])

    df = pd.DataFrame(records)
    df["af"] = pd.to_numeric(df["af"], errors="coerce")
    df["quality"] = pd.to_numeric(df["quality"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce").fillna(0).astype(int)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0).astype(int)
    return df


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_lines(vcf_file) -> list:
    """Read lines from a file path or Streamlit UploadedFile, including .gz."""
    import gzip

    if hasattr(vcf_file, "read"):
        raw = vcf_file.read()
        # Try gzip
        try:
            content = gzip.decompress(raw).decode("utf-8", errors="replace")
        except (OSError, EOFError):
            content = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        return content.splitlines()

    path = str(vcf_file)
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def _parse_info(info: str) -> dict:
    """Parse VCF INFO field into a dict."""
    result = {}
    for part in info.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
        elif part:
            result[part] = True
    return result


def _parse_float(val) -> float | None:
    if val is None or str(val).strip() in (".", "", "NA", "nan"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_int(val, default: int = 0) -> int:
    if val is None or str(val).strip() in (".", "", "NA"):
        return default
    try:
        return int(float(str(val).split(",")[0]))
    except (ValueError, TypeError):
        return default


def _classify(ref: str, alt: str) -> str:
    """Classify variant as SNP, INDEL, or MNP."""
    if alt in (".", "*", "<NON_REF>"):
        return "OTHER"
    if alt.startswith("<"):
        return "SV"
    if len(ref) == 1 and len(alt) == 1:
        return "SNP"
    if len(ref) == len(alt) and len(ref) > 1:
        return "MNP"
    return "INDEL"


def classify_ts_tv(ref: str, alt: str) -> str:
    """Classify a SNP as Transition (Ts) or Transversion (Tv)."""
    transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    return "Ts" if (ref.upper(), alt.upper()) in transitions else "Tv"



