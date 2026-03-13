"""Multi-format variant input parser.

Supports:
  - VCF / VCF.GZ    (standard, annotated, GATK, DeepVariant, etc.)
  - MAF              (TCGA Mutation Annotation Format v2.4)
  - TSV / CSV        (generic variant table — auto-detects column names)
  - BED              (regions only — used as a filter, not variant calls)

All parsers return a DataFrame with canonical columns matching load_vcf():
    chrom, position, ref, alt, quality, depth, af, filter, variant_type, info_raw
plus any extra annotation columns that were present in the input.
"""
from __future__ import annotations

import io
import gzip


import pandas as pd

# ── column-name synonyms for generic TSV/CSV detection ───────────────────────
_CHROM_ALIASES  = {"chrom", "chromosome", "chr", "contig", "seqname", "chrom_name"}
_POS_ALIASES    = {"pos", "position", "start", "chromstart", "start_position"}
_REF_ALIASES    = {"ref", "reference", "ref_allele", "reference_allele"}
_ALT_ALIASES    = {"alt", "alternate", "alt_allele", "tumor_seq_allele2",
                   "alt_allele", "alternative"}
_QUAL_ALIASES   = {"qual", "quality", "phred_quality", "t_depth", "tumor_depth"}
_DEPTH_ALIASES  = {"dp", "depth", "read_depth", "t_depth", "tumor_read_count",
                   "read_count"}
_AF_ALIASES     = {"af", "allele_freq", "allele_frequency", "vaf", "tumor_vaf",
                   "tumor_f", "t_vaf"}
_FILTER_ALIASES = {"filter", "filter_status", "variant_filter"}

# Standard MAF column names (GDC/TCGA MAF v2.4)
_MAF_REQUIRED = {"Hugo_Symbol", "Chromosome", "Start_Position",
                 "Reference_Allele", "Tumor_Seq_Allele2"}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(filename: str, first_bytes: bytes) -> str:
    """Guess file format from extension and/or content.

    Returns: 'vcf', 'maf', 'tsv', 'csv', or 'unknown'
    """
    name = filename.lower()
    if name.endswith((".vcf", ".vcf.gz")):
        return "vcf"
    if name.endswith(".maf") or name.endswith(".maf.gz"):
        return "maf"
    if name.endswith(".tsv") or name.endswith(".txt"):
        return "tsv"
    if name.endswith(".csv"):
        return "csv"

    # Inspect content
    try:
        text = _decode_bytes(first_bytes)
        lines = [line for line in text.splitlines() if line.strip()]
        for line in lines[:5]:
            if line.startswith("##fileformat=VCF"):
                return "vcf"
            if line.startswith("#CHROM"):
                return "vcf"
            if "Hugo_Symbol" in line and "Chromosome" in line:
                return "maf"
        # Tab-separated?
        if lines and "\t" in lines[0]:
            return "tsv"
        return "csv"
    except Exception:
        return "unknown"


def load_any(vcf_file, filename: str = "") -> pd.DataFrame:
    """Load a variant file of any supported format into a canonical DataFrame.

    Parameters
    ----------
    vcf_file : str | pathlib.Path | UploadedFile | BytesIO
        File to load.
    filename : str
        Used for format detection; if vcf_file has a `.name` attribute it is
        used automatically.
    """
    if hasattr(vcf_file, "name") and not filename:
        filename = vcf_file.name

    raw = _read_bytes(vcf_file)
    fmt = detect_format(filename, raw[:4096])

    if fmt == "vcf":
        from utils.vcf_parser import load_vcf
        vcf_file.seek(0) if hasattr(vcf_file, "seek") else None
        return load_vcf(vcf_file)

    if fmt == "maf":
        return _load_maf(raw)

    if fmt in ("tsv", "csv"):
        sep = "\t" if fmt == "tsv" else ","
        return _load_table(raw, sep=sep)

    # Try VCF as last resort
    from utils.vcf_parser import load_vcf
    return load_vcf(io.BytesIO(raw))


def load_bed(bed_file) -> pd.DataFrame:
    """Load a BED file into a DataFrame with columns: chrom, start, end, name."""
    raw = _read_bytes(bed_file)
    text = _decode_bytes(raw)
    rows = []
    for line in text.splitlines():
        if line.startswith(("browser", "track", "#")) or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            parts = line.split()
        if len(parts) < 3:
            continue
        rows.append({
            "chrom": parts[0],
            "start": _safe_int(parts[1]),
            "end":   _safe_int(parts[2]),
            "name":  parts[3] if len(parts) > 3 else ".",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAF parser
# ─────────────────────────────────────────────────────────────────────────────

def _load_maf(raw: bytes) -> pd.DataFrame:
    text = _decode_bytes(raw)
    # Skip comment lines (start with #)
    clean = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    maf = pd.read_csv(io.StringIO(clean), sep="\t", dtype=str, low_memory=False)

    missing = _MAF_REQUIRED - set(maf.columns)
    if missing:
        raise ValueError(
            f"MAF file is missing required columns: {', '.join(sorted(missing))}.\n"
            "Expected GDC/TCGA MAF v2.4 format."
        )

    records = []
    for _, row in maf.iterrows():
        chrom = str(row.get("Chromosome", "")).strip()
        pos = _safe_int(row.get("Start_Position", 0))
        ref = str(row.get("Reference_Allele", "N")).strip()
        alt = str(row.get("Tumor_Seq_Allele2", "N")).strip()
        qual = _safe_float(row.get("t_depth") or row.get("Tumor_Depth"))
        depth = _safe_int(row.get("t_depth") or row.get("Tumor_Depth") or row.get("n_depth"))
        af = _safe_float(row.get("t_vaf") or row.get("tumor_f") or row.get("VAF"))
        gene = str(row.get("Hugo_Symbol", "")).strip()
        effect = str(row.get("Variant_Classification", "")).strip()
        hgvsp = str(row.get("HGVSp_Short", "")).strip()
        dbsnp = str(row.get("dbSNP_RS", "")).strip()

        # Build a pseudo INFO string so downstream tools can parse it
        info_parts = []
        if depth:
            info_parts.append(f"DP={depth}")
        if af is not None:
            info_parts.append(f"AF={af}")
        if gene:
            info_parts.append(f"GENE={gene}")
        if effect:
            info_parts.append(f"EFFECT={effect}")
        if hgvsp and hgvsp not in (".", "nan", ""):
            info_parts.append(f"HGVSP={hgvsp}")
        if dbsnp and dbsnp not in (".", "nan", ""):
            info_parts.append(f"DBSNP={dbsnp}")

        # Variant type from MAF
        maf_type = str(row.get("Variant_Type", "")).strip()
        if maf_type in ("SNP",):
            vtype = "SNP"
        elif maf_type in ("INS", "DEL", "DNP", "TNP", "ONP"):
            vtype = "INDEL"
        else:
            vtype = _classify(ref, alt)

        # ClinVar from MAF if present
        clnsig = row.get("CLIN_SIG") or row.get("ClinVar_VCF_CLNSIG") or ""
        clndn = row.get("ClinVar_VCF_CLNDN") or ""
        if clnsig and str(clnsig) not in (".", "nan", ""):
            info_parts.append(f"CLNSIG={clnsig}")
        if clndn and str(clndn) not in (".", "nan", ""):
            info_parts.append(f"CLNDN={clndn}")

        rec = {
            "chrom": chrom,
            "position": pos,
            "ref": ref,
            "alt": alt,
            "quality": qual,
            "depth": depth,
            "af": af,
            "filter": str(row.get("FILTER", "PASS")).strip(),
            "variant_type": vtype,
            "info_raw": ";".join(info_parts),
            "gene_name": gene,
            "effect": effect,
        }
        records.append(rec)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Generic TSV / CSV parser
# ─────────────────────────────────────────────────────────────────────────────

def _load_table(raw: bytes, sep: str = "\t") -> pd.DataFrame:
    text = _decode_bytes(raw)
    clean = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    tbl = pd.read_csv(io.StringIO(clean), sep=sep, dtype=str, low_memory=False)

    # Map columns to canonical names
    col_map = {}
    for col in tbl.columns:
        lower = col.lower().strip()
        if lower in _CHROM_ALIASES and "chrom" not in col_map:
            col_map[col] = "chrom"
        elif lower in _POS_ALIASES and "position" not in col_map:
            col_map[col] = "position"
        elif lower in _REF_ALIASES and "ref" not in col_map:
            col_map[col] = "ref"
        elif lower in _ALT_ALIASES and "alt" not in col_map:
            col_map[col] = "alt"
        elif lower in _QUAL_ALIASES and "quality" not in col_map:
            col_map[col] = "quality"
        elif lower in _DEPTH_ALIASES and "depth" not in col_map:
            col_map[col] = "depth"
        elif lower in _AF_ALIASES and "af" not in col_map:
            col_map[col] = "af"
        elif lower in _FILTER_ALIASES and "filter" not in col_map:
            col_map[col] = "filter"

    tbl = tbl.rename(columns=col_map)

    required = {"chrom", "position", "ref", "alt"}
    missing = required - set(tbl.columns)
    if missing:
        raise ValueError(
            f"Could not find required columns in table: {', '.join(sorted(missing))}.\n"
            "Expected columns (case-insensitive): CHROM/Chromosome, POS/Position, "
            "REF/Reference, ALT/Alternate."
        )

    tbl["position"] = tbl["position"].apply(lambda x: _safe_int(x))
    tbl["quality"] = tbl.get("quality", pd.Series(dtype=float)).apply(_safe_float)
    tbl["depth"] = tbl.get("depth", pd.Series(dtype=int)).apply(_safe_int)
    tbl["af"] = tbl.get("af", pd.Series(dtype=float)).apply(_safe_float)

    if "filter" not in tbl.columns:
        tbl["filter"] = "."
    if "variant_type" not in tbl.columns:
        tbl["variant_type"] = tbl.apply(lambda r: _classify(str(r["ref"]), str(r["alt"])), axis=1)
    if "info_raw" not in tbl.columns:
        tbl["info_raw"] = ""

    return tbl.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_bytes(vcf_file) -> bytes:
    if hasattr(vcf_file, "read"):
        data = vcf_file.read()
        if hasattr(vcf_file, "seek"):
            vcf_file.seek(0)
        return data if isinstance(data, bytes) else data.encode()
    with open(str(vcf_file), "rb") as f:
        return f.read()


def _decode_bytes(raw: bytes) -> str:
    try:
        return gzip.decompress(raw).decode("utf-8", errors="replace")
    except (OSError, EOFError):
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw


def _classify(ref: str, alt: str) -> str:
    if alt in (".", "*", "<NON_REF>") or alt.startswith("<"):
        return "OTHER"
    if len(ref) == 1 and len(alt) == 1:
        return "SNP"
    if len(ref) == len(alt) and len(ref) > 1:
        return "MNP"
    return "INDEL"


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(float(str(val).split(",")[0]))
    except (ValueError, TypeError):
        return default


def _safe_float(val) -> float | None:
    if val is None or str(val).strip() in (".", "", "NA", "nan", "None"):
        return None
    try:
        return float(str(val).split(",")[0])
    except (ValueError, TypeError):
        return None


def supported_extensions() -> list[str]:
    """Return list of file extensions accepted by load_any()."""
    return ["vcf", "vcf.gz", "maf", "maf.gz", "tsv", "txt", "csv"]
