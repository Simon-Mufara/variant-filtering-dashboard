import pandas as pd
import io


def load_vcf(vcf_file) -> pd.DataFrame:
    """Load a VCF file (path or Streamlit UploadedFile) into a DataFrame.
    Pure-Python parser — no compiled dependencies required.
    """
    if hasattr(vcf_file, "read"):
        content = vcf_file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        lines = content.splitlines()
    else:
        with open(vcf_file, "r") as f:
            lines = f.read().splitlines()

    header = None
    samples = []
    records = []

    for line in lines:
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            cols = line.lstrip("#").split("\t")
            header = cols
            # samples are columns after FORMAT (index 8)
            samples = cols[9:] if len(cols) > 9 else []
            continue
        if header is None:
            continue

        parts = line.split("\t")
        if len(parts) < 8:
            continue

        chrom, pos, _, ref, alt, qual, _, info = parts[:8]
        fmt_keys = parts[8].split(":") if len(parts) > 8 else []
        sample_values = parts[9:] if len(parts) > 9 else []

        # Parse INFO field
        info_dict = _parse_info(info)
        depth = int(info_dict.get("DP", 0) or 0)
        af_raw = info_dict.get("AF", None)
        af = float(af_raw.split(",")[0]) if af_raw else None

        # Classify variant
        alts = alt.split(",")
        variant_type = "SNP" if (len(ref) == 1 and len(alts[0]) == 1) else "INDEL"

        row = {
            "chrom": chrom,
            "position": int(pos),
            "ref": ref,
            "alt": alts[0],
            "quality": float(qual) if qual not in (".", "") else None,
            "depth": depth,
            "af": af,
            "variant_type": variant_type,
        }

        # Per-sample genotypes
        for i, sample in enumerate(samples):
            if i < len(sample_values) and fmt_keys:
                vals = sample_values[i].split(":")
                gt_idx = fmt_keys.index("GT") if "GT" in fmt_keys else None
                row[f"sample_{sample}_GT"] = vals[gt_idx] if gt_idx is not None and gt_idx < len(vals) else "."
            else:
                row[f"sample_{sample}_GT"] = "."

        records.append(row)

    df = pd.DataFrame(records)
    if "af" in df.columns:
        df["af"] = pd.to_numeric(df["af"], errors="coerce")
    if "quality" in df.columns:
        df["quality"] = pd.to_numeric(df["quality"], errors="coerce")
    return df


def _parse_info(info: str) -> dict:
    """Parse a VCF INFO field string into a dict."""
    result = {}
    for part in info.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
        else:
            result[part] = True
    return result


def classify_ts_tv(ref: str, alt: str) -> str:
    """Classify a SNP as Transition (Ts) or Transversion (Tv)."""
    transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    return "Ts" if (ref.upper(), alt.upper()) in transitions else "Tv"


