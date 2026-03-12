import pysam
import pandas as pd


def load_vcf(vcf_file) -> pd.DataFrame:
    """Load a VCF file (path or file-like object) into a pandas DataFrame.
    Supports single-sample and multi-sample VCFs.
    """
    import tempfile, os, shutil

    if hasattr(vcf_file, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf") as tmp:
            shutil.copyfileobj(vcf_file, tmp)
            vcf_path = tmp.name
        cleanup = True
    else:
        vcf_path = vcf_file
        cleanup = False

    vcf = pysam.VariantFile(vcf_path)
    samples = list(vcf.header.samples)
    records = []

    for record in vcf.fetch():
        depth = record.info.get("DP", 0)
        af = record.info.get("AF", None)
        if isinstance(af, tuple):
            af = af[0]

        variant_type = "SNP"
        if len(record.ref) != 1 or len(record.alts[0]) != 1:
            variant_type = "INDEL"

        row = {
            "chrom": record.chrom,
            "position": record.pos,
            "ref": record.ref,
            "alt": record.alts[0],
            "quality": record.qual,
            "depth": depth,
            "af": af,
            "variant_type": variant_type,
        }

        # Per-sample genotypes for multi-sample VCFs
        for sample in samples:
            s = record.samples[sample]
            gt = s.get("GT", None)
            gt_str = "/".join(str(a) if a is not None else "." for a in gt) if gt else "."
            row[f"sample_{sample}_GT"] = gt_str

        records.append(row)

    if cleanup:
        os.unlink(vcf_path)

    df = pd.DataFrame(records)
    df["af"] = pd.to_numeric(df["af"], errors="coerce")
    return df


def get_samples(vcf_file) -> list:
    """Return list of sample names from a VCF file."""
    import tempfile, os, shutil

    if hasattr(vcf_file, "read"):
        vcf_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf") as tmp:
            shutil.copyfileobj(vcf_file, tmp)
            vcf_path = tmp.name
        cleanup = True
        vcf_file.seek(0)
    else:
        vcf_path = vcf_file
        cleanup = False

    vcf = pysam.VariantFile(vcf_path)
    samples = list(vcf.header.samples)

    if cleanup:
        os.unlink(vcf_path)

    return samples


def classify_ts_tv(ref: str, alt: str) -> str:
    """Classify a SNP as transition (Ts) or transversion (Tv)."""
    transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    pair = (ref.upper(), alt.upper())
    if pair in transitions:
        return "Ts"
    return "Tv"

