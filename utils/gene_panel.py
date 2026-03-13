"""Gene panel filtering — filter variants to genes in a named or custom panel.

Built-in panels cover common clinical scenarios. Users can also upload
a plain-text file (one gene per line) or a CSV with a 'gene' column.
"""
from __future__ import annotations
import io
import pandas as pd

# ── Built-in gene panels ──────────────────────────────────────────────────────
PANELS: dict[str, list[str]] = {
    "BRCA (Hereditary Breast/Ovarian)": [
        "BRCA1", "BRCA2", "PALB2", "CHEK2", "ATM", "RAD51C", "RAD51D",
        "BRIP1", "CDH1", "PTEN", "STK11", "TP53",
    ],
    "Lynch Syndrome (Colorectal)": [
        "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM",
    ],
    "Cardio Panel (Inherited Cardiac)": [
        "MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1", "MYL2", "MYL3",
        "ACTC1", "SCN5A", "KCNQ1", "KCNH2", "KCNE1", "KCNE2", "RYR2",
        "CASQ2", "PKP2", "DSP", "DSC2", "JUP", "TMEM43", "DSG2",
        "LMNA", "PLN",
    ],
    "Neurodevelopmental Panel": [
        "MECP2", "PTEN", "TSC1", "TSC2", "NF1", "NF2", "SHANK3",
        "DYRK1A", "ADNP", "CHD8", "FOXP1", "GRIN2B", "KMT2A", "KMT2D",
        "ANKRD11", "ARID1B", "MED13L", "SETD5",
    ],
    "Pharmacogenomics (PGx Core)": [
        "CYP2C19", "CYP2C9", "CYP2D6", "CYP3A4", "CYP3A5",
        "DPYD", "TPMT", "UGT1A1", "VKORC1", "SLCO1B1", "ABCG2",
        "G6PD", "NUDT15", "HLA-A", "HLA-B",
    ],
    "Cancer Predisposition (Broad)": [
        "BRCA1", "BRCA2", "TP53", "PTEN", "STK11", "CDH1", "VHL",
        "SDHA", "SDHB", "SDHC", "SDHD", "SDHAF2", "MAX", "TMEM127",
        "RB1", "WT1", "APC", "MUTYH", "BMPR1A", "SMAD4", "AXIN2",
        "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM", "RET", "MEN1",
        "CDKN2A", "BAP1", "FH", "FLCN", "DICER1", "NBN", "PALB2",
        "ATM", "CHEK2", "RAD51C", "RAD51D", "BARD1", "BRIP1",
    ],
}


def list_panels() -> list[str]:
    return ["Custom upload"] + sorted(PANELS.keys())


def get_panel_genes(panel_name: str) -> list[str]:
    """Return gene list for a named built-in panel."""
    return PANELS.get(panel_name, [])


def parse_custom_panel(file_or_text) -> list[str]:
    """Parse a user-supplied gene list from an UploadedFile or raw text.

    Accepts:
    - Plain text: one gene per line
    - CSV: column named 'gene', 'Gene', 'GENE', 'symbol', or first column
    """
    if hasattr(file_or_text, "read"):
        raw = file_or_text.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
    else:
        raw = str(file_or_text)

    # Try CSV
    try:
        df = pd.read_csv(io.StringIO(raw))
        for col in df.columns:
            if col.lower() in ("gene", "symbol", "gene_symbol", "hgnc_symbol"):
                return df[col].dropna().str.strip().str.upper().tolist()
        # Fall back to first column
        return df.iloc[:, 0].dropna().str.strip().str.upper().tolist()
    except Exception:
        pass

    # Plain text
    genes = [
        line.strip().upper()
        for line in raw.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return genes


def filter_to_panel(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """Keep only variants whose gene annotation is in the panel.

    Checks columns: gene_name (SnpEff), gene (VEP), gene (generic).
    Falls back to searching info_raw for GENE= tags.
    """
    if not genes or df.empty:
        return df

    gene_set = {g.upper() for g in genes}

    for col in ("gene_name", "gene", "Gene"):
        if col in df.columns:
            mask = df[col].astype(str).str.upper().isin(gene_set)
            return df[mask].reset_index(drop=True)

    # Last resort: scan info_raw for GENE=X or ANN contains gene name
    if "info_raw" in df.columns:
        import re
        pattern = re.compile(r"(?:GENE|gene)=([^;,|]+)")

        def _in_panel(info):
            m = pattern.search(str(info))
            return m.group(1).upper() in gene_set if m else False

        mask = df["info_raw"].apply(_in_panel)
        return df[mask].reset_index(drop=True)

    return pd.DataFrame(columns=df.columns)
