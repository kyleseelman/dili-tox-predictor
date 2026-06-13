"""
Download real DILI datasets from public sources:
  - DILIst (FDA): 1,279 drugs with binary DILI classification
  - Proxy-DILI dataset: ~13,700 compounds with 9 proxy-DILI assay labels
  - Cmax pharmacokinetic data: ~730 compounds with clinical exposure data

Sources:
  - FDA LTKB: https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb
  - Seal et al. 2024: https://github.com/srijitseal/DILI
"""

import os
import gzip
import io
from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd

DATA_DIR = Path(__file__).parent

URLS = {
    "dilist": "https://raw.githubusercontent.com/srijitseal/DILI/main/data/DILI_Goldstandard_1111.csv",
    "proxy_dili": "https://raw.githubusercontent.com/srijitseal/DILI/main/data/proxy-DILI_dataset_withDILIcompounds_13703.csv",
    "cmax": "https://raw.githubusercontent.com/srijitseal/DILI/main/data/Cmax_processed.csv",
}


def _fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def download_dilist(force: bool = False) -> pd.DataFrame:
    out = DATA_DIR / "dilist.csv"
    if out.exists() and not force:
        return pd.read_csv(out)

    raw = _fetch(URLS["dilist"])
    df = pd.read_csv(io.BytesIO(raw))
    df.to_csv(out, index=False)
    print(f"DILIst: {len(df)} compounds -> {out}")
    return df


def download_proxy_dili(force: bool = False) -> pd.DataFrame:
    out = DATA_DIR / "proxy_dili.csv"
    if out.exists() and not force:
        return pd.read_csv(out)

    raw = _fetch(URLS["proxy_dili"])
    df = pd.read_csv(io.BytesIO(raw))
    df.to_csv(out, index=False)
    print(f"Proxy-DILI: {len(df)} compounds -> {out}")
    return df


def download_cmax(force: bool = False) -> pd.DataFrame:
    out = DATA_DIR / "cmax.csv"
    if out.exists() and not force:
        return pd.read_csv(out)

    raw = _fetch(URLS["cmax"])
    df = pd.read_csv(io.BytesIO(raw))
    df.to_csv(out, index=False)
    print(f"Cmax PK: {len(df)} compounds -> {out}")
    return df


def download_all(force: bool = False):
    """Download all datasets."""
    dilist = download_dilist(force)
    proxy = download_proxy_dili(force)
    cmax = download_cmax(force)
    return dilist, proxy, cmax


if __name__ == "__main__":
    download_all(force=True)
