from __future__ import annotations

import numpy as np
import pandas as pd

from manuscript_paths import output_path


INPUT = output_path("pharmacological_specificity", "lsd_core_metric_tests.csv")
OUTPUT = output_path("pharmacological_specificity", "lsd_core_metric_tests_fdr.csv")


def fdr_bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def main() -> None:
    df = pd.read_csv(INPUT)
    if "PPaired" in df.columns:
        mask = df["PPaired"].notna()
        df.loc[mask, "QPaired"] = fdr_bh(df.loc[mask, "PPaired"])
    if "PLME" in df.columns:
        mask = df["PLME"].notna()
        df.loc[mask, "QLME"] = fdr_bh(df.loc[mask, "PLME"])
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
