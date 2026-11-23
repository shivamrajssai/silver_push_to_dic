"""DRY RUN ONLY: shows what would be backfilled in the silver parquets from master_upc_project.csv.

This script has no write code at all: it only lists / reads S3 objects. Nothing is changed, nothing is uploaded.

Survey:  upc N/02015/03002/GJ | cycle 1 | phase 1 | survey_date (fill in below if you want it in the report)

For each of the 14 parquets in  DHMS/NSV/Silver/Zone=B/Phase=1/Cycle=1/UPC=N_02015_03002_GJ/  it reports, per ID column:
  - the value from master_upc_project.csv
  - how many rows of the survey are in the file
  - how many of them are NULL in that column (these are the cells that WOULD be filled)
  - how many already hold a value that differs from the master (never touched, only reported)

Usage:  python3 backfill_silver_ids_dryrun.py [--only master_pothole.parquet] [--trust-partition]
Install: pip install boto3 pandas pyarrow numpy
"""
import argparse
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ===================== CONFIG =====================
DRY_RUN = True                      # fixed: this script cannot write

S3_BUCKET = "dhms-new-stgae"
MASTER_KEY = "NSV/DHMS/DIC_Staging/Latest/master_upc_project.csv"
S3_PREFIX = "DHMS/NSV/Silver/"

ZONE = "B"
#            upc                 cycle  phase  survey_date
SURVEYS = [("N/02015/03002/GJ",   1,     1,     None)]       # <- put the survey_date here, e.g. "2025-11-05"

PARQUET_FILES = [
    "geometry_child.parquet", "geometry_parent.parquet", "master_chainage.parquet", "master_cracking_10m.parquet",
    "master_cracking_as_per_ca.parquet", "master_pothole.parquet", "master_ravelling_10m.parquet",
    "master_ravelling.parquet", "master_roughness_as_per_ca.parquet", "master_roughness_child_10m.parquet",
    "master_roughness_parent.parquet", "master_rutting_as_per_ca.parquet", "master_rutting_child_10m.parquet",
    "master_rutting_parent.parquet",
]
ID_COLUMNS = ["region_id", "piu_id", "state_id", "contractor_id", "ae_ie_id", "nsv_zone_id", "dams_zone_id",
              "das_zone_id"]
MISSING_TEXT = {"", "nan", "none", "null", "<na>"}


# ===================== MASTER UPC =====================
def clean(v):
    s = None if v is None else str(v).strip()
    return None if s is None or s.lower() in MISSING_TEXT else s


def master_values(master, upc):
    """({id_col: value}, {id_col: why not available}); (None, {}) if the upc is not in the master."""
    rows = master[master["upc"].str.strip() == upc]
    if rows.empty:
        return None, {}
    values, notes = {}, {}
    for col in ID_COLUMNS:
        if col not in master.columns:
            notes[col] = "column missing in master_upc_project"
            continue
        vals = {clean(v) for v in rows[col]} - {None}
        if len(vals) == 1:
            values[col] = vals.pop()
        elif not vals:
            notes[col] = "empty in master_upc_project"
        else:
            notes[col] = f"ambiguous in master_upc_project {sorted(vals)}"
    return values, notes


# ===================== PARQUET =====================
def survey_mask(tbl, cmap, upc, phase, cycle, trust_partition):
    """Boolean numpy mask of the rows of this survey (None = cannot tell)."""
    if "upc" not in cmap:
        return np.ones(tbl.num_rows, dtype=bool) if trust_partition else None
    df = tbl.select(sorted({cmap[c] for c in ("upc", "zone", "phase", "cycle") if c in cmap})).to_pandas()
    m = df[cmap["upc"]].astype(str).str.strip() == upc
    if "zone" in cmap:
        m &= df[cmap["zone"]].astype(str).str.strip().str.upper() == ZONE
    if "phase" in cmap:
        m &= pd.to_numeric(df[cmap["phase"]], errors="coerce") == phase
    if "cycle" in cmap:
        m &= pd.to_numeric(df[cmap["cycle"]], errors="coerce") == cycle
    return m.to_numpy()


def count(bool_arr):
    return int(np.count_nonzero(bool_arr.to_numpy(zero_copy_only=False)))


def inspect_column(tbl, name, value, mask):
    """(survey rows where the column is NULL, rows that would be filled, populated rows that differ from master)."""
    field = tbl.schema.field(name)
    arr = tbl.column(name).combine_chunks()
    m = pa.array(mask)
    nulls = pc.is_null(arr)
    n_null = count(pc.and_(m, nulls))
    if value is None:
        return n_null, 0, 0
    v = re.sub(r"\.0+$", "", value) if pa.types.is_integer(field.type) else value
    fill = pa.scalar(v, type=pa.string()).cast(field.type)          # raises if master value does not fit the column
    differs = pc.and_(pc.and_(m, pc.invert(nulls)), pc.fill_null(pc.not_equal(arr, fill), False))
    return n_null, n_null, count(differs)


# ===================== MAIN =====================
def main():
    ap = argparse.ArgumentParser(description="DRY RUN: what would be backfilled in the silver parquets")
    ap.add_argument("--only", choices=PARQUET_FILES, action="append", help="inspect only this parquet (repeatable)")
    ap.add_argument("--trust-partition", action="store_true",
                    help="parquet without a upc column: treat all its rows as the survey of its S3 folder")
    args = ap.parse_args()
    files = [f for f in PARQUET_FILES if not args.only or f in args.only]
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 100)
    print("DRY RUN ONLY - S3 is read, nothing is written or changed")
    print("=" * 100)

    s3 = boto3.client("s3")
    master = pd.read_csv(s3.get_object(Bucket=S3_BUCKET, Key=MASTER_KEY)["Body"], dtype=str)
    master.columns = master.columns.str.strip()
    paginator = s3.get_paginator("list_objects_v2")
    report = []

    for upc, cycle, phase, survey_date in SURVEYS:
        print(f"\n{'#' * 100}\n# {upc} | zone {ZONE} | phase {phase} | cycle {cycle} | survey_date {survey_date}\n{'#' * 100}")
        values, notes = master_values(master, upc)
        if values is None:
            print("NOT FOUND in master_upc_project - nothing to backfill")
            continue
        print("Master values:", {c: values.get(c, f"({notes.get(c)})") for c in ID_COLUMNS})

        prefix = f"{S3_PREFIX}Zone={ZONE}/Phase={phase}/Cycle={cycle}/UPC={upc.replace('/', '_')}/"
        found = {}
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                name = obj["Key"].rsplit("/", 1)[-1]
                if name in files:
                    found.setdefault(name, []).append(obj["Key"])
        print(f"S3 prefix: {prefix}   ({len(found)}/{len(files)} parquets found)")

        for filename in files:
            keys = found.get(filename, [])
            print(f"\n[{filename}]")
            base = {"survey": upc, "survey_date": survey_date, "zone": ZONE, "phase": phase, "cycle": cycle,
                    "parquet_file": filename}
            if not keys:
                print("  NOT FOUND")
                report.append({**base, "id_column": "ALL", "status": "parquet not found"})
                continue
            if len(keys) > 1:
                print(f"  !! {len(keys)} files with this name under the survey prefix: {keys}")
                report.append({**base, "id_column": "ALL", "status": f"ambiguous: {len(keys)} files"})
                continue
            try:
                tbl = pq.read_table(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=keys[0])["Body"].read()))
                cmap = {n.lower(): n for n in tbl.schema.names}
                mask = survey_mask(tbl, cmap, upc, phase, cycle, args.trust_partition)
                if mask is None:
                    print("  UPC column missing - skipped (use --trust-partition to treat all rows as this survey's)")
                    report.append({**base, "id_column": "ALL", "status": "UPC column missing"})
                    continue
                matched = int(mask.sum())
                print(f"  rows in file {tbl.num_rows:,} | rows of this survey {matched:,}")
                for col in ID_COLUMNS:
                    if col not in cmap:
                        print(f"  {col:<14} column missing in parquet (would not be added)")
                        report.append({**base, "id_column": col, "master_value": values.get(col),
                                       "status": "column missing in parquet"})
                        continue
                    try:
                        n_null, n_fill, n_diff = inspect_column(tbl, cmap[col], values.get(col), mask)
                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as e:
                        print(f"  {col:<14} !! master value {values.get(col)!r} does not fit the column type: {e}")
                        report.append({**base, "id_column": col, "master_value": values.get(col),
                                       "matched_rows": matched, "status": f"cannot cast master value: {e}"})
                        continue
                    if col in values:
                        status = "would fill" if n_fill else "nothing to fill"
                    else:
                        status = f"no master value: {notes.get(col)}"
                    print(f"  {col:<14} master={values.get(col)!s:<8} null={n_null:<8} would fill={n_fill:<8} "
                          f"populated-but-different={n_diff}")
                    report.append({**base, "id_column": col, "master_value": values.get(col),
                                   "matched_rows": matched, "null_count": n_null, "would_fill": n_fill,
                                   "populated_but_differs": n_diff, "status": status})
            except Exception as e:
                print(f"  !! error: {e}")
                report.append({**base, "id_column": "ALL", "status": f"error: {e}"})

    df = pd.DataFrame(report)
    if df.empty:
        print("\nNothing to report.")
        return
    out = Path(__file__).resolve().parent / f"backfill_dryrun_report_{run_ts}.csv"
    df.to_csv(out, index=False)
    for c in ("matched_rows", "null_count", "would_fill", "populated_but_differs"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0).astype(int)
    per_file = df.groupby("parquet_file", sort=False).agg(
        matched_rows=("matched_rows", "max"), cells_would_fill=("would_fill", "sum"),
        populated_but_differs=("populated_but_differs", "sum")).reset_index()
    pd.set_option("display.width", 250, "display.max_rows", None)
    print(f"\n{'=' * 100}\nSUMMARY PER FILE\n{'=' * 100}\n{per_file.to_string(index=False)}")
    print(f"\nTotal cells that WOULD be filled: {int(df['would_fill'].sum()):,}")
    print(f"Full per-column report: {out}")
    print("DRY RUN - nothing was written.")


if __name__ == "__main__":
    main()
