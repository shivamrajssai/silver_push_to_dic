"""Backfill the mapping ID columns in the EXISTING silver parquets (S3) from master_upc_project.csv.

For every survey (upc, cycle, phase) below it:
  1. takes region_id, piu_id, state_id, contractor_id, ae_ie_id, nsv_zone_id, dams_zone_id, das_zone_id
     from master_upc_project.csv (matched on upc)
  2. finds the 14 parquets under  DHMS/NSV/Silver/Zone=B/Phase=<p>/Cycle=<c>/UPC=<upc with / -> _>/
  3. in each parquet, for the rows of that survey, fills an ID column ONLY where it is NULL
     (values already there are never changed - differing ones are only reported)
  4. LIVE run: backs up the original file, writes the new file with the same schema / compression / row-group size,
     verifies it (schema, row count, null counts) BEFORE uploading, then overwrites the original key.

Columns that do not exist in a parquet are NOT added (the schema stays the same) - they are listed in the report.
DRY_RUN is True by default: S3 is only read, nothing is written.

HOW TO RUN:  1) leave DRY_RUN = True below and run it: report only, nothing is written
             2) check the numbers, set DRY_RUN = False, run again: it asks you to type YES, then backfills
             (the flags --dry-run true/false override the DRY_RUN setting; --yes skips the YES question)

Usage:
    python backfill_silver_ids.py                          # follows DRY_RUN below
    python backfill_silver_ids.py --dry-run false          # really backfill
    python backfill_silver_ids.py --only master_pothole.parquet --only geometry_child.parquet
    python backfill_silver_ids.py --trust-partition        # files without a upc column: treat all rows as that survey's

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
DRY_RUN = True        # True = report only. Set to False to really write to S3.

S3_BUCKET = "dhms-new-stgae"
MASTER_KEY = "NSV/DHMS/DIC_Staging/Latest/master_upc_project.csv"
S3_PREFIX = "DHMS/NSV/Silver/"
BACKUP_PREFIX = "DHMS/NSV/Silver_Backups/id_backfill/"      # originals are copied here before any overwrite

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


def load_master(s3):
    df = pd.read_csv(s3.get_object(Bucket=S3_BUCKET, Key=MASTER_KEY)["Body"], dtype=str)
    df.columns = df.columns.str.strip()
    return df


def master_values(master, upc):
    """({id_col: value}, {id_col: why not available}) for this upc; (None, {}) if the upc is not in the master."""
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
    """Boolean numpy mask of the rows that belong to this survey (None = cannot tell)."""
    if "upc" not in cmap:
        return np.ones(tbl.num_rows, dtype=bool) if trust_partition else None
    cols = sorted({cmap[c] for c in ("upc", "zone", "phase", "cycle") if c in cmap})
    df = tbl.select(cols).to_pandas()
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


def fill_column(tbl, name, value, mask):
    """Return (new column array, nulls in survey rows, rows filled, populated rows that differ from master)."""
    idx = tbl.schema.get_field_index(name)
    field = tbl.schema.field(idx)
    arr = tbl.column(idx).combine_chunks()
    m = pa.array(mask)
    nulls = pc.is_null(arr)
    n_null = count(pc.and_(m, nulls))
    if value is None:
        return arr, n_null, 0, 0
    v = re.sub(r"\.0+$", "", value) if pa.types.is_integer(field.type) else value
    fill = pa.scalar(v, type=pa.string()).cast(field.type)
    to_fill = pc.and_(m, nulls)
    differs = pc.and_(pc.and_(m, pc.invert(nulls)), pc.fill_null(pc.not_equal(arr, fill), False))
    return pc.if_else(to_fill, fill, arr), n_null, count(to_fill), count(differs)


def serialize(tbl, pf):
    md = pf.metadata
    comp, rgs = "snappy", None
    if md.num_row_groups:
        comp = md.row_group(0).column(0).compression.lower()
        comp = "none" if comp == "uncompressed" else comp
        rgs = max(md.row_group(0).num_rows, 1)
    buf = BytesIO()
    pq.write_table(tbl, buf, compression=comp, row_group_size=rgs)
    return buf.getvalue()


def verify(data, new_tbl, orig_schema, changed):
    back = pq.read_table(BytesIO(data))
    if not back.schema.equals(orig_schema, check_metadata=False):
        raise RuntimeError("schema would change on write")
    if back.num_rows != new_tbl.num_rows:
        raise RuntimeError("row count would change on write")
    for name in new_tbl.schema.names:
        if back[name].null_count != new_tbl[name].null_count:
            raise RuntimeError(f"null count differs after write in {name}")
    for name in changed:
        if not back[name].equals(new_tbl[name]):
            raise RuntimeError(f"values differ after write in {name}")


# ===================== ONE FILE =====================
def process_file(s3, key, filename, values, notes, survey, dry_run, trust_partition, run_ts, report):
    upc, cycle, phase = survey
    base = {"survey": upc, "zone": ZONE, "phase": phase, "cycle": cycle, "parquet_file": filename, "s3_key": key}

    def add(col="ALL", file_status="", status="", master_value=None, matched=0, null_before=0, filled=0, differs=0):
        report.append({**base, "id_column": col, "master_value": master_value, "matched_rows": matched,
                       "null_before": null_before, "filled": filled, "populated_but_differs": differs,
                       "status": status, "file_status": file_status})

    body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    pf = pq.ParquetFile(BytesIO(body))
    tbl = pf.read()
    cmap = {n.lower(): n for n in tbl.schema.names}
    mask = survey_mask(tbl, cmap, upc, phase, cycle, trust_partition)
    if mask is None:
        print("  UPC column missing - skipped (use --trust-partition to treat all rows as this survey's)")
        add(file_status="UPC column missing - skipped", status="skipped")
        return
    matched = int(mask.sum())

    new_tbl, changed, total = tbl, [], 0
    for col in ID_COLUMNS:
        if col not in cmap:
            add(col, status="column missing in parquet (not added)", master_value=values.get(col))
            continue
        name = cmap[col]
        try:
            arr, n_null, n_fill, n_diff = fill_column(tbl, name, values.get(col), mask)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as e:
            add(col, file_status="ERROR - nothing written", status=f"cannot cast master value: {e}",
                master_value=values.get(col), matched=matched)
            print(f"  !! {col}: cannot cast master value {values.get(col)!r} to the column type - file skipped")
            return
        if col in values:
            status = "filled" if n_fill else ("nothing to fill" if not n_null else "-")
        else:
            status = f"not filled: {notes.get(col, 'no master value')}"
        if n_fill:
            new_tbl = new_tbl.set_column(tbl.schema.get_field_index(name), tbl.schema.field(name), arr)
            changed.append(name)
        total += n_fill
        add(col, status=status, master_value=values.get(col), matched=matched, null_before=n_null, filled=n_fill,
            differs=n_diff)
        print(f"  {col:<14} master={values.get(col)!s:<8} null in survey rows={n_null:<8} "
              f"{'would fill' if dry_run else 'fill'}={n_fill:<8} populated-but-different={n_diff}")

    first = len(report) - len(ID_COLUMNS)
    def set_file_status(s):
        for r in report[first:]:
            r["file_status"] = s

    if not total:
        print(f"  matched rows {matched:,} - nothing to fill")
        set_file_status("nothing to fill")
        return
    if dry_run:
        print(f"  matched rows {matched:,} - would fill {total:,} cells (dry run)")
        set_file_status("would fill")
        return
    try:
        data = serialize(new_tbl, pf)
        verify(data, new_tbl, tbl.schema, changed)
        backup_key = f"{BACKUP_PREFIX}{run_ts}/{key}"
        s3.copy_object(Bucket=S3_BUCKET, Key=backup_key, CopySource={"Bucket": S3_BUCKET, "Key": key})
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data)
        print(f"  filled {total:,} cells; backup: s3://{S3_BUCKET}/{backup_key}")
        set_file_status(f"FILLED (backup {backup_key})")
    except Exception as e:
        print(f"  !! not written: {e}")
        set_file_status(f"ERROR - not written: {e}")
        for r in report[first:]:
            r["filled"] = 0


# ===================== MAIN =====================
def main():
    ap = argparse.ArgumentParser(description="Backfill ID columns in silver parquets from master_upc_project")
    ap.add_argument("--dry-run", choices=["true", "false"], default=None)
    ap.add_argument("--only", choices=PARQUET_FILES, action="append", help="process only this parquet (repeatable)")
    ap.add_argument("--yes", action="store_true", help="live run: do not ask for the YES confirmation")
    ap.add_argument("--trust-partition", action="store_true",
                    help="parquet without a upc column: treat all its rows as the survey of its S3 folder")
    args = ap.parse_args()
    dry_run = DRY_RUN if args.dry_run is None else args.dry_run == "true"
    files = [f for f in PARQUET_FILES if not args.only or f in args.only]
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 100)
    print("DRY RUN - S3 is only read, nothing is written" if dry_run else
          f"LIVE RUN - originals are backed up to s3://{S3_BUCKET}/{BACKUP_PREFIX}{run_ts}/ then overwritten")
    print("=" * 100)

    if not dry_run and not args.yes:
        if input("LIVE RUN: parquets in S3 will be overwritten (originals backed up). Type YES to continue: ") != "YES":
            print("Cancelled - nothing written.")
            return

    s3 = boto3.client("s3")
    master = load_master(s3)
    paginator = s3.get_paginator("list_objects_v2")
    report = []

    for upc, cycle, phase, survey_date in SURVEYS:
        survey = (upc, cycle, phase)
        print(f"\n{'#' * 100}\n# {upc} | zone {ZONE} | phase {phase} | cycle {cycle} | survey_date {survey_date}\n{'#' * 100}")
        values, notes = master_values(master, upc)
        if values is None:
            print("NOT FOUND in master_upc_project - skipped")
            report.append({"survey": upc, "zone": ZONE, "phase": phase, "cycle": cycle, "id_column": "ALL",
                           "file_status": "UPC not in master_upc_project", "status": "skipped"})
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
            base = {"survey": upc, "zone": ZONE, "phase": phase, "cycle": cycle, "parquet_file": filename,
                    "id_column": "ALL"}
            if not keys:
                print("  NOT FOUND")
                report.append({**base, "file_status": "parquet not found", "status": "skipped"})
                continue
            if len(keys) > 1:
                print(f"  !! {len(keys)} files with this name under the survey prefix - skipped: {keys}")
                report.append({**base, "file_status": f"ambiguous: {len(keys)} files", "status": "skipped"})
                continue
            try:
                process_file(s3, keys[0], filename, values, notes, survey, dry_run, args.trust_partition, run_ts,
                             report)
            except Exception as e:
                print(f"  !! error: {e}")
                report.append({**base, "s3_key": keys[0], "file_status": f"ERROR: {e}", "status": "error"})

    df = pd.DataFrame(report)
    dates = {(u, c, ph): d for u, c, ph, d in SURVEYS}
    df.insert(1, "survey_date", [dates.get((r.survey, r.cycle, r.phase)) for r in df.itertuples()])
    out = Path(__file__).resolve().parent / f"backfill_report_{'dry' if dry_run else 'live'}_{run_ts}.csv"
    df.to_csv(out, index=False)
    for c in ("matched_rows", "null_before", "filled", "populated_but_differs"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0).astype(int)
    per_file = df.groupby(["survey", "cycle", "phase", "parquet_file"], sort=False).agg(
        matched_rows=("matched_rows", "max"), cells_to_fill=("filled", "sum"),
        populated_but_differs=("populated_but_differs", "sum"), result=("file_status", "first")).reset_index()
    pd.set_option("display.width", 300, "display.max_colwidth", 60, "display.max_rows", None)
    print(f"\n{'=' * 100}\nSUMMARY PER FILE\n{'=' * 100}\n{per_file.to_string(index=False)}")
    print(f"\nTotal cells {'that would be filled' if dry_run else 'filled'}: {int(df['filled'].sum()):,}")
    print(f"Full per-column report: {out}")
    if dry_run:
        print("DRY RUN - nothing written. Run with --dry-run false to backfill.")


if __name__ == "__main__":
    main()
