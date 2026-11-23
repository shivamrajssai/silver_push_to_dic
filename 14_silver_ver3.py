"""Backfill the mapping ID columns in the EXISTING silver parquets (S3) from master_upc_project.csv.

For every survey (upc, cycle, phase) below it:
  1. takes region_id, piu_id, state_id, contractor_id, ae_ie_id, nsv_zone_id, dams_zone_id, das_zone_id
     from master_upc_project.csv (matched on upc)
  2. finds the 14 parquets under  DHMS/NSV/Silver/Zone=B/Phase=<p>/Cycle=<c>/UPC=<upc with / -> _>/
  3. in each parquet, for the rows of that survey, fills an ID column ONLY where it is NULL
     (values already there are never changed - differing ones are only reported)
  4. if the master value does not fit the column type (null / double / int ...), the column is retyped to string
     (existing values are kept as text) and this is written to the report
  5. LIVE run: backs up the original file, writes the new file with the same compression / row-group size,
     verifies it (schema, row count, null counts) BEFORE uploading, then overwrites the original key.

DRY_RUN is True by default: S3 is only read, nothing is written.

HOW TO RUN:  1) leave DRY_RUN = True and run: report only, nothing is written
             2) check the numbers, then run with --dry-run false: it asks you to type YES, then backfills

Usage:
    python 14_silver_ver2.py                          # follows DRY_RUN below
    python 14_silver_ver2.py --dry-run false          # really backfill
    python 14_silver_ver2.py --only master_pothole.parquet --only geometry_child.parquet
    python 14_silver_ver2.py --no-trust-partition     # files without a upc column are skipped

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

# Parquet has no upc column -> the file sits in a per-UPC/cycle/phase folder, so treat all its rows as that survey's.
TRUST_PARTITION = True
# True = add ID columns that a parquet does not have at all (as text). Leave False unless the DB table needs them.
ADD_MISSING_ID_COLUMNS = False

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
def to_num(series):
    """'1', 1.0, 'Phase 1', 'P1' -> 1 (first digits found), anything else -> NaN."""
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def survey_mask(tbl, cmap, upc, phase, cycle, trust_partition):
    """Boolean numpy mask of the rows that belong to this survey (None = cannot tell)."""
    if "upc" not in cmap:
        return np.ones(tbl.num_rows, dtype=bool) if trust_partition else None
    cols = sorted({cmap[c] for c in ("upc", "zone", "phase", "cycle") if c in cmap})
    df = tbl.select(cols).to_pandas()
    norm = lambda s: s.astype(str).str.strip().str.upper().str.replace("/", "_", regex=False)
    m = norm(df[cmap["upc"]]) == upc.upper().replace("/", "_")
    if "zone" in cmap:
        z = df[cmap["zone"]].astype(str).str.upper().str.replace(r"ZONE|[\s=_\-]", "", regex=True)
        m &= z == ZONE
    if "phase" in cmap:
        m &= to_num(df[cmap["phase"]]) == phase
    if "cycle" in cmap:
        m &= to_num(df[cmap["cycle"]]) == cycle
    return m.to_numpy()


def describe_keys(tbl, cmap):
    for c in ("upc", "zone", "phase", "cycle"):
        if c in cmap:
            vals = pc.unique(tbl[cmap[c]]).to_pylist()
            print(f"    {cmap[c]} in file ({len(vals)} distinct): {vals[:5]}")
        else:
            print(f"    {c}: column not in file")


def count(bool_arr):
    return int(np.count_nonzero(bool_arr.to_numpy(zero_copy_only=False)))


def fill_column(tbl, name, value, mask):
    """Return (new array, nulls in survey rows, rows filled, populated rows differing from master, old type if retyped)."""
    idx = tbl.schema.get_field_index(name)
    field = tbl.schema.field(idx)
    arr = tbl.column(idx).combine_chunks()
    m = pa.array(mask)
    n_null = count(pc.and_(m, pc.is_null(arr)))
    if value is None:
        return arr, n_null, 0, 0, None
    v = re.sub(r"\.0+$", "", value) if pa.types.is_integer(field.type) else value
    retyped = None
    try:
        fill = pa.scalar(v, type=pa.string()).cast(field.type)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        retyped = str(field.type)                 # master value is a UUID text: column becomes string
        arr = arr.cast(pa.string())               # existing values (if any) are kept as text
        fill = pa.scalar(v, type=pa.string())
    nulls = pc.is_null(arr)
    to_fill = pc.and_(m, nulls)
    differs = pc.and_(pc.and_(m, pc.invert(nulls)), pc.fill_null(pc.not_equal(arr, fill), False))
    out = pc.if_else(to_fill, fill, arr)
    n_fill = count(to_fill)
    if out.null_count != tbl.column(idx).null_count - n_fill:
        raise RuntimeError(f"{name}: null count after fill is not as expected")
    return out, n_null, n_fill, count(differs), retyped


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


def verify(data, new_tbl, changed):
    back = pq.read_table(BytesIO(data))
    if not back.schema.equals(new_tbl.schema, check_metadata=False):
        raise RuntimeError("schema differs after write")
    if back.num_rows != new_tbl.num_rows:
        raise RuntimeError("row count differs after write")
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
    first = len(report)

    def add(col="ALL", status="", master_value=None, matched=0, null_before=0, filled=0, differs=0,
            old_type="", new_type=""):
        report.append({**base, "id_column": col, "master_value": master_value, "matched_rows": matched,
                       "null_before": null_before, "filled": filled, "populated_but_differs": differs,
                       "old_type": old_type, "new_type": new_type, "status": status, "file_status": ""})

    def set_file_status(s):
        for r in report[first:]:
            r["file_status"] = s

    body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    pf = pq.ParquetFile(BytesIO(body))
    tbl = pf.read()
    cmap = {n.lower(): n for n in tbl.schema.names}
    mask = survey_mask(tbl, cmap, upc, phase, cycle, trust_partition)
    if mask is None:
        print("  UPC column missing - skipped (trust-partition is off)")
        add(status="skipped")
        set_file_status("UPC column missing - skipped")
        return
    matched = int(mask.sum())
    note = " (no upc column: all rows trusted from the partition folder)" if "upc" not in cmap else ""
    print(f"  rows in file {tbl.num_rows:,} | rows of this survey {matched:,}{note}")
    if matched == 0 and tbl.num_rows:
        print("  !! no rows match this survey - values in the file:")
        describe_keys(tbl, cmap)

    new_tbl, changed, total = tbl, [], 0
    for col in ID_COLUMNS:
        if col not in cmap:
            if ADD_MISSING_ID_COLUMNS and col in values:
                new_tbl = new_tbl.append_column(col, pa.array([values[col] if x else None for x in mask], pa.string()))
                changed.append(col)
                total += matched
                add(col, status="column ADDED (text)", master_value=values[col], matched=matched,
                    null_before=matched, filled=matched, new_type="string")
                print(f"  {col:<14} column missing in parquet -> {'would add' if dry_run else 'add'} as text, "
                      f"{matched:,} rows")
            else:
                add(col, status="column missing in parquet (not added)", master_value=values.get(col),
                    matched=matched)
                print(f"  {col:<14} column missing in parquet (not added)")
            continue
        name = cmap[col]
        field = tbl.schema.field(name)
        try:
            arr, n_null, n_fill, n_diff, retyped = fill_column(tbl, name, values.get(col), mask)
        except Exception as e:
            print(f"  !! {col}: type {field.type}, master value {values.get(col)!r} - file skipped ({e})")
            add(col, status=f"error: {e}", master_value=values.get(col), matched=matched, old_type=str(field.type))
            set_file_status("ERROR - nothing written")
            return
        if col in values:
            status = "filled" if n_fill else "nothing to fill (no NULLs in survey rows)"
            if retyped:
                had_values = tbl[name].null_count < tbl.num_rows
                status += f" | retyped {retyped} -> string" + (" | WARNING: existing values converted to text"
                                                               if had_values else "")
        else:
            status = f"not filled: {notes.get(col, 'no master value')}"
        if n_fill:
            new_tbl = new_tbl.set_column(tbl.schema.get_field_index(name),
                                         pa.field(name, arr.type, field.nullable, field.metadata), arr)
            changed.append(name)
        total += n_fill
        add(col, status=status, master_value=values.get(col), matched=matched, null_before=n_null, filled=n_fill,
            differs=n_diff, old_type=str(field.type), new_type=str(arr.type) if n_fill else str(field.type))
        print(f"  {col:<14} type={str(field.type):<10} null in survey rows={n_null:<8} "
              f"{'would fill' if dry_run else 'fill'}={n_fill:<8} populated-but-different={n_diff}"
              + (f"  [type {retyped} -> string]" if retyped and n_fill else ""))

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
        verify(data, new_tbl, changed)
        backup_key = f"{BACKUP_PREFIX}{run_ts}/{key}"
        s3.copy_object(Bucket=S3_BUCKET, Key=backup_key, CopySource={"Bucket": S3_BUCKET, "Key": key})
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data)
        if s3.head_object(Bucket=S3_BUCKET, Key=key)["ContentLength"] != len(data):
            raise RuntimeError("uploaded size differs from local file - restore from backup " + backup_key)
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
    ap.add_argument("--no-trust-partition", action="store_true",
                    help="skip parquets that have no upc column instead of trusting the S3 folder")
    args = ap.parse_args()
    dry_run = DRY_RUN if args.dry_run is None else args.dry_run == "true"
    trust_partition = TRUST_PARTITION and not args.no_trust_partition
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
                process_file(s3, keys[0], filename, values, notes, survey, dry_run, trust_partition, run_ts, report)
            except Exception as e:
                print(f"  !! error: {e}")
                report.append({**base, "s3_key": keys[0], "file_status": f"ERROR: {e}", "status": "error"})

    df = pd.DataFrame(report)
    dates = {(u, c, ph): d for u, c, ph, d in SURVEYS}
    df.insert(1, "survey_date", [dates.get((r.survey, r.cycle, r.phase)) for r in df.itertuples()])
    out = Path(__file__).resolve().parent / f"backfill_report_{'dry' if dry_run else 'live'}_{run_ts}.csv"
    df.to_csv(out, index=False)
    for c in ("matched_rows", "null_before", "filled", "populated_but_differs"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int) if c in df else 0
    per_file = df.groupby(["survey", "cycle", "phase", "parquet_file"], sort=False).agg(
        matched_rows=("matched_rows", "max"), cells_to_fill=("filled", "sum"),
        populated_but_differs=("populated_but_differs", "sum"), result=("file_status", "first")).reset_index()
    pd.set_option("display.width", 300, "display.max_colwidth", 60, "display.max_rows", None)
    print(f"\n{'=' * 100}\nSUMMARY PER FILE\n{'=' * 100}\n{per_file.to_string(index=False)}")
    print(f"\nTotal cells {'that would be filled' if dry_run else 'filled'}: {int(df['filled'].sum()):,}")
    print(f"Full per-column audit report: {out}")
    if dry_run:
        print("DRY RUN - nothing written. Run with --dry-run false to backfill.")


if __name__ == "__main__":
    main()
