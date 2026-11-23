
"""
Backfill mapping ID columns in existing Silver Parquet files on S3.

Features:
1. Reads mapping IDs from master_upc_project.csv.
2. Matches surveys using UPC, cycle, phase and zone.
3. Fills only NULL mapping IDs.
4. Converts incompatible Parquet columns to string when required.
5. Preserves existing populated values.
6. Updates the Parquet schema correctly after retyping.
7. Verifies the new Parquet before uploading.
8. Backs up original S3 files before overwriting.
9. Generates a detailed CSV audit report.

DRY RUN is enabled by default.

Usage:
    python -u 14_silver_ver3.py
    python -u 14_silver_ver3.py --dry-run true
    python -u 14_silver_ver3.py --dry-run false
    python -u 14_silver_ver3.py --only master_ravelling.parquet
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


# ============================================================
# CONFIGURATION
# ============================================================

DRY_RUN = True

S3_BUCKET = "dhms-new-stgae"

MASTER_KEY = (
    "NSV/DHMS/DIC_Staging/Latest/master_upc_project.csv"
)

S3_PREFIX = "DHMS/NSV/Silver/"

BACKUP_PREFIX = (
    "DHMS/NSV/Silver_Backups/id_backfill/"
)

TRUST_PARTITION = True

ADD_MISSING_ID_COLUMNS = False

ZONE = "B"

# UPC, cycle, phase, survey_date
SURVEYS = [
    ("N/02015/03002/GJ", 1, 1, None),
]

PARQUET_FILES = [
    "geometry_child.parquet",
    "geometry_parent.parquet",
    "master_chainage.parquet",
    "master_cracking_10m.parquet",
    "master_cracking_as_per_ca.parquet",
    "master_pothole.parquet",
    "master_ravelling_10m.parquet",
    "master_ravelling.parquet",
    "master_roughness_as_per_ca.parquet",
    "master_roughness_child_10m.parquet",
    "master_roughness_parent.parquet",
    "master_rutting_as_per_ca.parquet",
    "master_rutting_child_10m.parquet",
    "master_rutting_parent.parquet",
]

ID_COLUMNS = [
    "region_id",
    "piu_id",
    "state_id",
    "contractor_id",
    "ae_ie_id",
    "nsv_zone_id",
    "dams_zone_id",
    "das_zone_id",
]

MISSING_TEXT = {"", "nan", "none", "null", "<na>"}


# ============================================================
# MASTER UPC CSV
# ============================================================

def clean(value):
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in MISSING_TEXT:
        return None

    return value


def load_master(s3):

    response = s3.get_object(
        Bucket=S3_BUCKET,
        Key=MASTER_KEY
    )

    df = pd.read_csv(
        response["Body"],
        dtype=str
    )

    df.columns = df.columns.str.strip()

    if "upc" not in df.columns:
        raise RuntimeError(
            "upc column missing in master_upc_project.csv"
        )

    df["upc"] = df["upc"].astype(str).str.strip()

    return df


def master_values(master, upc):

    rows = master[
        master["upc"] == upc
    ]

    if rows.empty:
        return None, {}

    values = {}
    notes = {}

    for col in ID_COLUMNS:

        if col not in master.columns:
            notes[col] = "column missing in master"
            continue

        vals = {
            clean(v)
            for v in rows[col]
        } - {None}

        if len(vals) == 1:

            values[col] = vals.pop()

        elif not vals:

            notes[col] = "empty in master"

        else:

            notes[col] = (
                f"ambiguous master values: {sorted(vals)}"
            )

    return values, notes


# ============================================================
# PARQUET HELPERS
# ============================================================

def to_num(series):

    extracted = series.astype(str).str.extract(
        r"(\d+)"
    )[0]

    return pd.to_numeric(
        extracted,
        errors="coerce"
    )


def survey_mask(
    tbl,
    cmap,
    upc,
    phase,
    cycle,
    trust_partition
):

    if "upc" not in cmap:

        if trust_partition:
            return np.ones(
                tbl.num_rows,
                dtype=bool
            )

        return None

    cols = sorted({
        cmap[c]
        for c in ("upc", "zone", "phase", "cycle")
        if c in cmap
    })

    df = tbl.select(cols).to_pandas()

    def norm(series):

        return (
            series.astype(str)
            .str.strip()
            .str.upper()
            .str.replace("/", "_", regex=False)
        )

    m = (
        norm(df[cmap["upc"]])
        == upc.upper().replace("/", "_")
    )

    if "zone" in cmap:

        z = (
            df[cmap["zone"]]
            .astype(str)
            .str.upper()
            .str.replace(
                r"ZONE|[\s=_\-]",
                "",
                regex=True
            )
        )

        m &= z == ZONE

    if "phase" in cmap:

        m &= (
            to_num(df[cmap["phase"]])
            == phase
        )

    if "cycle" in cmap:

        m &= (
            to_num(df[cmap["cycle"]])
            == cycle
        )

    return m.to_numpy()


def describe_keys(tbl, cmap):

    for col in ("upc", "zone", "phase", "cycle"):

        if col in cmap:

            vals = pc.unique(
                tbl[cmap[col]]
            ).to_pylist()

            print(
                f"    {cmap[col]} "
                f"({len(vals)} distinct): {vals[:5]}"
            )

        else:

            print(
                f"    {col}: column not in file"
            )


def count(bool_arr):

    return int(
        np.count_nonzero(
            bool_arr.to_numpy(
                zero_copy_only=False
            )
        )
    )


# ============================================================
# SAFE ID COLUMN FILLING
# ============================================================

def fill_column(tbl, name, value, mask):

    """
    Fill NULL values only.

    If master UUID cannot fit the existing Parquet type,
    convert the complete column to string.

    Existing populated values are preserved.
    """

    idx = tbl.schema.get_field_index(name)

    if idx < 0:
        raise RuntimeError(
            f"{name}: column not found"
        )

    field = tbl.schema.field(idx)

    arr = tbl.column(idx).combine_chunks()

    m = pa.array(
        mask,
        type=pa.bool_()
    )

    n_null = count(
        pc.and_(
            m,
            pc.is_null(arr)
        )
    )

    if value is None:

        return arr, n_null, 0, 0, None

    v = str(value).strip()

    if not v:

        return arr, n_null, 0, 0, None

    retyped = None

    # --------------------------------------------------------
    # STEP 1: Try converting master value to existing type.
    # --------------------------------------------------------

    try:

        fill = pa.scalar(
            v,
            type=pa.string()
        ).cast(
            field.type,
            safe=True
        )

    except (
        pa.ArrowInvalid,
        pa.ArrowNotImplementedError,
        pa.ArrowTypeError,
        ValueError,
        TypeError
    ):

        # ----------------------------------------------------
        # STEP 2: Incompatible type: convert column to string.
        # ----------------------------------------------------

        retyped = str(field.type)

        arr = pc.cast(
            arr,
            pa.string()
        )

        fill = pa.scalar(
            v,
            type=pa.string()
        )

    # --------------------------------------------------------
    # STEP 3: Identify NULL values in the survey.
    # --------------------------------------------------------

    nulls = pc.is_null(arr)

    to_fill = pc.and_(
        m,
        nulls
    )

    # --------------------------------------------------------
    # STEP 4: Identify existing populated values that differ.
    # --------------------------------------------------------

    differs = pc.and_(
        pc.and_(
            m,
            pc.invert(nulls)
        ),
        pc.fill_null(
            pc.not_equal(arr, fill),
            False
        )
    )

    # --------------------------------------------------------
    # STEP 5: Fill NULL values only.
    # --------------------------------------------------------

    out = pc.if_else(
        to_fill,
        fill,
        arr
    )

    n_fill = count(to_fill)

    expected_nulls = arr.null_count - n_fill

    if out.null_count != expected_nulls:

        raise RuntimeError(
            f"{name}: null count after fill is incorrect"
        )

    return (
        out,
        n_null,
        n_fill,
        count(differs),
        retyped
    )


# ============================================================
# PARQUET SERIALIZATION
# ============================================================

def serialize(tbl, pf):

    md = pf.metadata

    compression = "snappy"
    row_group_size = None

    if md.num_row_groups:

        compression = (
            md.row_group(0)
            .column(0)
            .compression.lower()
        )

        if compression == "uncompressed":
            compression = "none"

        row_group_size = max(
            md.row_group(0).num_rows,
            1
        )

    buf = BytesIO()

    pq.write_table(
        tbl,
        buf,
        compression=compression,
        row_group_size=row_group_size
    )

    return buf.getvalue()


def verify(data, new_tbl, changed):

    back = pq.read_table(
        BytesIO(data)
    )

    # Schema verification
    if not back.schema.equals(
        new_tbl.schema,
        check_metadata=False
    ):

        raise RuntimeError(
            "Schema differs after serialization"
        )

    # Row count verification
    if back.num_rows != new_tbl.num_rows:

        raise RuntimeError(
            "Row count differs after serialization"
        )

    # NULL count verification
    for name in new_tbl.schema.names:

        if back[name].null_count != new_tbl[name].null_count:

            raise RuntimeError(
                f"NULL count differs after serialization: {name}"
            )

    # Changed column value verification
    for name in changed:

        if not back[name].equals(new_tbl[name]):

            raise RuntimeError(
                f"Values differ after serialization: {name}"
            )

    return True


# ============================================================
# PROCESS ONE PARQUET FILE
# ============================================================

def process_file(
    s3,
    key,
    filename,
    values,
    notes,
    survey,
    dry_run,
    trust_partition,
    run_ts,
    report
):

    upc, cycle, phase = survey

    base = {
        "survey": upc,
        "zone": ZONE,
        "phase": phase,
        "cycle": cycle,
        "parquet_file": filename,
        "s3_key": key
    }

    first = len(report)

    def add(
        col="ALL",
        status="",
        master_value=None,
        matched=0,
        null_before=0,
        filled=0,
        differs=0,
        old_type="",
        new_type=""
    ):

        report.append({
            **base,
            "id_column": col,
            "master_value": master_value,
            "matched_rows": matched,
            "null_before": null_before,
            "filled": filled,
            "populated_but_differs": differs,
            "old_type": old_type,
            "new_type": new_type,
            "status": status,
            "file_status": ""
        })

    def set_file_status(status):

        for r in report[first:]:
            r["file_status"] = status

    # --------------------------------------------------------
    # Read original S3 file.
    # --------------------------------------------------------

    body = s3.get_object(
        Bucket=S3_BUCKET,
        Key=key
    )["Body"].read()

    pf = pq.ParquetFile(
        BytesIO(body)
    )

    tbl = pf.read()

    cmap = {
        n.lower(): n
        for n in tbl.schema.names
    }

    # --------------------------------------------------------
    # Identify survey rows.
    # --------------------------------------------------------

    mask = survey_mask(
        tbl,
        cmap,
        upc,
        phase,
        cycle,
        trust_partition
    )

    if mask is None:

        print(
            "  UPC column missing - skipped"
        )

        add(
            status="UPC column missing - skipped"
        )

        set_file_status(
            "UPC column missing - skipped"
        )

        return

    matched = int(mask.sum())

    note = (
        " (all rows trusted from S3 partition)"
        if "upc" not in cmap
        else ""
    )

    print(
        f"  rows in file {tbl.num_rows:,} | "
        f"rows of this survey {matched:,}{note}"
    )

    if matched == 0 and tbl.num_rows:

        print(
            "  WARNING: no rows match this survey"
        )

        describe_keys(
            tbl,
            cmap
        )

    # --------------------------------------------------------
    # Process ID columns.
    # --------------------------------------------------------

    new_tbl = tbl

    changed = []

    total_filled = 0

    total_retyped = 0

    for col in ID_COLUMNS:

        # ----------------------------------------------------
        # Missing column.
        # ----------------------------------------------------

        if col not in cmap:

            if ADD_MISSING_ID_COLUMNS and col in values:

                new_tbl = new_tbl.append_column(
                    col,
                    pa.array(
                        [
                            values[col] if x else None
                            for x in mask
                        ],
                        type=pa.string()
                    )
                )

                changed.append(col)

                total_filled += matched

                add(
                    col,
                    status="column ADDED as string",
                    master_value=values[col],
                    matched=matched,
                    null_before=matched,
                    filled=matched,
                    new_type="string"
                )

                print(
                    f"  {col:<14} added as string "
                    f"for {matched:,} rows"
                )

            else:

                add(
                    col,
                    status="column missing in parquet - not added",
                    master_value=values.get(col),
                    matched=matched
                )

                print(
                    f"  {col:<14} column missing in parquet"
                )

            continue

        # ----------------------------------------------------
        # Existing column.
        # ----------------------------------------------------

        name = cmap[col]

        field = new_tbl.schema.field(
            new_tbl.schema.get_field_index(name)
        )

        original_type = str(field.type)

        try:

            arr, n_null, n_fill, n_diff, retyped = fill_column(
                tbl,
                name,
                values.get(col),
                mask
            )

        except Exception as e:

            print(
                f"  ERROR {col}: {e}"
            )

            add(
                col,
                status=f"error: {e}",
                master_value=values.get(col),
                matched=matched,
                old_type=original_type
            )

            set_file_status(
                "ERROR - nothing written"
            )

            return

        # ----------------------------------------------------
        # Update table when values are filled OR type changes.
        # This is the important fix.
        # ----------------------------------------------------

        if n_fill > 0 or retyped is not None:

            current_idx = new_tbl.schema.get_field_index(
                name
            )

            current_field = new_tbl.schema.field(
                current_idx
            )

            new_tbl = new_tbl.set_column(
                current_idx,
                pa.field(
                    name,
                    arr.type,
                    current_field.nullable,
                    current_field.metadata
                ),
                arr
            )

            if name not in changed:
                changed.append(name)

        total_filled += n_fill

        if retyped is not None:
            total_retyped += 1

        # ----------------------------------------------------
        # Status reporting.
        # ----------------------------------------------------

        if col in values:

            status = (
                "filled"
                if n_fill
                else "nothing to fill"
            )

            if retyped:

                status += (
                    f" | RETYPED {retyped} -> string"
                )

                if tbl[name].null_count < tbl.num_rows:

                    status += (
                        " | existing values converted to text"
                    )

        else:

            status = (
                "not filled: "
                + notes.get(col, "no master value")
            )

        add(
            col,
            status=status,
            master_value=values.get(col),
            matched=matched,
            null_before=n_null,
            filled=n_fill,
            differs=n_diff,
            old_type=original_type,
            new_type=str(arr.type)
        )

        print(
            f"  {col:<14} "
            f"type={original_type:<12} "
            f"NULL={n_null:<8} "
            f"{'would fill' if dry_run else 'fill'}={n_fill:<8} "
            f"different={n_diff}"
            + (
                f" [RETYPE {retyped} -> string]"
                if retyped else ""
            )
        )

    # --------------------------------------------------------
    # Nothing changed.
    # --------------------------------------------------------

    if not changed:

        print(
            f"  matched rows {matched:,} - nothing to change"
        )

        set_file_status(
            "nothing to fill"
        )

        return

    # --------------------------------------------------------
    # Dry run.
    # --------------------------------------------------------

    if dry_run:

        print(
            f"  matched rows {matched:,} | "
            f"cells to fill: {total_filled:,} | "
            f"columns to retype: {total_retyped}"
        )

        set_file_status(
            "DRY RUN - would update"
        )

        return

    # --------------------------------------------------------
    # LIVE WRITE.
    # --------------------------------------------------------

    try:

        print(
            "  Serializing updated Parquet..."
        )

        data = serialize(
            new_tbl,
            pf
        )

        print(
            "  Verifying updated Parquet..."
        )

        verify(
            data,
            new_tbl,
            changed
        )

        print(
            "  Verification successful"
        )

        # ----------------------------------------------------
        # Backup original.
        # ----------------------------------------------------

        backup_key = (
            f"{BACKUP_PREFIX}{run_ts}/{key}"
        )

        s3.copy_object(
            Bucket=S3_BUCKET,
            Key=backup_key,
            CopySource={
                "Bucket": S3_BUCKET,
                "Key": key
            }
        )

        print(
            f"  Backup created: {backup_key}"
        )

        # ----------------------------------------------------
        # Upload updated file.
        # ----------------------------------------------------

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=data
        )

        uploaded_size = s3.head_object(
            Bucket=S3_BUCKET,
            Key=key
        )["ContentLength"]

        if uploaded_size != len(data):

            raise RuntimeError(
                "Uploaded file size differs. "
                f"Restore from backup: {backup_key}"
            )

        print(
            f"  SUCCESS: filled {total_filled:,} cells"
        )

        print(
            f"  Retyped columns: {total_retyped}"
        )

        set_file_status(
            f"SUCCESS - backup: {backup_key}"
        )

    except Exception as e:

        print(
            f"  ERROR - not written: {e}"
        )

        set_file_status(
            f"ERROR - not written: {e}"
        )

        for r in report[first:]:

            r["filled"] = 0


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser(
        description="Backfill Silver Parquet mapping IDs"
    )

    ap.add_argument(
        "--dry-run",
        choices=["true", "false"],
        default=None
    )

    ap.add_argument(
        "--only",
        choices=PARQUET_FILES,
        action="append"
    )

    ap.add_argument(
        "--yes",
        action="store_true"
    )

    ap.add_argument(
        "--no-trust-partition",
        action="store_true"
    )

    args = ap.parse_args()

    dry_run = (
        DRY_RUN
        if args.dry_run is None
        else args.dry_run == "true"
    )

    trust_partition = (
        TRUST_PARTITION
        and not args.no_trust_partition
    )

    files = [
        f for f in PARQUET_FILES
        if not args.only or f in args.only
    ]

    run_ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    print("=" * 100)

    if dry_run:

        print(
            "DRY RUN - S3 is only read, nothing is written"
        )

    else:

        print(
            "LIVE RUN - original files will be backed up "
            "before overwriting"
        )

        print(
            f"Backup: s3://{S3_BUCKET}/{BACKUP_PREFIX}{run_ts}/"
        )

    print("=" * 100)

    # --------------------------------------------------------
    # Confirmation before live execution.
    # --------------------------------------------------------

    if not dry_run and not args.yes:

        confirmation = input(
            "\nLIVE RUN: S3 Parquet files will be overwritten. "
            "Type YES to continue: "
        )

        if confirmation != "YES":

            print(
                "Cancelled - nothing written."
            )

            return

    # --------------------------------------------------------
    # S3 connection.
    # --------------------------------------------------------

    s3 = boto3.client("s3")

    print(
        "\nLoading master_upc_project.csv..."
    )

    master = load_master(s3)

    print(
        f"Master records loaded: {len(master):,}"
    )

    paginator = s3.get_paginator(
        "list_objects_v2"
    )

    report = []

    # --------------------------------------------------------
    # Process surveys.
    # --------------------------------------------------------

    for upc, cycle, phase, survey_date in SURVEYS:

        survey = (
            upc,
            cycle,
            phase
        )

        print(
            f"\n{'#' * 100}\n"
            f"# {upc} | zone {ZONE} | "
            f"phase {phase} | cycle {cycle} | "
            f"survey_date {survey_date}\n"
            f"{'#' * 100}"
        )

        values, notes = master_values(
            master,
            upc
        )

        if values is None:

            print(
                "UPC NOT FOUND in master_upc_project"
            )

            report.append({
                "survey": upc,
                "zone": ZONE,
                "phase": phase,
                "cycle": cycle,
                "id_column": "ALL",
                "status": "UPC not found",
                "file_status": "skipped"
            })

            continue

        print(
            "\nMaster values:"
        )

        for col in ID_COLUMNS:

            print(
                f"  {col:<15}: "
                f"{values.get(col, notes.get(col, 'unavailable'))}"
            )

        # ----------------------------------------------------
        # Find survey S3 files.
        # ----------------------------------------------------

        prefix = (
            f"{S3_PREFIX}"
            f"Zone={ZONE}/"
            f"Phase={phase}/"
            f"Cycle={cycle}/"
            f"UPC={upc.replace('/', '_')}/"
        )

        found = {}

        for page in paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix=prefix
        ):

            for obj in page.get("Contents", []):

                name = obj["Key"].rsplit(
                    "/",
                    1
                )[-1]

                if name in files:

                    found.setdefault(
                        name,
                        []
                    ).append(
                        obj["Key"]
                    )

        print(
            f"\nS3 prefix: {prefix}"
        )

        print(
            f"Parquets found: {len(found)}/{len(files)}"
        )

        # ----------------------------------------------------
        # Process each Parquet.
        # ----------------------------------------------------

        for filename in files:

            keys = found.get(
                filename,
                []
            )

            print(
                f"\n[{filename}]"
            )

            base = {
                "survey": upc,
                "zone": ZONE,
                "phase": phase,
                "cycle": cycle,
                "parquet_file": filename,
                "id_column": "ALL"
            }

            if not keys:

                print(
                    "  NOT FOUND"
                )

                report.append({
                    **base,
                    "status": "not found",
                    "file_status": "parquet not found"
                })

                continue

            if len(keys) > 1:

                print(
                    f"  Multiple matching files found: {keys}"
                )

                report.append({
                    **base,
                    "status": "ambiguous",
                    "file_status": "multiple files found"
                })

                continue

            try:

                process_file(
                    s3=s3,
                    key=keys[0],
                    filename=filename,
                    values=values,
                    notes=notes,
                    survey=survey,
                    dry_run=dry_run,
                    trust_partition=trust_partition,
                    run_ts=run_ts,
                    report=report
                )

            except Exception as e:

                print(
                    f"  ERROR: {e}"
                )

                report.append({
                    **base,
                    "s3_key": keys[0],
                    "status": "error",
                    "file_status": f"ERROR: {e}"
                })

    # ========================================================
    # REPORT
    # ========================================================

    df = pd.DataFrame(report)

    dates = {
        (u, c, ph): d
        for u, c, ph, d in SURVEYS
    }

    df.insert(
        1,
        "survey_date",
        [
            dates.get(
                (r.survey, r.cycle, r.phase)
            )
            for r in df.itertuples()
        ]
    )

    out = (
        Path(__file__).resolve().parent
        / f"backfill_report_"
          f"{'dry' if dry_run else 'live'}_"
          f"{run_ts}.csv"
    )

    df.to_csv(
        out,
        index=False
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    for col in (
        "matched_rows",
        "null_before",
        "filled",
        "populated_but_differs"
    ):

        if col in df:

            df[col] = (
                pd.to_numeric(
                    df[col],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )

        else:

            df[col] = 0

    if "file_status" not in df:
        df["file_status"] = ""

    per_file = (
        df.groupby(
            [
                "survey",
                "cycle",
                "phase",
                "parquet_file"
            ],
            sort=False
        )
        .agg(
            matched_rows=(
                "matched_rows",
                "max"
            ),
            cells_to_fill=(
                "filled",
                "sum"
            ),
            populated_but_differs=(
                "populated_but_differs",
                "sum"
            ),
            result=(
                "file_status",
                "first"
            )
        )
        .reset_index()
    )

    pd.set_option(
        "display.width",
        300
    )

    pd.set_option(
        "display.max_colwidth",
        70
    )

    pd.set_option(
        "display.max_rows",
        None
    )

    print(
        f"\n{'=' * 100}\n"
        "SUMMARY PER FILE\n"
        f"{'=' * 100}"
    )

    print(
        per_file.to_string(
            index=False
        )
    )

    print(
        f"\nTotal cells {'that would be filled' if dry_run else 'filled'}: "
        f"{int(df['filled'].sum()):,}"
    )

    print(
        f"Full per-column audit report: {out}"
    )

    if dry_run:

        print(
            "\nDRY RUN - nothing written."
        )

        print(
            "Review the report before running with --dry-run false."
        )


if __name__ == "__main__":

    main()
