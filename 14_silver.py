
import boto3
import pandas as pd
import pyarrow.parquet as pq
from io import BytesIO

# ===================== S3 CONFIGURATION =====================

S3_BUCKET = "dhms-new-stgae"
S3_PREFIX = "DHMS/NSV/Silver/"

# ===================== SURVEY INPUT =====================

ZONE = "B"

SURVEYS = [
    ("N/01002/01001/GJ", 1, 1),
    ("N/01002/01001/GJ", 2, 1),
    ("N/02001/12001/GJ", 1, 1),
    ("N/04006/03001/GJ", 2, 1),
    ("N/08009/04001/GJ", 1, 2),
    ("N/08009/04002/GJ", 1, 2),
    ("N/08009/05001/GJ", 1, 2),
    ("N/08009/07002/GJ", 1, 2),
    ("N/08009/08002/GJ", 1, 2),
    ("N/08081/01002/GJ", 2, 1)
]

# ===================== PARQUET FILES =====================

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
    "master_rutting_parent.parquet"
]

# ===================== MAPPING COLUMNS =====================

ID_COLUMNS = [
    "region_id",
    "piu_id",
    "state_id",
    "contractor_id",
    "ae_ie_id",
    "nsv_zone_id",
    "dams_zone_id",
    "das_zone_id"
]

# ===================== S3 CLIENT =====================

s3 = boto3.client("s3")

paginator = s3.get_paginator("list_objects_v2")

schema_report = []
id_report = []

schema_seen = set()

# ===================== COUNTERS =====================

total_files_found = 0
total_files_missing = 0
total_matched_rows = 0

print("\n" + "=" * 100)
print("       S3 SILVER PARQUET MAPPING ID INSPECTION")
print("=" * 100)

print("\nTotal survey combinations:", len(SURVEYS))
print("Total unique UPCs:", len(set(x[0] for x in SURVEYS)))
print("Total Parquet files to inspect:", len(PARQUET_FILES))

# ===================== SURVEY-WISE SEARCH =====================

for upc, cycle, phase in SURVEYS:

    # Convert UPC for S3 folder naming
    s3_upc = upc.replace("/", "_")

    # Construct exact survey partition prefix
    survey_prefix = (
        f"{S3_PREFIX}"
        f"Zone={ZONE}/"
        f"Phase={phase}/"
        f"Cycle={cycle}/"
        f"UPC={s3_upc}/"
    )

    print("\n" + "=" * 100)
    print(f"UPC: {upc}")
    print(f"ZONE: {ZONE} | PHASE: {phase} | CYCLE: {cycle}")
    print(f"S3 PREFIX: {survey_prefix}")
    print("=" * 100)

    try:

        # ===================== RECURSIVE S3 SEARCH =====================

        found_objects = {}

        for page in paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix=survey_prefix
        ):

            for obj in page.get("Contents", []):

                key = obj["Key"]

                filename = key.rsplit("/", 1)[-1]

                if filename in PARQUET_FILES:
                    found_objects[filename] = key

        # ===================== FILE AVAILABILITY =====================

        print("\nPARQUET FILE SEARCH RESULT:")

        for filename in PARQUET_FILES:

            if filename in found_objects:

                print(f"[FOUND] {filename}")

            else:

                print(f"[NOT FOUND] {filename}")

                total_files_missing += 1

                # Record missing file
                id_report.append({
                    "parquet_file": filename,
                    "id_column": "ALL",
                    "survey": upc,
                    "zone": ZONE,
                    "phase": phase,
                    "cycle": cycle,
                    "matched_rows": 0,
                    "null_count": "N/A",
                    "status": "Parquet file not found"
                })

        # ===================== INSPECT FOUND FILES =====================

        for filename in PARQUET_FILES:

            if filename not in found_objects:
                continue

            key = found_objects[filename]

            total_files_found += 1

            print("\n" + "-" * 90)
            print(f"FILE: {filename}")
            print(f"S3 KEY: {key}")
            print("-" * 90)

            try:

                # ===================== READ PARQUET =====================

                obj = s3.get_object(
                    Bucket=S3_BUCKET,
                    Key=key
                )

                parquet = pq.ParquetFile(
                    BytesIO(obj["Body"].read())
                )

                columns = parquet.schema.names

                column_map = {
                    col.lower(): col
                    for col in columns
                }

                # ===================== SCHEMA REPORT =====================

                for col in columns:

                    schema_key = (filename, col)

                    if schema_key not in schema_seen:

                        schema_seen.add(schema_key)

                        schema_report.append({
                            "parquet_file": filename,
                            "column_name": col
                        })

                # ===================== MAPPING COLUMNS =====================

                found_ids = [
                    col for col in ID_COLUMNS
                    if col in column_map
                ]

                missing_ids = [
                    col for col in ID_COLUMNS
                    if col not in column_map
                ]

                print("\nMAPPING COLUMNS FOUND:")
                print(found_ids)

                print("\nMAPPING COLUMNS MISSING:")
                print(missing_ids)

                # ===================== UPC COLUMN =====================

                upc_col = column_map.get("upc")

                if not upc_col:

                    print("\nUPC column missing. Skipping inspection.")

                    for col in found_ids:

                        id_report.append({
                            "parquet_file": filename,
                            "id_column": col,
                            "survey": upc,
                            "zone": ZONE,
                            "phase": phase,
                            "cycle": cycle,
                            "matched_rows": "N/A",
                            "null_count": "N/A",
                            "status": "UPC column missing"
                        })

                    continue

                # ===================== REQUIRED COLUMNS =====================

                read_cols = found_ids.copy()

                read_cols.append(upc_col)

                for col in ["zone", "phase", "cycle"]:

                    if col in column_map:
                        read_cols.append(column_map[col])

                read_cols = list(dict.fromkeys(read_cols))

                # ===================== ROW GROUP INSPECTION =====================

                file_matched_rows = 0

                print("\nTOTAL ROW GROUPS:", parquet.num_row_groups)

                for rg in range(parquet.num_row_groups):

                    table = parquet.read_row_group(
                        rg,
                        columns=read_cols
                    )

                    batch = table.to_pandas()

                    # Filter target UPC
                    batch = batch[
                        batch[upc_col]
                        .astype(str)
                        .str.strip() == upc
                    ]

                    if batch.empty:
                        continue

                    # Filter zone
                    if "zone" in column_map:

                        batch = batch[
                            batch[column_map["zone"]]
                            .astype(str)
                            .str.strip()
                            .str.upper() == ZONE
                        ]

                    # Filter phase
                    if "phase" in column_map:

                        batch = batch[
                            batch[column_map["phase"]]
                            .astype(str) == str(phase)
                        ]

                    # Filter cycle
                    if "cycle" in column_map:

                        batch = batch[
                            batch[column_map["cycle"]]
                            .astype(str) == str(cycle)
                        ]

                    if batch.empty:
                        continue

                    matched = len(batch)

                    file_matched_rows += matched
                    total_matched_rows += matched

                    # ===================== ID NULL CHECK =====================

                    for col in found_ids:

                        actual_col = column_map[col]

                        nulls = batch[actual_col].isna().sum()

                        id_report.append({
                            "parquet_file": filename,
                            "id_column": col,
                            "survey": upc,
                            "zone": ZONE,
                            "phase": phase,
                            "cycle": cycle,
                            "matched_rows": matched,
                            "null_count": nulls,
                            "status": (
                                "NULL - Candidate"
                                if nulls
                                else "Already Populated"
                            )
                        })

                print("\nMATCHED ROWS:", file_matched_rows)

                print("\nFile inspection completed.")

            except Exception as e:

                print(f"FILE INSPECTION ERROR: {e}")

                id_report.append({
                    "parquet_file": filename,
                    "id_column": "ALL",
                    "survey": upc,
                    "zone": ZONE,
                    "phase": phase,
                    "cycle": cycle,
                    "matched_rows": "N/A",
                    "null_count": "N/A",
                    "status": f"Inspection error: {str(e)}"
                })

    except Exception as e:

        print(f"SURVEY SEARCH ERROR: {e}")

# ===================== FINAL ID REPORT =====================

print("\n\n" + "=" * 110)
print("                    FINAL PARQUET ID INSPECTION REPORT")
print("=" * 110)

if id_report:

    result = pd.DataFrame(id_report)

    print("\nMAPPING ID STATUS:\n")

    print(result.to_string(index=False))

    result.to_csv(
        "parquet_id_inspection.csv",
        index=False
    )

    print("\nSaved: parquet_id_inspection.csv")

else:

    print("\nNo inspection records generated.")

# ===================== FINAL SCHEMA REPORT =====================

if schema_report:

    schema = pd.DataFrame(schema_report)

    schema.to_csv(
        "parquet_all_columns.csv",
        index=False
    )

    print("Saved: parquet_all_columns.csv")

# ===================== SUMMARY =====================

print("\n" + "=" * 110)
print("                         EXECUTION SUMMARY")
print("=" * 110)

print("Survey combinations searched:", len(SURVEYS))
print("Unique UPCs:", len(set(x[0] for x in SURVEYS)))
print("Total expected Parquet files:", len(SURVEYS) * len(PARQUET_FILES))
print("Parquet files found:", total_files_found)
print("Parquet files not found:", total_files_missing)
print("Total matched rows:", total_matched_rows)

print("\nOutput files:")
print("1. parquet_id_inspection.csv")
print("2. parquet_all_columns.csv")

print("\nData modified: NO")

print("\nINSPECTION COMPLETED")
print("=" * 110)

