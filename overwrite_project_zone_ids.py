
import argparse
import io
import boto3
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

S3_BUCKET = "dhms-new-stgae"

UPC = "N_02015_03002_GJ"
ZONE = "B"
PHASE = 1
CYCLE = 1

PREFIX = (
    f"DHMS/NSV/Silver/Zone={ZONE}/"
    f"Phase={PHASE}/Cycle={CYCLE}/UPC={UPC}/"
)

BACKUP_PREFIX = "DHMS/NSV/Silver_Backups/project_zone_id_overwrite/"

PROJECT_ID = "a28a054f-d34e-4950-85fc-38fcf9ce9bb0"
ZONE_ID = "0acf9d55-b081-4012-998a-e490fb016db0"

DRY_RUN = True

# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--write",
    action="store_true",
    help="Actually write changes to S3. Without this, only dry run."
)
args = parser.parse_args()

if args.write:
    DRY_RUN = False

# --------------------------------------------------
# S3 CONNECTION
# --------------------------------------------------

s3 = boto3.client("s3")

print("=" * 90)
print("PROJECT ID AND ZONE ID OVERWRITE")
print("=" * 90)
print(f"Bucket:     {S3_BUCKET}")
print(f"UPC:        {UPC}")
print(f"Project ID: {PROJECT_ID}")
print(f"Zone ID:    {ZONE_ID}")
print(f"Dry run:    {DRY_RUN}")
print("=" * 90)

# --------------------------------------------------
# FIND PARQUET FILES
# --------------------------------------------------

paginator = s3.get_paginator("list_objects_v2")

files = []

for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        key = obj["Key"]

        if key.endswith(".parquet"):
            files.append(key)

if not files:
    print("No Parquet files found.")
    raise SystemExit(0)

print(f"\nFound {len(files)} Parquet files.\n")

# --------------------------------------------------
# UPDATE PROJECT ID AND ZONE ID
# --------------------------------------------------

for key in files:

    print(f"\nProcessing: {key}")

    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        original_data = response["Body"].read()

        table = pq.read_table(io.BytesIO(original_data))
        row_count = table.num_rows

        for column_name, fixed_value in [
            ("project_id", PROJECT_ID),
            ("zone_id", ZONE_ID),
        ]:

            new_array = pa.array(
                [fixed_value] * row_count,
                type=pa.string()
            )

            if column_name in table.column_names:
                column_index = table.schema.get_field_index(column_name)
                table = table.set_column(
                    column_index,
                    pa.field(column_name, pa.string()),
                    new_array
                )
            else:
                table = table.append_column(column_name, new_array)

        # Serialize updated Parquet
        output = io.BytesIO()
        pq.write_table(table, output)
        updated_data = output.getvalue()

        # Verify updated file before upload
        verified = pq.read_table(io.BytesIO(updated_data))

        assert all(
            value == PROJECT_ID
            for value in verified["project_id"].to_pylist()
        )

        assert all(
            value == ZONE_ID
            for value in verified["zone_id"].to_pylist()
        )

        print(f"Rows: {row_count}")
        print("project_id: VERIFIED")
        print("zone_id: VERIFIED")

        if DRY_RUN:
            print("DRY RUN - Would overwrite this file")

        else:
            # Backup original file first
            backup_key = BACKUP_PREFIX + key.split("/")[-1]

            s3.copy_object(
                Bucket=S3_BUCKET,
                CopySource={"Bucket": S3_BUCKET, "Key": key},
                Key=backup_key
            )

            # Upload updated Parquet
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=updated_data
            )

            print(f"BACKUP: {backup_key}")
            print("SUCCESS - File overwritten")

    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "=" * 90)
print("PROCESS COMPLETED")
print("=" * 90)

if DRY_RUN:
    print("No S3 files were modified.")
    print("After reviewing the output, run with --write to apply changes.")
