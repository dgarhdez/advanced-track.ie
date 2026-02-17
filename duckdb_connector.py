import duckdb
import os

con = duckdb.connect("IE_AIRPLANES.duckdb")

# iterate through all parquet files under data/parquet
parquet_dir = os.path.join(os.getcwd(), "data", "parquet")
for fname in os.listdir(parquet_dir):
    if not fname.lower().endswith(".parquet"):
        continue
    path = os.path.join(parquet_dir, fname)
    # derive a table name from the file name (strip extension, uppercase)
    table_name = os.path.splitext(fname)[0].upper()
    # create table if not exists by reading the parquet file
    con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM '{path}'")
    print(f"loaded {fname} into table {table_name}")