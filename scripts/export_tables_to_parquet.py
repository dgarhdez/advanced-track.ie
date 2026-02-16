from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database_connector import DatabaseConnector


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def discover_tables(connector: DatabaseConnector, schema: str) -> list[str]:
    query = """
        SELECT tabname
        FROM SYSCAT.TABLES
        WHERE tabschema = :schema_name
          AND type IN ('T', 'N')
        ORDER BY tabname
    """
    tables_df = connector.execute_query(query, params={"schema_name": schema})
    tables_df.columns = [column.lower() for column in tables_df.columns]
    return [str(value) for value in tables_df["tabname"].dropna().tolist()]


def export_table(
    connector: DatabaseConnector,
    schema: str,
    table: str,
    limit: int,
    output_dir: Path,
) -> tuple[Path, int]:
    qualified_name = f"{quote_identifier(schema)}.{quote_identifier(table)}"
    query = f"SELECT * FROM {qualified_name} FETCH FIRST {int(limit)} ROWS ONLY"
    df = connector.execute_query(query)
    output_path = output_dir / f"{table.lower()}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path, len(df.index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export up to N rows from each table in a schema to parquet files."
    )
    parser.add_argument("--schema", default="IEPLANE", help="DB2 schema to export from.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Maximum rows to export from each table.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/parquet",
        help="Destination directory for parquet files.",
    )
    parser.add_argument(
        "--env-path",
        default=".env",
        help="Path to environment file containing DB credentials.",
    )
    args = parser.parse_args()

    schema = str(args.schema).upper()
    limit = int(args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    connector = DatabaseConnector(env_path=args.env_path)
    try:
        tables = discover_tables(connector=connector, schema=schema)
        if not tables:
            print(f"No tables found in schema {schema}.")
            return

        print(f"Found {len(tables)} tables/nicknames in schema {schema}.")
        for table in tables:
            try:
                output_path, row_count = export_table(
                    connector=connector,
                    schema=schema,
                    table=table,
                    limit=limit,
                    output_dir=output_dir,
                )
                print(f"- {table}: exported {row_count} rows -> {output_path}")
            except Exception as error:  # noqa: BLE001
                print(f"- {table}: failed to export ({error})")
    finally:
        connector.dispose()


if __name__ == "__main__":
    main()
