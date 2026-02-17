from __future__ import annotations

import os
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb
import pandas as pd
from dotenv import load_dotenv

try:
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - exercised when polars isn't installed
    pl = None  # type: ignore[assignment]

DEFAULT_SCHEMA = "IEPLANE"
DEFAULT_DUCKDB_PATH = "IE_AIRPLANES.duckdb"


def sanitize_schema(schema: str | None) -> str:
    candidate = (schema or DEFAULT_SCHEMA).strip().upper()
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", candidate):
        raise ValueError("Schema name can only contain letters, numbers, and underscores.")
    return candidate


def resolve_duckdb_path(
    env_path: str = ".env",
    database_path: str | os.PathLike[str] | None = None,
) -> Path:
    if database_path is not None:
        candidate = Path(database_path)
    else:
        load_dotenv(dotenv_path=env_path, override=False)
        candidate = Path(os.getenv("DUCKDB_PATH", DEFAULT_DUCKDB_PATH))

    resolved = candidate.expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"DuckDB database not found at {resolved}. "
            "Set DUCKDB_PATH in .env or pass database_path explicitly."
        )
    return resolved


def _to_duckdb_named_params(
    query: str,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if not params:
        return query, {}
    # Convert SQLAlchemy-style named params (:name) to DuckDB named params ($name).
    converted_query = re.sub(r":([A-Za-z_][A-Za-z0-9_]*)", r"$\1", query)
    return converted_query, dict(params)


def _to_pandas_frame(df: Any) -> pd.DataFrame:
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return pd.DataFrame(df)


def _to_polars_frame(df: Any) -> "pl.DataFrame":
    if pl is None:
        raise RuntimeError("polars is not installed.")
    if isinstance(df, pl.DataFrame):
        return df.clone()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return pl.DataFrame(df)


def run_duckdb_query(
    query: str,
    params: Mapping[str, Any] | None = None,
    env_path: str = ".env",
    database_path: str | os.PathLike[str] | None = None,
    as_polars: bool = False,
) -> pd.DataFrame | "pl.DataFrame":
    db_path = resolve_duckdb_path(env_path=env_path, database_path=database_path)
    converted_query, converted_params = _to_duckdb_named_params(query=query, params=params)
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        result_df: pd.DataFrame
        if converted_params:
            result_df = connection.execute(converted_query, converted_params).fetchdf()
        else:
            result_df = connection.execute(converted_query).fetchdf()
        if as_polars and pl is not None:
            return pl.from_pandas(result_df)
        return result_df
    finally:
        connection.close()


def build_ticket_filters(
    alias: str,
    start_date: date | None = None,
    end_date: date | None = None,
    classes: Iterable[str] | None = None,
    route_codes: Iterable[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if start_date is not None:
        params["start_departure"] = datetime.combine(start_date, time.min)
        clauses.append(f"{alias}.departure >= :start_departure")

    if end_date is not None:
        exclusive_end = datetime.combine(end_date + timedelta(days=1), time.min)
        params["end_departure"] = exclusive_end
        clauses.append(f"{alias}.departure < :end_departure")

    class_values = [item for item in (classes or []) if item]
    if class_values:
        placeholders: list[str] = []
        for index, class_value in enumerate(class_values):
            key = f"class_{index}"
            placeholders.append(f":{key}")
            params[key] = class_value
        clauses.append(f"{alias}.class IN ({', '.join(placeholders)})")

    route_values = [item for item in (route_codes or []) if item]
    if route_values:
        placeholders = []
        for index, route_code in enumerate(route_values):
            key = f"route_{index}"
            placeholders.append(f":{key}")
            params[key] = route_code
        clauses.append(f"{alias}.route_code IN ({', '.join(placeholders)})")

    if not clauses:
        return "", params

    return f"WHERE {' AND '.join(clauses)}", params


def get_financial_filter_options(
    env_path: str = ".env",
    schema: str | None = None,
    database_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    sanitize_schema(schema)
    bounds_query = """
        SELECT
            MIN(CAST(departure AS DATE)) AS min_date,
            MAX(CAST(departure AS DATE)) AS max_date
        FROM TICKETS
    """
    class_query = """
        SELECT DISTINCT class AS ticket_class
        FROM TICKETS
        ORDER BY ticket_class
    """
    route_query = """
        SELECT DISTINCT route_code
        FROM ROUTES
        ORDER BY route_code
    """

    bounds_df = run_duckdb_query(
        bounds_query,
        env_path=env_path,
        database_path=database_path,
        as_polars=pl is not None,
    )
    classes_df = run_duckdb_query(
        class_query,
        env_path=env_path,
        database_path=database_path,
        as_polars=pl is not None,
    )
    routes_df = run_duckdb_query(
        route_query,
        env_path=env_path,
        database_path=database_path,
        as_polars=pl is not None,
    )

    bounds_df = _to_pandas_frame(bounds_df)
    classes_df = _to_pandas_frame(classes_df)
    routes_df = _to_pandas_frame(routes_df)

    bounds_df.columns = [column.lower() for column in bounds_df.columns]
    classes_df.columns = [column.lower() for column in classes_df.columns]
    routes_df.columns = [column.lower() for column in routes_df.columns]

    min_date = pd.to_datetime(bounds_df.loc[0, "min_date"]).date()
    max_date = pd.to_datetime(bounds_df.loc[0, "max_date"]).date()
    classes = [str(value) for value in classes_df["ticket_class"].dropna().tolist()]
    routes = [str(value) for value in routes_df["route_code"].dropna().tolist()]

    return {
        "min_date": min_date,
        "max_date": max_date,
        "classes": classes,
        "routes": routes,
    }


def extract_financial_base_data(
    start_date: date,
    end_date: date,
    classes: Iterable[str] | None = None,
    route_codes: Iterable[str] | None = None,
    env_path: str = ".env",
    schema: str | None = None,
    database_path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    sanitize_schema(schema)
    where_clause, params = build_ticket_filters(
        alias="t",
        start_date=start_date,
        end_date=end_date,
        classes=classes,
        route_codes=route_codes,
    )

    query = f"""
        WITH ticket_revenue AS (
            SELECT
                t.flight_id AS flight_id,
                t.route_code AS route_code,
                t.departure AS departure,
                t.class AS ticket_class,
                COUNT(*) AS tickets_sold,
                SUM(COALESCE(t.price, 0)) AS base_revenue,
                SUM(COALESCE(t.airport_tax, 0)) AS airport_tax_revenue,
                SUM(COALESCE(t.local_tax, 0)) AS local_tax_revenue,
                SUM(COALESCE(t.total_amount, 0)) AS total_revenue
            FROM TICKETS t
            {where_clause}
            GROUP BY
                t.flight_id,
                t.route_code,
                t.departure,
                t.class
        ),
        flight_dimension AS (
            SELECT
                f.flight_id AS flight_id,
                f.route_code AS route_code,
                MAX(f.airplane) AS airplane,
                MAX(r.origin) AS origin,
                MAX(r.destination) AS destination,
                MAX(COALESCE(r.distance, 0)) AS distance,
                MAX(COALESCE(r.flight_minutes, 0)) AS flight_minutes,
                MAX(COALESCE(a.fuel_gallons_hour, 0)) AS fuel_gallons_hour,
                MAX(
                    COALESCE(a.seats_business, 0)
                    + COALESCE(a.seats_premium, 0)
                    + COALESCE(a.seats_economy, 0)
                ) AS total_seats
            FROM FLIGHTS f
            INNER JOIN ROUTES r
                ON f.route_code = r.route_code
            INNER JOIN AIRPLANES a
                ON f.airplane = a.aircraft_registration
            GROUP BY
                f.flight_id,
                f.route_code
        )
        SELECT
            tr.flight_id,
            tr.route_code,
            tr.departure,
            DATE(tr.departure) AS flight_date,
            fd.airplane,
            fd.origin,
            fd.destination,
            fd.distance,
            fd.flight_minutes,
            fd.fuel_gallons_hour,
            fd.total_seats,
            tr.ticket_class,
            tr.tickets_sold,
            tr.base_revenue,
            tr.airport_tax_revenue,
            tr.local_tax_revenue,
            tr.total_revenue
        FROM ticket_revenue tr
        INNER JOIN flight_dimension fd
            ON tr.flight_id = fd.flight_id
            AND tr.route_code = fd.route_code
    """
    df = run_duckdb_query(
        query,
        params=params,
        env_path=env_path,
        database_path=database_path,
        as_polars=pl is not None,
    )

    return _to_pandas_frame(normalize_financial_base_data(df))


def normalize_financial_base_data(
    df: pd.DataFrame | "pl.DataFrame",
) -> pd.DataFrame | "pl.DataFrame":
    if pl is not None and isinstance(df, pl.DataFrame):
        normalized = df.rename({column: column.lower() for column in df.columns})

        numeric_columns = [
            "distance",
            "flight_minutes",
            "fuel_gallons_hour",
            "total_seats",
            "tickets_sold",
            "base_revenue",
            "airport_tax_revenue",
            "local_tax_revenue",
            "total_revenue",
        ]
        for column in numeric_columns:
            if column in normalized.columns:
                normalized = normalized.with_columns(
                    pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0).alias(column)
                )

        if "departure" in normalized.columns:
            normalized = normalized.with_columns(
                pl.col("departure").cast(pl.Datetime, strict=False).alias("departure")
            )
        if "flight_date" in normalized.columns:
            normalized = normalized.with_columns(
                pl.col("flight_date").cast(pl.Date, strict=False).alias("flight_date")
            )
        return normalized

    normalized = df.copy()
    normalized.columns = [column.lower() for column in normalized.columns]

    numeric_columns = [
        "distance",
        "flight_minutes",
        "fuel_gallons_hour",
        "total_seats",
        "tickets_sold",
        "base_revenue",
        "airport_tax_revenue",
        "local_tax_revenue",
        "total_revenue",
    ]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)

    if "departure" in normalized.columns:
        normalized["departure"] = pd.to_datetime(normalized["departure"], errors="coerce")
    if "flight_date" in normalized.columns:
        normalized["flight_date"] = pd.to_datetime(normalized["flight_date"], errors="coerce").dt.date

    return normalized

def _empty_financial_views() -> dict[str, Any]:
    return {
        "kpis": {
            "total_revenue": 0.0,
            "ancillary_revenue": 0.0,
            "available_seat_miles": 0.0,
            "rasm": 0.0,
            "estimated_fuel_cost": 0.0,
            "estimated_total_cost": 0.0,
            "estimated_profit": 0.0,
            "ancillary_share": 0.0,
        },
        "daily_revenue": pd.DataFrame(),
        "route_profitability": pd.DataFrame(),
        "ancillary_by_class": pd.DataFrame(),
    }


def _compute_financial_views_pandas(
    base_df: pd.DataFrame,
    fuel_price_per_gallon: float,
) -> dict[str, Any]:
    if base_df.empty:
        return _empty_financial_views()

    fuel_price = float(fuel_price_per_gallon)
    working = _to_pandas_frame(normalize_financial_base_data(base_df))

    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("financial_base", working)

        flight_level = connection.execute(
            """
            SELECT
                flight_id,
                route_code,
                flight_date,
                origin,
                destination,
                MAX(total_seats) AS total_seats,
                MAX(distance) AS distance,
                MAX(flight_minutes) AS flight_minutes,
                MAX(fuel_gallons_hour) AS fuel_gallons_hour,
                SUM(tickets_sold) AS tickets_sold,
                SUM(base_revenue) AS base_revenue,
                SUM(airport_tax_revenue) AS airport_tax_revenue,
                SUM(local_tax_revenue) AS local_tax_revenue,
                SUM(total_revenue) AS total_revenue
            FROM financial_base
            GROUP BY
                flight_id,
                route_code,
                flight_date,
                origin,
                destination
            """
        ).fetchdf()
        connection.register("flight_level", flight_level)

        kpis_df = connection.execute(
            f"""
            SELECT
                SUM(total_revenue) AS total_revenue,
                SUM(airport_tax_revenue + local_tax_revenue) AS ancillary_revenue,
                SUM(total_seats * distance) AS available_seat_miles,
                CASE
                    WHEN SUM(total_seats * distance) = 0 THEN 0
                    ELSE SUM(total_revenue) / SUM(total_seats * distance)
                END AS rasm,
                SUM((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price}) AS estimated_fuel_cost,
                SUM((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price}) AS estimated_total_cost,
                SUM(
                    total_revenue - ((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price})
                ) AS estimated_profit
            FROM flight_level
            """
        ).fetchdf()
        kpis = kpis_df.iloc[0].to_dict()
        total_revenue = float(kpis["total_revenue"] or 0.0)
        ancillary_revenue = float(kpis["ancillary_revenue"] or 0.0)
        kpis["ancillary_share"] = ancillary_revenue / total_revenue if total_revenue else 0.0

        daily_revenue = connection.execute(
            """
            SELECT
                flight_date,
                SUM(total_revenue) AS total_revenue,
                SUM(base_revenue) AS base_revenue,
                SUM(airport_tax_revenue + local_tax_revenue) AS ancillary_revenue
            FROM flight_level
            GROUP BY flight_date
            ORDER BY flight_date
            """
        ).fetchdf()

        route_profitability = connection.execute(
            f"""
            SELECT
                route_code,
                origin,
                destination,
                COUNT(*) AS flights,
                SUM(tickets_sold) AS tickets_sold,
                SUM(total_revenue) AS route_revenue,
                SUM(total_seats * distance) AS route_available_seat_miles,
                CASE
                    WHEN SUM(total_seats * distance) = 0 THEN 0
                    ELSE SUM(total_revenue) / SUM(total_seats * distance)
                END AS route_rasm,
                SUM((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price}) AS estimated_fuel_cost,
                SUM((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price}) AS estimated_total_cost,
                SUM(
                    total_revenue - ((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price})
                ) AS estimated_profit,
                CASE
                    WHEN SUM(total_revenue) = 0 THEN 0
                    ELSE SUM(
                        total_revenue - ((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price})
                    ) / SUM(total_revenue)
                END AS profit_margin,
                CASE
                    WHEN SUM(total_seats) = 0 THEN 0
                    ELSE SUM(tickets_sold) * 1.0 / SUM(total_seats)
                END AS load_factor
            FROM flight_level
            GROUP BY
                route_code,
                origin,
                destination
            ORDER BY estimated_profit DESC
            """
        ).fetchdf()

        ancillary_by_class = connection.execute(
            """
            SELECT
                ticket_class,
                SUM(base_revenue) AS base_revenue,
                SUM(airport_tax_revenue) AS airport_tax_revenue,
                SUM(local_tax_revenue) AS local_tax_revenue,
                SUM(total_revenue) AS total_revenue,
                CASE
                    WHEN SUM(total_revenue) = 0 THEN 0
                    ELSE SUM(airport_tax_revenue + local_tax_revenue) / SUM(total_revenue)
                END AS ancillary_share
            FROM financial_base
            GROUP BY ticket_class
            ORDER BY ticket_class
            """
        ).fetchdf()
    finally:
        connection.close()

    route_profitability["segment"] = route_profitability["estimated_profit"].apply(
        lambda value: "Cash Cow" if value >= 0 else "Money Pit"
    )
    return {
        "kpis": kpis,
        "daily_revenue": daily_revenue,
        "route_profitability": route_profitability,
        "ancillary_by_class": ancillary_by_class,
    }


def _compute_financial_views_polars(
    base_df: pd.DataFrame | "pl.DataFrame",
    fuel_price_per_gallon: float,
) -> dict[str, Any]:
    if pl is None:
        return _compute_financial_views_pandas(_to_pandas_frame(base_df), fuel_price_per_gallon)

    working = normalize_financial_base_data(_to_polars_frame(base_df))
    if working.is_empty():
        return _empty_financial_views()

    fuel_price = float(fuel_price_per_gallon)

    flight_level = (
        working.group_by(["flight_id", "route_code", "flight_date", "origin", "destination"])
        .agg(
            [
                pl.col("total_seats").max().alias("total_seats"),
                pl.col("distance").max().alias("distance"),
                pl.col("flight_minutes").max().alias("flight_minutes"),
                pl.col("fuel_gallons_hour").max().alias("fuel_gallons_hour"),
                pl.col("tickets_sold").sum().alias("tickets_sold"),
                pl.col("base_revenue").sum().alias("base_revenue"),
                pl.col("airport_tax_revenue").sum().alias("airport_tax_revenue"),
                pl.col("local_tax_revenue").sum().alias("local_tax_revenue"),
                pl.col("total_revenue").sum().alias("total_revenue"),
            ]
        )
        .with_columns(
            [
                (pl.col("total_seats") * pl.col("distance")).alias("available_seat_miles"),
                (
                    (pl.col("fuel_gallons_hour") * pl.col("flight_minutes") / 60.0) * fuel_price
                ).alias("estimated_fuel_cost"),
            ]
        )
        .with_columns(
            [
                pl.col("estimated_fuel_cost").alias("estimated_total_cost"),
                (pl.col("total_revenue") - pl.col("estimated_fuel_cost")).alias("estimated_profit"),
            ]
        )
    )

    total_revenue = float(flight_level["total_revenue"].sum())
    ancillary_revenue = float(
        (flight_level["airport_tax_revenue"] + flight_level["local_tax_revenue"]).sum()
    )
    available_seat_miles = float(flight_level["available_seat_miles"].sum())
    estimated_fuel_cost = float(flight_level["estimated_fuel_cost"].sum())
    estimated_total_cost = float(flight_level["estimated_total_cost"].sum())
    estimated_profit = float(flight_level["estimated_profit"].sum())
    kpis = {
        "total_revenue": total_revenue,
        "ancillary_revenue": ancillary_revenue,
        "available_seat_miles": available_seat_miles,
        "rasm": total_revenue / available_seat_miles if available_seat_miles else 0.0,
        "estimated_fuel_cost": estimated_fuel_cost,
        "estimated_total_cost": estimated_total_cost,
        "estimated_profit": estimated_profit,
        "ancillary_share": ancillary_revenue / total_revenue if total_revenue else 0.0,
    }

    daily_revenue = (
        flight_level.group_by("flight_date")
        .agg(
            [
                pl.col("total_revenue").sum().alias("total_revenue"),
                pl.col("base_revenue").sum().alias("base_revenue"),
                (pl.col("airport_tax_revenue") + pl.col("local_tax_revenue"))
                .sum()
                .alias("ancillary_revenue"),
            ]
        )
        .sort("flight_date")
    )

    route_profitability = (
        flight_level.group_by(["route_code", "origin", "destination"])
        .agg(
            [
                pl.len().alias("flights"),
                pl.col("tickets_sold").sum().alias("tickets_sold"),
                pl.col("total_revenue").sum().alias("route_revenue"),
                pl.col("available_seat_miles").sum().alias("route_available_seat_miles"),
                pl.col("estimated_fuel_cost").sum().alias("estimated_fuel_cost"),
                pl.col("total_seats").sum().alias("total_seats_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("route_available_seat_miles") == 0)
                .then(0.0)
                .otherwise(pl.col("route_revenue") / pl.col("route_available_seat_miles"))
                .alias("route_rasm"),
                pl.col("estimated_fuel_cost").alias("estimated_total_cost"),
                (pl.col("route_revenue") - pl.col("estimated_fuel_cost")).alias("estimated_profit"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("route_revenue") == 0)
                .then(0.0)
                .otherwise(pl.col("estimated_profit") / pl.col("route_revenue"))
                .alias("profit_margin"),
                pl.when(pl.col("total_seats_sum") == 0)
                .then(0.0)
                .otherwise(pl.col("tickets_sold") / pl.col("total_seats_sum"))
                .alias("load_factor"),
                pl.when(pl.col("estimated_profit") >= 0)
                .then(pl.lit("Cash Cow"))
                .otherwise(pl.lit("Money Pit"))
                .alias("segment"),
            ]
        )
        .drop("total_seats_sum")
        .sort("estimated_profit", descending=True)
    )

    ancillary_by_class = (
        working.group_by("ticket_class")
        .agg(
            [
                pl.col("base_revenue").sum().alias("base_revenue"),
                pl.col("airport_tax_revenue").sum().alias("airport_tax_revenue"),
                pl.col("local_tax_revenue").sum().alias("local_tax_revenue"),
                pl.col("total_revenue").sum().alias("total_revenue"),
            ]
        )
        .with_columns(
            pl.when(pl.col("total_revenue") == 0)
            .then(0.0)
            .otherwise((pl.col("airport_tax_revenue") + pl.col("local_tax_revenue")) / pl.col("total_revenue"))
            .alias("ancillary_share")
        )
        .sort("ticket_class")
    )

    return {
        "kpis": kpis,
        "daily_revenue": daily_revenue.to_pandas(),
        "route_profitability": route_profitability.to_pandas(),
        "ancillary_by_class": ancillary_by_class.to_pandas(),
    }


def compute_financial_views(
    base_df: pd.DataFrame | "pl.DataFrame",
    fuel_price_per_gallon: float,
) -> dict[str, Any]:
    if pl is not None:
        return _compute_financial_views_polars(base_df=base_df, fuel_price_per_gallon=fuel_price_per_gallon)
    return _compute_financial_views_pandas(
        base_df=_to_pandas_frame(base_df), fuel_price_per_gallon=fuel_price_per_gallon
    )
