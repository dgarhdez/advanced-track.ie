from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - exercised when polars isn't installed
    pl = None  # type: ignore[assignment]

from dashboard.financial_pipeline import run_duckdb_query, sanitize_schema


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


def build_commercial_filters(
    alias: str,
    start_date: date | None = None,
    end_date: date | None = None,
    route_codes: Iterable[str] | None = None,
    classes: Iterable[str] | None = None,
    genders: Iterable[str] | None = None,
    countries: Iterable[str] | None = None,
    continents: Iterable[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if start_date is not None:
        params["start_departure"] = datetime.combine(start_date, time.min)
        clauses.append(f"{alias}.departure >= :start_departure")

    if end_date is not None:
        params["end_departure"] = datetime.combine(end_date + timedelta(days=1), time.min)
        clauses.append(f"{alias}.departure < :end_departure")

    route_values = [item for item in (route_codes or []) if item]
    if route_values:
        placeholders: list[str] = []
        for index, route_code in enumerate(route_values):
            key = f"route_{index}"
            params[key] = route_code
            placeholders.append(f":{key}")
        clauses.append(f"{alias}.route_code IN ({', '.join(placeholders)})")

    class_values = [item for item in (classes or []) if item]
    if class_values:
        placeholders = []
        for index, class_value in enumerate(class_values):
            key = f"class_{index}"
            params[key] = class_value
            placeholders.append(f":{key}")
        clauses.append(f"{alias}.class IN ({', '.join(placeholders)})")

    gender_values = [item for item in (genders or []) if item]
    if gender_values:
        placeholders = []
        for index, gender in enumerate(gender_values):
            key = f"gender_{index}"
            params[key] = gender
            placeholders.append(f":{key}")
        clauses.append(f"p.gender IN ({', '.join(placeholders)})")

    country_values = [item for item in (countries or []) if item]
    if country_values:
        placeholders = []
        for index, country in enumerate(country_values):
            key = f"country_{index}"
            params[key] = country
            placeholders.append(f":{key}")
        clauses.append(f"COALESCE(c.name, p.country) IN ({', '.join(placeholders)})")

    continent_values = [item for item in (continents or []) if item]
    if continent_values:
        placeholders = []
        for index, continent in enumerate(continent_values):
            key = f"continent_{index}"
            params[key] = continent
            placeholders.append(f":{key}")
        clauses.append(f"COALESCE(c.continent, 'Unknown') IN ({', '.join(placeholders)})")

    if not clauses:
        return "", params
    return f"WHERE {' AND '.join(clauses)}", params


def get_commercial_filter_options(
    env_path: str = ".env",
    schema: str | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    sanitize_schema(schema)
    bounds_query = """
        SELECT
            MIN(CAST(departure AS DATE)) AS min_date,
            MAX(CAST(departure AS DATE)) AS max_date
        FROM TICKETS
    """
    route_query = """
        SELECT DISTINCT route_code
        FROM ROUTES
        ORDER BY route_code
    """
    class_query = """
        SELECT DISTINCT class AS ticket_class
        FROM TICKETS
        ORDER BY ticket_class
    """
    gender_query = """
        SELECT DISTINCT gender
        FROM PASSENGERS
        WHERE gender IS NOT NULL
        ORDER BY gender
    """
    country_query = """
        SELECT DISTINCT COALESCE(c.name, p.country) AS passenger_country
        FROM PASSENGERS p
        LEFT JOIN COUNTRIES c
            ON UPPER(TRIM(p.country)) = UPPER(TRIM(c.name))
        WHERE COALESCE(c.name, p.country) IS NOT NULL
        ORDER BY passenger_country
    """
    continent_query = """
        SELECT DISTINCT COALESCE(c.continent, 'Unknown') AS passenger_continent
        FROM PASSENGERS p
        LEFT JOIN COUNTRIES c
            ON UPPER(TRIM(p.country)) = UPPER(TRIM(c.name))
        WHERE COALESCE(c.continent, 'Unknown') IS NOT NULL
        ORDER BY passenger_continent
    """

    bounds_df = _to_pandas_frame(
        run_duckdb_query(
            bounds_query,
            env_path=env_path,
            database_path=database_path,
            as_polars=pl is not None,
        )
    )
    route_df = _to_pandas_frame(
        run_duckdb_query(
            route_query,
            env_path=env_path,
            database_path=database_path,
            as_polars=pl is not None,
        )
    )
    class_df = _to_pandas_frame(
        run_duckdb_query(
            class_query,
            env_path=env_path,
            database_path=database_path,
            as_polars=pl is not None,
        )
    )
    gender_df = _to_pandas_frame(
        run_duckdb_query(
            gender_query,
            env_path=env_path,
            database_path=database_path,
            as_polars=pl is not None,
        )
    )
    country_df = _to_pandas_frame(
        run_duckdb_query(
            country_query,
            env_path=env_path,
            database_path=database_path,
            as_polars=pl is not None,
        )
    )
    continent_df = _to_pandas_frame(
        run_duckdb_query(
            continent_query,
            env_path=env_path,
            database_path=database_path,
            as_polars=pl is not None,
        )
    )

    bounds_df.columns = [column.lower() for column in bounds_df.columns]
    route_df.columns = [column.lower() for column in route_df.columns]
    class_df.columns = [column.lower() for column in class_df.columns]
    gender_df.columns = [column.lower() for column in gender_df.columns]
    country_df.columns = [column.lower() for column in country_df.columns]
    continent_df.columns = [column.lower() for column in continent_df.columns]

    return {
        "min_date": pd.to_datetime(bounds_df.loc[0, "min_date"]).date(),
        "max_date": pd.to_datetime(bounds_df.loc[0, "max_date"]).date(),
        "routes": [str(value) for value in route_df["route_code"].dropna().tolist()],
        "classes": [str(value) for value in class_df["ticket_class"].dropna().tolist()],
        "genders": [str(value) for value in gender_df["gender"].dropna().tolist()],
        "countries": [str(value) for value in country_df["passenger_country"].dropna().tolist()],
        "continents": [str(value) for value in continent_df["passenger_continent"].dropna().tolist()],
    }


def extract_commercial_base_data(
    start_date: date,
    end_date: date,
    route_codes: Iterable[str] | None = None,
    classes: Iterable[str] | None = None,
    genders: Iterable[str] | None = None,
    countries: Iterable[str] | None = None,
    continents: Iterable[str] | None = None,
    env_path: str = ".env",
    schema: str | None = None,
    database_path: str | Path | None = None,
) -> pd.DataFrame:
    sanitize_schema(schema)
    where_clause, params = build_commercial_filters(
        alias="t",
        start_date=start_date,
        end_date=end_date,
        route_codes=route_codes,
        classes=classes,
        genders=genders,
        countries=countries,
        continents=continents,
    )
    query = f"""
        WITH flight_dimension AS (
            SELECT
                f.flight_id,
                f.route_code,
                MAX(f.airplane) AS airplane
            FROM FLIGHTS f
            GROUP BY
                f.flight_id,
                f.route_code
        )
        SELECT
            t.ticket_id,
            t.passenger_id,
            t.flight_id,
            t.route_code,
            t.departure,
            t.class AS ticket_class,
            p.gender,
            p.birth_date,
            COALESCE(c.name, p.country) AS passenger_country,
            COALESCE(c.continent, 'Unknown') AS passenger_continent,
            c.capital AS passenger_country_capital,
            r.origin,
            r.destination,
            COALESCE(r.distance, 0) AS distance,
            COALESCE(a.seats_business, 0) + COALESCE(a.seats_premium, 0) + COALESCE(a.seats_economy, 0)
                AS total_seats,
            ao.latitude AS origin_latitude,
            ao.longitude AS origin_longitude,
            ad.latitude AS destination_latitude,
            ad.longitude AS destination_longitude
        FROM TICKETS t
        INNER JOIN PASSENGERS p
            ON t.passenger_id = p.id
        LEFT JOIN COUNTRIES c
            ON UPPER(TRIM(p.country)) = UPPER(TRIM(c.name))
        INNER JOIN flight_dimension f
            ON t.flight_id = f.flight_id
            AND t.route_code = f.route_code
        INNER JOIN ROUTES r
            ON t.route_code = r.route_code
        INNER JOIN AIRPLANES a
            ON f.airplane = a.aircraft_registration
        LEFT JOIN AIRPORTS ao
            ON r.origin = ao.iata_code
        LEFT JOIN AIRPORTS ad
            ON r.destination = ad.iata_code
        {where_clause}
    """
    df = run_duckdb_query(
        query,
        params=params,
        env_path=env_path,
        database_path=database_path,
        as_polars=pl is not None,
    )
    return _to_pandas_frame(normalize_commercial_base_data(df))


def normalize_commercial_base_data(
    df: pd.DataFrame | "pl.DataFrame",
) -> pd.DataFrame | "pl.DataFrame":
    if pl is not None and isinstance(df, pl.DataFrame):
        normalized = df.rename({column: column.lower() for column in df.columns})
        numeric_columns = [
            "distance",
            "total_seats",
            "origin_latitude",
            "origin_longitude",
            "destination_latitude",
            "destination_longitude",
        ]
        for column in numeric_columns:
            if column in normalized.columns:
                normalized = normalized.with_columns(
                    pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0).alias(column)
                )
        for column in ("departure",):
            if column in normalized.columns:
                normalized = normalized.with_columns(
                    pl.col(column).cast(pl.Datetime, strict=False).alias(column)
                )
        for column in ("birth_date",):
            if column in normalized.columns:
                normalized = normalized.with_columns(
                    pl.col(column).cast(pl.Date, strict=False).alias(column)
                )
        return normalized

    normalized_pd = df.copy()
    normalized_pd.columns = [column.lower() for column in normalized_pd.columns]
    for column in [
        "distance",
        "total_seats",
        "origin_latitude",
        "origin_longitude",
        "destination_latitude",
        "destination_longitude",
    ]:
        if column in normalized_pd.columns:
            normalized_pd[column] = pd.to_numeric(normalized_pd[column], errors="coerce").fillna(0.0)
    if "departure" in normalized_pd.columns:
        normalized_pd["departure"] = pd.to_datetime(normalized_pd["departure"], errors="coerce")
    if "birth_date" in normalized_pd.columns:
        normalized_pd["birth_date"] = pd.to_datetime(normalized_pd["birth_date"], errors="coerce")
    return normalized_pd


def _age_bucket(age: float) -> str:
    if age < 18:
        return "<18"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


def _empty_commercial_views() -> dict[str, Any]:
    return {
        "kpis": {
            "total_passengers": 0,
            "avg_load_factor": 0.0,
            "active_routes": 0,
            "active_countries": 0,
            "busiest_route": "N/A",
        },
        "load_factor_by_route": pd.DataFrame(),
        "route_heatmap": pd.DataFrame(),
        "passenger_demographics": pd.DataFrame(),
        "gender_distribution": pd.DataFrame(),
        "country_distribution": pd.DataFrame(),
        "continent_distribution": pd.DataFrame(),
    }


def _compute_commercial_views_pandas(base_df: pd.DataFrame) -> dict[str, Any]:
    working = _to_pandas_frame(normalize_commercial_base_data(base_df))
    if working.empty:
        return _empty_commercial_views()

    flight_level = (
        working.groupby(
            ["flight_id", "route_code", "departure", "origin", "destination", "distance", "total_seats"],
            as_index=False,
        )
        .agg(tickets_sold=("ticket_id", "count"))
    )
    flight_level["load_factor"] = flight_level.apply(
        lambda row: float(row["tickets_sold"]) / float(row["total_seats"]) if float(row["total_seats"]) > 0 else 0.0,
        axis=1,
    )

    route_summary = (
        flight_level.groupby(["route_code", "origin", "destination"], as_index=False)
        .agg(
            flights=("flight_id", "count"),
            tickets_sold=("tickets_sold", "sum"),
            avg_load_factor=("load_factor", "mean"),
            avg_distance=("distance", "mean"),
        )
        .sort_values("tickets_sold", ascending=False)
    )

    heatmap = (
        working.groupby(
            [
                "route_code",
                "origin",
                "destination",
                "origin_latitude",
                "origin_longitude",
                "destination_latitude",
                "destination_longitude",
            ],
            as_index=False,
        )
        .agg(passengers=("ticket_id", "count"))
        .sort_values("passengers", ascending=False)
    )

    demo = working.copy()
    demo["age"] = (
        (pd.to_datetime(demo["departure"]).dt.date - pd.to_datetime(demo["birth_date"]).dt.date).dt.days / 365.25
    )
    demo["age"] = pd.to_numeric(demo["age"], errors="coerce").fillna(0.0).clip(lower=0.0)
    demo["age_bucket"] = demo["age"].apply(_age_bucket)

    demographics = (
        demo.groupby(["age_bucket", "gender"], as_index=False)
        .agg(passengers=("ticket_id", "count"))
        .sort_values(["age_bucket", "gender"])
    )
    gender_distribution = (
        demo.groupby(["gender"], as_index=False)
        .agg(passengers=("ticket_id", "count"))
        .sort_values("passengers", ascending=False)
    )
    country_distribution = (
        demo.groupby(["passenger_country"], as_index=False)
        .agg(passengers=("ticket_id", "count"))
        .sort_values("passengers", ascending=False)
    )
    continent_distribution = (
        demo.groupby(["passenger_continent"], as_index=False)
        .agg(passengers=("ticket_id", "count"))
        .sort_values("passengers", ascending=False)
    )

    busiest_route = str(route_summary.iloc[0]["route_code"]) if not route_summary.empty else "N/A"
    return {
        "kpis": {
            "total_passengers": int(len(working)),
            "avg_load_factor": float(flight_level["load_factor"].mean()) if not flight_level.empty else 0.0,
            "active_routes": int(route_summary["route_code"].nunique()),
            "active_countries": int(country_distribution["passenger_country"].nunique()),
            "busiest_route": busiest_route,
        },
        "load_factor_by_route": route_summary,
        "route_heatmap": heatmap,
        "passenger_demographics": demographics,
        "gender_distribution": gender_distribution,
        "country_distribution": country_distribution,
        "continent_distribution": continent_distribution,
    }


def _compute_commercial_views_polars(base_df: pd.DataFrame | "pl.DataFrame") -> dict[str, Any]:
    if pl is None:
        return _compute_commercial_views_pandas(_to_pandas_frame(base_df))

    working = normalize_commercial_base_data(_to_polars_frame(base_df))
    if working.is_empty():
        return _empty_commercial_views()

    flight_level = (
        working.group_by(["flight_id", "route_code", "departure", "origin", "destination", "distance", "total_seats"])
        .agg(pl.len().alias("tickets_sold"))
        .with_columns(
            pl.when(pl.col("total_seats") <= 0)
            .then(0.0)
            .otherwise(pl.col("tickets_sold").cast(pl.Float64) / pl.col("total_seats").cast(pl.Float64))
            .alias("load_factor")
        )
    )

    route_summary = (
        flight_level.group_by(["route_code", "origin", "destination"])
        .agg(
            [
                pl.len().alias("flights"),
                pl.col("tickets_sold").sum().alias("tickets_sold"),
                pl.col("load_factor").mean().alias("avg_load_factor"),
                pl.col("distance").mean().alias("avg_distance"),
            ]
        )
        .sort("tickets_sold", descending=True)
    )

    heatmap = (
        working.group_by(
            [
                "route_code",
                "origin",
                "destination",
                "origin_latitude",
                "origin_longitude",
                "destination_latitude",
                "destination_longitude",
            ]
        )
        .agg(pl.len().alias("passengers"))
        .sort("passengers", descending=True)
    )

    demo = (
        working.with_columns(
            (
                (pl.col("departure").cast(pl.Date) - pl.col("birth_date").cast(pl.Date)).dt.total_days()
                / 365.25
            )
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .clip(lower_bound=0.0)
            .alias("age")
        )
        .with_columns(
            pl.when(pl.col("age") < 18)
            .then(pl.lit("<18"))
            .when(pl.col("age") < 25)
            .then(pl.lit("18-24"))
            .when(pl.col("age") < 35)
            .then(pl.lit("25-34"))
            .when(pl.col("age") < 45)
            .then(pl.lit("35-44"))
            .when(pl.col("age") < 55)
            .then(pl.lit("45-54"))
            .when(pl.col("age") < 65)
            .then(pl.lit("55-64"))
            .otherwise(pl.lit("65+"))
            .alias("age_bucket")
        )
    )

    demographics = (
        demo.group_by(["age_bucket", "gender"])
        .agg(pl.len().alias("passengers"))
        .sort(["age_bucket", "gender"])
    )
    gender_distribution = demo.group_by("gender").agg(pl.len().alias("passengers")).sort("passengers", descending=True)
    country_distribution = (
        demo.group_by("passenger_country").agg(pl.len().alias("passengers")).sort("passengers", descending=True)
    )
    continent_distribution = (
        demo.group_by("passenger_continent").agg(pl.len().alias("passengers")).sort("passengers", descending=True)
    )

    busiest_route = (
        str(route_summary.select("route_code").to_series()[0]) if route_summary.height > 0 else "N/A"
    )
    return {
        "kpis": {
            "total_passengers": int(working.height),
            "avg_load_factor": float(flight_level["load_factor"].mean()) if flight_level.height > 0 else 0.0,
            "active_routes": int(route_summary["route_code"].n_unique()),
            "active_countries": int(country_distribution["passenger_country"].n_unique()),
            "busiest_route": busiest_route,
        },
        "load_factor_by_route": route_summary.to_pandas(),
        "route_heatmap": heatmap.to_pandas(),
        "passenger_demographics": demographics.to_pandas(),
        "gender_distribution": gender_distribution.to_pandas(),
        "country_distribution": country_distribution.to_pandas(),
        "continent_distribution": continent_distribution.to_pandas(),
    }


def compute_commercial_views(
    base_df: pd.DataFrame | "pl.DataFrame",
) -> dict[str, Any]:
    if pl is not None:
        return _compute_commercial_views_polars(base_df=base_df)
    return _compute_commercial_views_pandas(_to_pandas_frame(base_df))
