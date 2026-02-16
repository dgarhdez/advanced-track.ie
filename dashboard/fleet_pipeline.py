from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Iterable

import duckdb
import pandas as pd

from database_connector import DatabaseConnector
from dashboard.financial_pipeline import DEFAULT_SCHEMA, sanitize_schema


def build_fleet_filters(
    start_date: date | None = None,
    end_date: date | None = None,
    models: Iterable[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if start_date is not None:
        params["start_departure"] = datetime.combine(start_date, time.min)
        clauses.append("f.departure >= :start_departure")

    if end_date is not None:
        params["end_departure"] = datetime.combine(end_date + timedelta(days=1), time.min)
        clauses.append("f.departure < :end_departure")

    model_values = [item for item in (models or []) if item]
    if model_values:
        placeholders: list[str] = []
        for index, model in enumerate(model_values):
            key = f"model_{index}"
            params[key] = model
            placeholders.append(f":{key}")
        clauses.append(f"a.model IN ({', '.join(placeholders)})")

    if not clauses:
        return "", params
    return f"WHERE {' AND '.join(clauses)}", params


def get_fleet_filter_options(
    env_path: str = ".env",
    schema: str | None = None,
) -> dict[str, Any]:
    schema_name = sanitize_schema(schema)
    connector = DatabaseConnector(env_path=env_path)
    try:
        bounds_query = f"""
            SELECT
                MIN(DATE(departure)) AS min_date,
                MAX(DATE(departure)) AS max_date
            FROM {schema_name}.FLIGHTS
        """
        model_query = f"""
            SELECT DISTINCT model
            FROM {schema_name}.AIRPLANES
            ORDER BY model
        """
        bounds_df = connector.execute_query(bounds_query)
        model_df = connector.execute_query(model_query)
    finally:
        connector.dispose()

    bounds_df.columns = [column.lower() for column in bounds_df.columns]
    model_df.columns = [column.lower() for column in model_df.columns]

    min_date = pd.to_datetime(bounds_df.loc[0, "min_date"]).date()
    max_date = pd.to_datetime(bounds_df.loc[0, "max_date"]).date()
    models = [str(value) for value in model_df["model"].dropna().tolist()]

    return {"min_date": min_date, "max_date": max_date, "models": models}


def extract_fleet_base_data(
    start_date: date,
    end_date: date,
    models: Iterable[str] | None = None,
    env_path: str = ".env",
    schema: str | None = None,
) -> pd.DataFrame:
    schema_name = sanitize_schema(schema)
    where_clause, params = build_fleet_filters(
        start_date=start_date, end_date=end_date, models=models
    )

    query = f"""
        SELECT
            f.flight_id,
            f.route_code,
            f.departure,
            f.arrival,
            f.airplane AS aircraft_registration,
            a.model,
            COALESCE(a.crew_members, 0) AS crew_members,
            COALESCE(a.fuel_gallons_hour, 0) AS fuel_gallons_hour,
            a.maintenance_last_acheck,
            a.maintenance_last_bcheck,
            COALESCE(a.maintenance_takeoffs, 0) AS maintenance_takeoffs,
            COALESCE(a.maintenance_flight_hours, 0) AS maintenance_flight_hours,
            COALESCE(a.total_flight_distance, 0) AS total_flight_distance,
            COALESCE(r.distance, 0) AS route_distance,
            COALESCE(r.flight_minutes, 0) AS route_flight_minutes
        FROM {schema_name}.FLIGHTS f
        INNER JOIN {schema_name}.AIRPLANES a
            ON f.airplane = a.aircraft_registration
        LEFT JOIN {schema_name}.ROUTES r
            ON f.route_code = r.route_code
        {where_clause}
    """

    connector = DatabaseConnector(env_path=env_path)
    try:
        df = connector.execute_query(query, params=params)
    finally:
        connector.dispose()
    return normalize_fleet_base_data(df)


def normalize_fleet_base_data(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [column.lower() for column in normalized.columns]

    numeric_columns = [
        "crew_members",
        "fuel_gallons_hour",
        "maintenance_takeoffs",
        "maintenance_flight_hours",
        "total_flight_distance",
        "route_distance",
        "route_flight_minutes",
        "observed_flight_hours",
    ]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)

    for column in [
        "departure",
        "arrival",
        "maintenance_last_acheck",
        "maintenance_last_bcheck",
    ]:
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], errors="coerce")

    if "observed_flight_hours" not in normalized.columns:
        normalized["observed_flight_hours"] = 0.0

    if "departure" in normalized.columns and "arrival" in normalized.columns:
        elapsed_hours = (
            (normalized["arrival"] - normalized["departure"]).dt.total_seconds() / 3600.0
        )
        elapsed_hours = elapsed_hours.where(elapsed_hours > 0)
        normalized["observed_flight_hours"] = elapsed_hours.fillna(
            normalized.get("route_flight_minutes", 0) / 60.0
        )
    else:
        normalized["observed_flight_hours"] = normalized.get("route_flight_minutes", 0) / 60.0

    normalized["observed_flight_hours"] = pd.to_numeric(
        normalized["observed_flight_hours"], errors="coerce"
    ).fillna(0.0)
    return normalized


def _status_from_score(score: float, warning_ratio: float) -> str:
    if score >= 1.0:
        return "Overdue"
    if score >= warning_ratio:
        return "Warning"
    return "Healthy"


def _primary_trigger(row: pd.Series) -> str:
    ratios = {
        "A-check age": float(row.get("a_check_ratio", 0.0)),
        "B-check age": float(row.get("b_check_ratio", 0.0)),
        "Takeoffs": float(row.get("takeoff_ratio", 0.0)),
        "Flight hours": float(row.get("flight_hours_ratio", 0.0)),
    }
    return max(ratios, key=ratios.get)


def compute_fleet_views(
    base_df: pd.DataFrame,
    maintenance_takeoffs_threshold: int,
    maintenance_flight_hours_threshold: int,
    a_check_days_threshold: int,
    b_check_days_threshold: int,
    warning_ratio: float = 0.85,
    reference_date: date | None = None,
) -> dict[str, Any]:
    working = normalize_fleet_base_data(base_df)
    if working.empty:
        return {
            "kpis": {
                "active_aircraft": 0,
                "total_flights": 0,
                "total_flight_hours": 0.0,
                "avg_utilization_hours_per_aircraft": 0.0,
                "total_route_distance": 0.0,
                "avg_fuel_gallons_hour": 0.0,
                "at_risk_aircraft": 0,
            },
            "utilization_by_aircraft": pd.DataFrame(),
            "daily_operations": pd.DataFrame(),
            "fuel_efficiency_by_model": pd.DataFrame(),
            "maintenance_alerts": pd.DataFrame(),
        }

    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("fleet_base", working)
        flight_level = connection.execute(
            """
            SELECT
                flight_id,
                route_code,
                departure,
                DATE(departure) AS flight_date,
                aircraft_registration,
                model,
                MAX(crew_members) AS crew_members,
                MAX(fuel_gallons_hour) AS fuel_gallons_hour,
                MAX(maintenance_last_acheck) AS maintenance_last_acheck,
                MAX(maintenance_last_bcheck) AS maintenance_last_bcheck,
                MAX(maintenance_takeoffs) AS maintenance_takeoffs,
                MAX(maintenance_flight_hours) AS maintenance_flight_hours,
                MAX(total_flight_distance) AS total_flight_distance,
                MAX(route_distance) AS route_distance,
                MAX(observed_flight_hours) AS observed_flight_hours
            FROM fleet_base
            GROUP BY
                flight_id,
                route_code,
                departure,
                aircraft_registration,
                model
            """
        ).fetchdf()
        connection.register("flight_level", flight_level)

        utilization_by_aircraft = connection.execute(
            """
            SELECT
                aircraft_registration,
                model,
                COUNT(*) AS flights_operated,
                SUM(observed_flight_hours) AS flight_hours_operated,
                SUM(route_distance) AS distance_operated,
                AVG(observed_flight_hours) AS avg_hours_per_flight,
                MAX(total_flight_distance) AS lifetime_total_flight_distance,
                MAX(crew_members) AS crew_members,
                AVG(fuel_gallons_hour) AS avg_fuel_gallons_hour
            FROM flight_level
            GROUP BY
                aircraft_registration,
                model
            ORDER BY flight_hours_operated DESC
            """
        ).fetchdf()

        daily_operations = connection.execute(
            """
            SELECT
                flight_date,
                COUNT(*) AS flights_operated,
                SUM(observed_flight_hours) AS flight_hours_operated,
                SUM(route_distance) AS distance_operated
            FROM flight_level
            GROUP BY flight_date
            ORDER BY flight_date
            """
        ).fetchdf()

        fuel_efficiency_by_model = connection.execute(
            """
            SELECT
                model,
                COUNT(*) AS flights_operated,
                AVG(fuel_gallons_hour) AS avg_fuel_gallons_hour,
                AVG(route_distance) AS avg_route_distance,
                AVG(observed_flight_hours) AS avg_flight_hours,
                CASE
                    WHEN SUM(route_distance) = 0 THEN 0
                    ELSE SUM(fuel_gallons_hour * observed_flight_hours) * 100.0 / SUM(route_distance)
                END AS gallons_per_100_miles
            FROM flight_level
            GROUP BY model
            ORDER BY gallons_per_100_miles ASC
            """
        ).fetchdf()

        maintenance_snapshot = connection.execute(
            """
            SELECT
                aircraft_registration,
                model,
                MAX(maintenance_last_acheck) AS maintenance_last_acheck,
                MAX(maintenance_last_bcheck) AS maintenance_last_bcheck,
                MAX(maintenance_takeoffs) AS maintenance_takeoffs,
                MAX(maintenance_flight_hours) AS maintenance_flight_hours,
                MAX(total_flight_distance) AS total_flight_distance
            FROM flight_level
            GROUP BY
                aircraft_registration,
                model
            """
        ).fetchdf()
    finally:
        connection.close()

    if maintenance_snapshot.empty:
        maintenance_alerts = maintenance_snapshot.copy()
    else:
        alerts = maintenance_snapshot.copy()
        today = reference_date or date.today()

        for column in ["maintenance_last_acheck", "maintenance_last_bcheck"]:
            alerts[column] = pd.to_datetime(alerts[column], errors="coerce")

        alerts["days_since_acheck"] = (
            pd.Timestamp(today) - alerts["maintenance_last_acheck"]
        ).dt.days.fillna(0)
        alerts["days_since_bcheck"] = (
            pd.Timestamp(today) - alerts["maintenance_last_bcheck"]
        ).dt.days.fillna(0)

        takeoff_threshold = max(int(maintenance_takeoffs_threshold), 1)
        flight_hours_threshold = max(int(maintenance_flight_hours_threshold), 1)
        a_days_threshold = max(int(a_check_days_threshold), 1)
        b_days_threshold = max(int(b_check_days_threshold), 1)

        alerts["takeoff_ratio"] = (
            pd.to_numeric(alerts["maintenance_takeoffs"], errors="coerce").fillna(0.0)
            / takeoff_threshold
        )
        alerts["flight_hours_ratio"] = (
            pd.to_numeric(alerts["maintenance_flight_hours"], errors="coerce").fillna(0.0)
            / flight_hours_threshold
        )
        alerts["a_check_ratio"] = (
            pd.to_numeric(alerts["days_since_acheck"], errors="coerce").fillna(0.0) / a_days_threshold
        )
        alerts["b_check_ratio"] = (
            pd.to_numeric(alerts["days_since_bcheck"], errors="coerce").fillna(0.0) / b_days_threshold
        )
        alerts["risk_score"] = alerts[
            ["takeoff_ratio", "flight_hours_ratio", "a_check_ratio", "b_check_ratio"]
        ].max(axis=1)
        alerts["status"] = alerts["risk_score"].apply(
            lambda score: _status_from_score(float(score), float(warning_ratio))
        )
        alerts["primary_trigger"] = alerts.apply(_primary_trigger, axis=1)
        maintenance_alerts = alerts.sort_values(
            by=["risk_score", "maintenance_takeoffs", "maintenance_flight_hours"],
            ascending=[False, False, False],
        )

    active_aircraft = int(utilization_by_aircraft["aircraft_registration"].nunique())
    total_flights = int(len(working))
    total_flight_hours = float(
        pd.to_numeric(utilization_by_aircraft["flight_hours_operated"], errors="coerce").fillna(0.0).sum()
    )
    avg_utilization = total_flight_hours / active_aircraft if active_aircraft else 0.0
    total_route_distance = float(
        pd.to_numeric(utilization_by_aircraft["distance_operated"], errors="coerce").fillna(0.0).sum()
    )
    avg_fuel_gph = float(
        pd.to_numeric(utilization_by_aircraft["avg_fuel_gallons_hour"], errors="coerce")
        .fillna(0.0)
        .mean()
        if not utilization_by_aircraft.empty
        else 0.0
    )
    at_risk_aircraft = int((maintenance_alerts.get("status", pd.Series()) != "Healthy").sum())

    return {
        "kpis": {
            "active_aircraft": active_aircraft,
            "total_flights": total_flights,
            "total_flight_hours": total_flight_hours,
            "avg_utilization_hours_per_aircraft": avg_utilization,
            "total_route_distance": total_route_distance,
            "avg_fuel_gallons_hour": avg_fuel_gph,
            "at_risk_aircraft": at_risk_aircraft,
        },
        "utilization_by_aircraft": utilization_by_aircraft,
        "daily_operations": daily_operations,
        "fuel_efficiency_by_model": fuel_efficiency_by_model,
        "maintenance_alerts": maintenance_alerts,
    }

