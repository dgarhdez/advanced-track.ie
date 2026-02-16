from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Any, Iterable

import duckdb
import pandas as pd

from database_connector import DatabaseConnector

DEFAULT_SCHEMA = "IEPLANE"
DEFAULT_PILOT_ANNUAL_HOURS = 900.0
DEFAULT_ATTENDANT_ANNUAL_HOURS = 1000.0


def sanitize_schema(schema: str | None) -> str:
    candidate = (schema or DEFAULT_SCHEMA).strip().upper()
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", candidate):
        raise ValueError("Schema name can only contain letters, numbers, and underscores.")
    return candidate


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
) -> dict[str, Any]:
    schema_name = sanitize_schema(schema)
    connector = DatabaseConnector(env_path=env_path)
    try:
        bounds_query = f"""
            SELECT
                MIN(DATE(departure)) AS min_date,
                MAX(DATE(departure)) AS max_date
            FROM {schema_name}.TICKETS
        """
        class_query = f"""
            SELECT DISTINCT class AS ticket_class
            FROM {schema_name}.TICKETS
            ORDER BY ticket_class
        """
        route_query = f"""
            SELECT DISTINCT route_code
            FROM {schema_name}.ROUTES
            ORDER BY route_code
        """

        bounds_df = connector.execute_query(bounds_query)
        classes_df = connector.execute_query(class_query)
        routes_df = connector.execute_query(route_query)
    finally:
        connector.dispose()

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
) -> pd.DataFrame:
    schema_name = sanitize_schema(schema)
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
            FROM {schema_name}.TICKETS t
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
                f.departure AS departure,
                f.airplane AS airplane,
                r.origin AS origin,
                r.destination AS destination,
                COALESCE(r.distance, 0) AS distance,
                COALESCE(r.flight_minutes, 0) AS flight_minutes,
                MAX(COALESCE(a.fuel_gallons_hour, 0)) AS fuel_gallons_hour,
                MAX(
                    COALESCE(a.seats_business, 0)
                    + COALESCE(a.seats_premium, 0)
                    + COALESCE(a.seats_economy, 0)
                ) AS total_seats
            FROM {schema_name}.FLIGHTS f
            INNER JOIN {schema_name}.ROUTES r
                ON f.route_code = r.route_code
            INNER JOIN {schema_name}.AIRPLANES a
                ON f.airplane = a.aircraft_registration
            GROUP BY
                f.flight_id,
                f.route_code,
                f.departure,
                f.airplane,
                r.origin,
                r.destination,
                r.distance,
                r.flight_minutes
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
            AND tr.departure = fd.departure
    """

    connector = DatabaseConnector(env_path=env_path)
    try:
        df = connector.execute_query(query, params=params)
    finally:
        connector.dispose()

    return normalize_financial_base_data(df)


def normalize_financial_base_data(df: pd.DataFrame) -> pd.DataFrame:
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
        "pilot_count",
        "attendant_count",
        "estimated_staff_cost",
    ]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)

    if "departure" in normalized.columns:
        normalized["departure"] = pd.to_datetime(normalized["departure"], errors="coerce")
    if "flight_date" in normalized.columns:
        normalized["flight_date"] = pd.to_datetime(normalized["flight_date"], errors="coerce").dt.date

    return normalized


def _quoted_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _get_table_columns(
    connector: DatabaseConnector,
    schema_name: str,
    table_name: str,
) -> list[str]:
    query = """
        SELECT colname
        FROM SYSCAT.COLUMNS
        WHERE UPPER(tabschema) = :schema_name
          AND UPPER(tabname) = :table_name
        ORDER BY colno
    """
    columns_df = connector.execute_query(
        query,
        params={"schema_name": schema_name.upper(), "table_name": table_name.upper()},
    )
    if columns_df.empty:
        return []
    columns_df.columns = [column.lower() for column in columns_df.columns]
    return [str(value) for value in columns_df["colname"].dropna().tolist()]


def _resolve_column(columns: list[str], candidates: Iterable[str]) -> str | None:
    lookup = {column.upper(): column for column in columns}
    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]
    return None


def apply_staff_cost_model(
    base_df: pd.DataFrame,
    crew_assignments_df: pd.DataFrame,
    avg_pilot_salary: float,
    avg_attendant_salary: float,
    pilot_annual_hours: float = DEFAULT_PILOT_ANNUAL_HOURS,
    attendant_annual_hours: float = DEFAULT_ATTENDANT_ANNUAL_HOURS,
) -> pd.DataFrame:
    enriched = normalize_financial_base_data(base_df)

    for column in ("pilot_count", "attendant_count", "estimated_staff_cost"):
        if column not in enriched.columns:
            enriched[column] = 0.0

    if crew_assignments_df.empty:
        return enriched

    crew = crew_assignments_df.copy()
    crew.columns = [column.lower() for column in crew.columns]
    required_keys = ["flight_id", "route_code", "departure"]
    if not all(key in crew.columns and key in enriched.columns for key in required_keys):
        return enriched

    crew["departure"] = pd.to_datetime(crew["departure"], errors="coerce")
    for role_column in ("pilot_count", "attendant_count"):
        if role_column in crew.columns:
            crew[role_column] = pd.to_numeric(crew[role_column], errors="coerce").fillna(0.0)
        else:
            crew[role_column] = 0.0

    enriched = enriched.merge(
        crew[required_keys + ["pilot_count", "attendant_count"]],
        on=required_keys,
        how="left",
        suffixes=("", "_crew"),
    )

    for role_column in ("pilot_count", "attendant_count"):
        crew_column = f"{role_column}_crew"
        if crew_column in enriched.columns:
            enriched[role_column] = pd.to_numeric(
                enriched[crew_column], errors="coerce"
            ).fillna(0.0)
            enriched = enriched.drop(columns=[crew_column])

    pilot_hourly = (
        float(avg_pilot_salary) / float(pilot_annual_hours)
        if float(pilot_annual_hours) > 0 and float(avg_pilot_salary) > 0
        else 0.0
    )
    attendant_hourly = (
        float(avg_attendant_salary) / float(attendant_annual_hours)
        if float(attendant_annual_hours) > 0 and float(avg_attendant_salary) > 0
        else 0.0
    )
    flight_hours = pd.to_numeric(enriched["flight_minutes"], errors="coerce").fillna(0.0) / 60.0
    enriched["estimated_staff_cost"] = flight_hours * (
        enriched["pilot_count"] * pilot_hourly + enriched["attendant_count"] * attendant_hourly
    )

    return normalize_financial_base_data(enriched)


def enrich_financial_base_with_staff_costs(
    base_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    env_path: str = ".env",
    schema: str | None = None,
    pilot_annual_hours: float = DEFAULT_PILOT_ANNUAL_HOURS,
    attendant_annual_hours: float = DEFAULT_ATTENDANT_ANNUAL_HOURS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema_name = sanitize_schema(schema)
    normalized_base = normalize_financial_base_data(base_df)
    metadata: dict[str, Any] = {
        "staff_costs_applied": False,
        "message": "Federated labor cost model was not applied.",
        "avg_pilot_salary": 0.0,
        "avg_attendant_salary": 0.0,
        "pilot_annual_hours": float(pilot_annual_hours),
        "attendant_annual_hours": float(attendant_annual_hours),
    }

    if normalized_base.empty:
        metadata["message"] = "No base data available for staff cost enrichment."
        return normalized_base, metadata

    connector = DatabaseConnector(env_path=env_path)
    try:
        staff_columns = _get_table_columns(connector, schema_name=schema_name, table_name="STAFF")
        crew_columns = _get_table_columns(
            connector, schema_name=schema_name, table_name="FLIGHT_CREW"
        )

        if not staff_columns or not crew_columns:
            metadata["message"] = (
                f"Could not find federated tables STAFF/FLIGHT_CREW in schema {schema_name}."
            )
            return normalized_base, metadata

        staff_emp_col = _resolve_column(staff_columns, ["EMPNO", "EMP_ID", "EMPLOYEE_ID", "ID"])
        staff_department_col = _resolve_column(staff_columns, ["DEPARTMENT", "DEPT"])
        staff_salary_col = _resolve_column(
            staff_columns, ["SALARY", "ANNUAL_SALARY", "BASE_SALARY"]
        )
        crew_emp_col = _resolve_column(crew_columns, ["EMPNO", "EMP_ID", "EMPLOYEE_ID", "ID"])
        crew_flight_id_col = _resolve_column(crew_columns, ["FLIGHT_ID", "FLIGHTID"])
        crew_route_code_col = _resolve_column(crew_columns, ["ROUTE_CODE", "ROUTE"])
        crew_departure_col = _resolve_column(
            crew_columns, ["DEPARTURE", "FLIGHT_DATE", "DATE", "SCHEDULED_DEPARTURE"]
        )

        required_pairs = {
            "staff employee": staff_emp_col,
            "staff department": staff_department_col,
            "staff salary": staff_salary_col,
            "crew employee": crew_emp_col,
            "crew flight_id": crew_flight_id_col,
            "crew route_code": crew_route_code_col,
            "crew departure": crew_departure_col,
        }
        missing_required = [name for name, value in required_pairs.items() if value is None]
        if missing_required:
            metadata["message"] = (
                "Federated labor columns missing: " + ", ".join(sorted(missing_required))
            )
            return normalized_base, metadata

        q_staff_emp = _quoted_identifier(str(staff_emp_col))
        q_staff_department = _quoted_identifier(str(staff_department_col))
        q_staff_salary = _quoted_identifier(str(staff_salary_col))
        q_crew_emp = _quoted_identifier(str(crew_emp_col))
        q_crew_flight_id = _quoted_identifier(str(crew_flight_id_col))
        q_crew_route_code = _quoted_identifier(str(crew_route_code_col))
        q_crew_departure = _quoted_identifier(str(crew_departure_col))

        staff_role_expr = (
            f"CASE "
            f"WHEN REPLACE(UPPER(COALESCE(s.{q_staff_department}, '')), ' ', '') LIKE '%PILOT%' "
            f"THEN 'PILOT' "
            f"WHEN REPLACE(UPPER(COALESCE(s.{q_staff_department}, '')), ' ', '') LIKE '%FLIGHTATTENDANT%' "
            f"  OR REPLACE(UPPER(COALESCE(s.{q_staff_department}, '')), ' ', '') LIKE '%ATTENDANT%' "
            f"THEN 'FLIGHT_ATTENDANT' "
            f"ELSE 'OTHER' "
            f"END"
        )

        salary_query = f"""
            WITH staff_roles AS (
                SELECT
                    {staff_role_expr} AS staff_role,
                    DOUBLE(s.{q_staff_salary}) AS salary_amount
                FROM {schema_name}.STAFF s
            )
            SELECT
                AVG(CASE WHEN staff_role = 'PILOT' THEN salary_amount END) AS avg_pilot_salary,
                AVG(CASE WHEN staff_role = 'FLIGHT_ATTENDANT' THEN salary_amount END)
                    AS avg_attendant_salary
            FROM staff_roles
        """
        salary_df = connector.execute_query(salary_query)
        salary_df.columns = [column.lower() for column in salary_df.columns]
        avg_pilot_salary = float(salary_df.loc[0, "avg_pilot_salary"] or 0.0)
        avg_attendant_salary = float(salary_df.loc[0, "avg_attendant_salary"] or 0.0)

        params = {
            "start_departure": datetime.combine(start_date, time.min),
            "end_departure": datetime.combine(end_date + timedelta(days=1), time.min),
        }
        crew_query = f"""
            WITH staff_roles AS (
                SELECT
                    s.{q_staff_emp} AS staff_empno,
                    {staff_role_expr} AS staff_role
                FROM {schema_name}.STAFF s
            ),
            crew_roles AS (
                SELECT
                    c.{q_crew_flight_id} AS flight_id,
                    c.{q_crew_route_code} AS route_code,
                    c.{q_crew_departure} AS departure,
                    sr.staff_role
                FROM {schema_name}.FLIGHT_CREW c
                INNER JOIN staff_roles sr
                    ON c.{q_crew_emp} = sr.staff_empno
                WHERE c.{q_crew_departure} >= :start_departure
                  AND c.{q_crew_departure} < :end_departure
            )
            SELECT
                flight_id,
                route_code,
                departure,
                SUM(CASE WHEN staff_role = 'PILOT' THEN 1 ELSE 0 END) AS pilot_count,
                SUM(CASE WHEN staff_role = 'FLIGHT_ATTENDANT' THEN 1 ELSE 0 END) AS attendant_count
            FROM crew_roles
            GROUP BY
                flight_id,
                route_code,
                departure
        """
        crew_df = connector.execute_query(crew_query, params=params)
        enriched_base = apply_staff_cost_model(
            base_df=normalized_base,
            crew_assignments_df=crew_df,
            avg_pilot_salary=avg_pilot_salary,
            avg_attendant_salary=avg_attendant_salary,
            pilot_annual_hours=pilot_annual_hours,
            attendant_annual_hours=attendant_annual_hours,
        )

        crew_df.columns = [column.lower() for column in crew_df.columns]
        covered_flights = int(
            crew_df[["flight_id", "route_code", "departure"]].drop_duplicates().shape[0]
            if not crew_df.empty
            else 0
        )
        metadata.update(
            {
                "staff_costs_applied": True,
                "message": (
                    "Federated staff costs applied using STAFF and FLIGHT_CREW "
                    f"({covered_flights} crewed flights in selected period)."
                ),
                "avg_pilot_salary": avg_pilot_salary,
                "avg_attendant_salary": avg_attendant_salary,
                "covered_flights": covered_flights,
            }
        )
        return enriched_base, metadata
    except Exception as error:
        metadata["message"] = f"Federated staff cost model unavailable: {error}"
        return normalized_base, metadata
    finally:
        connector.dispose()


def compute_financial_views(
    base_df: pd.DataFrame,
    fuel_price_per_gallon: float,
) -> dict[str, Any]:
    if base_df.empty:
        return {
            "kpis": {
                "total_revenue": 0.0,
                "ancillary_revenue": 0.0,
                "available_seat_miles": 0.0,
                "rasm": 0.0,
                "estimated_fuel_cost": 0.0,
                "estimated_staff_cost": 0.0,
                "estimated_total_cost": 0.0,
                "estimated_profit": 0.0,
                "ancillary_share": 0.0,
            },
            "daily_revenue": pd.DataFrame(),
            "route_profitability": pd.DataFrame(),
            "ancillary_by_class": pd.DataFrame(),
        }

    fuel_price = float(fuel_price_per_gallon)
    working = normalize_financial_base_data(base_df)
    if "estimated_staff_cost" not in working.columns:
        working["estimated_staff_cost"] = 0.0
    if "pilot_count" not in working.columns:
        working["pilot_count"] = 0.0
    if "attendant_count" not in working.columns:
        working["attendant_count"] = 0.0

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
                MAX(pilot_count) AS pilot_count,
                MAX(attendant_count) AS attendant_count,
                MAX(estimated_staff_cost) AS estimated_staff_cost,
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
                SUM(estimated_staff_cost) AS estimated_staff_cost,
                SUM(((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price}) + estimated_staff_cost)
                    AS estimated_total_cost,
                SUM(
                    total_revenue
                    - (((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price}) + estimated_staff_cost)
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
                SUM(pilot_count) AS pilot_assignments,
                SUM(attendant_count) AS attendant_assignments,
                SUM(total_revenue) AS route_revenue,
                SUM(total_seats * distance) AS route_available_seat_miles,
                CASE
                    WHEN SUM(total_seats * distance) = 0 THEN 0
                    ELSE SUM(total_revenue) / SUM(total_seats * distance)
                END AS route_rasm,
                SUM((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price}) AS estimated_fuel_cost,
                SUM(estimated_staff_cost) AS estimated_staff_cost,
                SUM(((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price}) + estimated_staff_cost)
                    AS estimated_total_cost,
                SUM(
                    total_revenue
                    - (((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price}) + estimated_staff_cost)
                ) AS estimated_profit,
                CASE
                    WHEN SUM(total_revenue) = 0 THEN 0
                    ELSE SUM(
                        total_revenue
                        - (((fuel_gallons_hour * flight_minutes / 60.0) * {fuel_price}) + estimated_staff_cost)
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

