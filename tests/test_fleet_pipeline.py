from __future__ import annotations

from datetime import date
import unittest

import pandas as pd

from dashboard.fleet_pipeline import (
    build_fleet_filters,
    compute_fleet_views,
    normalize_fleet_base_data,
)


class TestFleetPipeline(unittest.TestCase):
    def test_build_fleet_filters_with_models(self) -> None:
        where_clause, params = build_fleet_filters(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            models=["A320", "B737"],
        )

        self.assertIn("f.departure >= :start_departure", where_clause)
        self.assertIn("f.departure < :end_departure", where_clause)
        self.assertIn("a.model IN (:model_0, :model_1)", where_clause)
        self.assertEqual(params["model_0"], "A320")
        self.assertEqual(params["model_1"], "B737")

    def test_normalize_fleet_base_data_derives_flight_hours(self) -> None:
        base_df = pd.DataFrame(
            [
                {
                    "flight_id": "IE1001",
                    "route_code": "R001",
                    "departure": "2025-01-10 08:00:00",
                    "arrival": "2025-01-10 10:30:00",
                    "route_flight_minutes": 120,
                },
                {
                    "flight_id": "IE1002",
                    "route_code": "R002",
                    "departure": "2025-01-11 08:00:00",
                    "arrival": None,
                    "route_flight_minutes": 90,
                },
            ]
        )

        normalized = normalize_fleet_base_data(base_df)
        hours = normalized["observed_flight_hours"].tolist()
        self.assertEqual(hours, [2.5, 1.5])

    def test_compute_fleet_views_includes_maintenance_status(self) -> None:
        base_df = pd.DataFrame(
            [
                {
                    "flight_id": "IE1001",
                    "route_code": "R001",
                    "departure": "2025-01-10 08:00:00",
                    "arrival": "2025-01-10 10:00:00",
                    "aircraft_registration": "A1",
                    "model": "A320",
                    "crew_members": 6,
                    "fuel_gallons_hour": 120,
                    "maintenance_last_acheck": "2025-01-01",
                    "maintenance_last_bcheck": "2024-08-01",
                    "maintenance_takeoffs": 2800,
                    "maintenance_flight_hours": 5800,
                    "total_flight_distance": 200000,
                    "route_distance": 600,
                    "route_flight_minutes": 120,
                },
                {
                    "flight_id": "IE1002",
                    "route_code": "R002",
                    "departure": "2025-01-11 08:00:00",
                    "arrival": "2025-01-11 11:00:00",
                    "aircraft_registration": "A1",
                    "model": "A320",
                    "crew_members": 6,
                    "fuel_gallons_hour": 120,
                    "maintenance_last_acheck": "2025-01-01",
                    "maintenance_last_bcheck": "2024-08-01",
                    "maintenance_takeoffs": 2800,
                    "maintenance_flight_hours": 5800,
                    "total_flight_distance": 200000,
                    "route_distance": 900,
                    "route_flight_minutes": 180,
                },
                {
                    "flight_id": "IE2001",
                    "route_code": "R003",
                    "departure": "2025-01-12 09:00:00",
                    "arrival": "2025-01-12 11:30:00",
                    "aircraft_registration": "B1",
                    "model": "B737",
                    "crew_members": 7,
                    "fuel_gallons_hour": 160,
                    "maintenance_last_acheck": "2024-10-01",
                    "maintenance_last_bcheck": "2023-12-01",
                    "maintenance_takeoffs": 3500,
                    "maintenance_flight_hours": 7100,
                    "total_flight_distance": 260000,
                    "route_distance": 1000,
                    "route_flight_minutes": 150,
                },
            ]
        )

        views = compute_fleet_views(
            base_df=base_df,
            maintenance_takeoffs_threshold=3000,
            maintenance_flight_hours_threshold=6000,
            a_check_days_threshold=90,
            b_check_days_threshold=365,
            warning_ratio=0.85,
            reference_date=date(2025, 2, 1),
        )
        kpis = views["kpis"]
        alerts = views["maintenance_alerts"]

        self.assertEqual(int(kpis["active_aircraft"]), 2)
        self.assertEqual(int(kpis["total_flights"]), 3)
        self.assertAlmostEqual(float(kpis["total_flight_hours"]), 7.5, places=3)
        self.assertEqual(int(kpis["at_risk_aircraft"]), 2)

        status_by_aircraft = {
            row["aircraft_registration"]: row["status"] for _, row in alerts.iterrows()
        }
        self.assertEqual(status_by_aircraft["A1"], "Warning")
        self.assertEqual(status_by_aircraft["B1"], "Overdue")


if __name__ == "__main__":
    unittest.main()
