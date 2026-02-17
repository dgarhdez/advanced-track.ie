from __future__ import annotations

from datetime import date, datetime
import unittest

import pandas as pd

from dashboard.financial_pipeline import (
    build_ticket_filters,
    compute_financial_views,
)


class TestFinancialPipeline(unittest.TestCase):
    def test_build_ticket_filters_with_all_options(self) -> None:
        where_clause, params = build_ticket_filters(
            alias="t",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            classes=["E", "B"],
            route_codes=["R001"],
        )

        self.assertIn("t.departure >= :start_departure", where_clause)
        self.assertIn("t.departure < :end_departure", where_clause)
        self.assertIn("t.class IN (:class_0, :class_1)", where_clause)
        self.assertIn("t.route_code IN (:route_0)", where_clause)
        self.assertEqual(params["class_0"], "E")
        self.assertEqual(params["class_1"], "B")
        self.assertEqual(params["route_0"], "R001")
        self.assertEqual(params["start_departure"], datetime(2025, 1, 1, 0, 0))
        self.assertEqual(params["end_departure"], datetime(2025, 2, 1, 0, 0))

    def test_compute_financial_views_returns_expected_kpis(self) -> None:
        base_df = pd.DataFrame(
            [
                {
                    "flight_id": "F10001",
                    "route_code": "R001",
                    "departure": "2025-01-10 08:00:00",
                    "flight_date": "2025-01-10",
                    "origin": "JFK",
                    "destination": "LAX",
                    "distance": 500,
                    "flight_minutes": 60,
                    "fuel_gallons_hour": 100,
                    "total_seats": 100,
                    "ticket_class": "E",
                    "tickets_sold": 70,
                    "base_revenue": 7000,
                    "airport_tax_revenue": 700,
                    "local_tax_revenue": 300,
                    "total_revenue": 8000,
                },
                {
                    "flight_id": "F10001",
                    "route_code": "R001",
                    "departure": "2025-01-10 08:00:00",
                    "flight_date": "2025-01-10",
                    "origin": "JFK",
                    "destination": "LAX",
                    "distance": 500,
                    "flight_minutes": 60,
                    "fuel_gallons_hour": 100,
                    "total_seats": 100,
                    "ticket_class": "B",
                    "tickets_sold": 10,
                    "base_revenue": 3000,
                    "airport_tax_revenue": 300,
                    "local_tax_revenue": 100,
                    "total_revenue": 3400,
                },
                {
                    "flight_id": "F20001",
                    "route_code": "R002",
                    "departure": "2025-01-11 09:00:00",
                    "flight_date": "2025-01-11",
                    "origin": "LHR",
                    "destination": "CDG",
                    "distance": 600,
                    "flight_minutes": 90,
                    "fuel_gallons_hour": 120,
                    "total_seats": 80,
                    "ticket_class": "E",
                    "tickets_sold": 40,
                    "base_revenue": 4000,
                    "airport_tax_revenue": 400,
                    "local_tax_revenue": 200,
                    "total_revenue": 4600,
                },
            ]
        )

        views = compute_financial_views(base_df=base_df, fuel_price_per_gallon=3.0)
        kpis = views["kpis"]
        route_profitability = views["route_profitability"]

        self.assertAlmostEqual(float(kpis["total_revenue"]), 16000.0, places=3)
        self.assertAlmostEqual(float(kpis["available_seat_miles"]), 98000.0, places=3)
        self.assertAlmostEqual(float(kpis["rasm"]), 16000.0 / 98000.0, places=6)
        self.assertAlmostEqual(float(kpis["ancillary_revenue"]), 2000.0, places=3)
        self.assertAlmostEqual(float(kpis["estimated_fuel_cost"]), 840.0, places=3)
        self.assertAlmostEqual(float(kpis["estimated_total_cost"]), 840.0, places=3)
        self.assertAlmostEqual(float(kpis["estimated_profit"]), 15160.0, places=3)
        self.assertAlmostEqual(float(kpis["ancillary_share"]), 2000.0 / 16000.0, places=6)
        self.assertEqual(len(route_profitability), 2)

    def test_route_segmentation_marks_money_pit_when_unprofitable(self) -> None:
        base_df = pd.DataFrame(
            [
                {
                    "flight_id": "F30001",
                    "route_code": "R003",
                    "departure": "2025-01-12 11:00:00",
                    "flight_date": "2025-01-12",
                    "origin": "DXB",
                    "destination": "SIN",
                    "distance": 1000,
                    "flight_minutes": 120,
                    "fuel_gallons_hour": 200,
                    "total_seats": 100,
                    "ticket_class": "E",
                    "tickets_sold": 50,
                    "base_revenue": 5000,
                    "airport_tax_revenue": 500,
                    "local_tax_revenue": 200,
                    "total_revenue": 5700,
                }
            ]
        )

        views = compute_financial_views(base_df=base_df, fuel_price_per_gallon=50.0)
        route_profitability = views["route_profitability"]
        self.assertEqual(route_profitability.iloc[0]["segment"], "Money Pit")

    def test_compute_financial_views_ignores_staff_columns_if_present(self) -> None:
        base_df = pd.DataFrame(
            [
                {
                    "flight_id": "IE2001",
                    "route_code": "R001",
                    "departure": "2025-01-10 08:00:00",
                    "flight_date": "2025-01-10",
                    "origin": "JFK",
                    "destination": "LAX",
                    "distance": 500,
                    "flight_minutes": 60,
                    "fuel_gallons_hour": 100,
                    "total_seats": 100,
                    "ticket_class": "E",
                    "tickets_sold": 70,
                    "base_revenue": 7000,
                    "airport_tax_revenue": 700,
                    "local_tax_revenue": 300,
                    "total_revenue": 8000,
                    "estimated_staff_cost": 9999,
                    "pilot_count": 2,
                    "attendant_count": 4,
                },
                {
                    "flight_id": "IE2001",
                    "route_code": "R001",
                    "departure": "2025-01-10 08:00:00",
                    "flight_date": "2025-01-10",
                    "origin": "JFK",
                    "destination": "LAX",
                    "distance": 500,
                    "flight_minutes": 60,
                    "fuel_gallons_hour": 100,
                    "total_seats": 100,
                    "ticket_class": "B",
                    "tickets_sold": 10,
                    "base_revenue": 3000,
                    "airport_tax_revenue": 300,
                    "local_tax_revenue": 100,
                    "total_revenue": 3400,
                    "estimated_staff_cost": 9999,
                    "pilot_count": 2,
                    "attendant_count": 4,
                },
            ]
        )
        views = compute_financial_views(base_df=base_df, fuel_price_per_gallon=3.0)
        kpis = views["kpis"]
        self.assertAlmostEqual(float(kpis["estimated_fuel_cost"]), 300.0, places=3)
        self.assertAlmostEqual(float(kpis["estimated_total_cost"]), 300.0, places=3)
        self.assertAlmostEqual(float(kpis["estimated_profit"]), 11100.0, places=3)


if __name__ == "__main__":
    unittest.main()
