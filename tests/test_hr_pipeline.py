from __future__ import annotations

from datetime import date
import unittest

import pandas as pd

from dashboard.hr_pipeline import (
    build_hr_filters,
    compute_hr_views,
)


class TestHRPipeline(unittest.TestCase):
    def test_build_hr_filters_with_all_dimensions(self) -> None:
        where_clause, params = build_hr_filters(
            alias="e",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            departments=["D11"],
            jobs=["PILOT"],
            genders=["F"],
        )

        self.assertIn("e.hiredate >= :start_hiredate", where_clause)
        self.assertIn("e.hiredate < :end_hiredate", where_clause)
        self.assertIn("e.workdept IN (:dept_0)", where_clause)
        self.assertIn("e.job IN (:job_0)", where_clause)
        self.assertIn("e.gender IN (:gender_0)", where_clause)
        self.assertEqual(params["dept_0"], "D11")
        self.assertEqual(params["job_0"], "PILOT")
        self.assertEqual(params["gender_0"], "F")

    def test_compute_hr_views(self) -> None:
        base_df = pd.DataFrame(
            [
                {
                    "empno": 1,
                    "deptno": "D11",
                    "deptname": "FLIGHT OPERATIONS",
                    "location": "HQ",
                    "dept_budget": 1_000_000,
                    "job": "PILOT",
                    "gender": "F",
                    "hiredate": "2020-01-10",
                    "salary": 120_000,
                    "bonus": 10_000,
                    "comm": 0,
                    "fleet_crew_capacity": 200,
                },
                {
                    "empno": 2,
                    "deptno": "D11",
                    "deptname": "FLIGHT OPERATIONS",
                    "location": "HQ",
                    "dept_budget": 1_000_000,
                    "job": "ATTENDANT",
                    "gender": "M",
                    "hiredate": "2020-02-10",
                    "salary": 60_000,
                    "bonus": 2_000,
                    "comm": 0,
                    "fleet_crew_capacity": 200,
                },
                {
                    "empno": 3,
                    "deptno": "D21",
                    "deptname": "FINANCE",
                    "location": "HQ",
                    "dept_budget": 500_000,
                    "job": "ANALYST",
                    "gender": "F",
                    "hiredate": "2020-03-10",
                    "salary": 80_000,
                    "bonus": 5_000,
                    "comm": 1_000,
                    "fleet_crew_capacity": 200,
                },
            ]
        )

        views = compute_hr_views(base_df)
        kpis = views["kpis"]
        dept = views["department_summary"]

        self.assertEqual(int(kpis["total_headcount"]), 3)
        self.assertAlmostEqual(float(kpis["total_salary"]), 260_000.0, places=3)
        self.assertAlmostEqual(float(kpis["total_compensation"]), 278_000.0, places=3)
        self.assertAlmostEqual(float(kpis["total_budget"]), 1_500_000.0, places=3)
        self.assertEqual(int(kpis["operational_headcount"]), 2)
        self.assertAlmostEqual(float(kpis["fleet_crew_capacity"]), 200.0, places=3)
        self.assertAlmostEqual(float(kpis["crew_to_operational_ratio"]), 100.0, places=3)
        self.assertEqual(len(dept), 2)


if __name__ == "__main__":
    unittest.main()
