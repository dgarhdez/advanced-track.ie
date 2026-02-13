import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from database_connector import DatabaseConnector


def write_env_file(path: Path, lines: list[str]) -> Path:
    env_path = path / ".env"
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


class TestDatabaseConnectorUnit(unittest.TestCase):
    def test_missing_environment_variable_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = write_env_file(
                Path(tmp_dir),
                [
                    "DB_USERNAME=test_user",
                    "DB_PASSWORD=test_password",
                    "DB_HOST=localhost",
                    "DB_PORT=25010",
                ],
            )

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError) as error:
                    DatabaseConnector(env_path=env_path)

            self.assertIn("DB_NAME", str(error.exception))

    def test_execute_query_returns_dataframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = write_env_file(
                Path(tmp_dir),
                [
                    "DB_USERNAME=test_user",
                    "DB_PASSWORD=test_password",
                    "DB_HOST=localhost",
                    "DB_PORT=25010",
                    "DB_NAME=test_db",
                ],
            )

            fake_engine = MagicMock(name="engine")
            expected_df = pd.DataFrame({"id": [1]})
            query = "SELECT 1 AS id"

            with patch.dict(os.environ, {}, clear=True):
                with patch("database_connector.create_engine", return_value=fake_engine):
                    connector = DatabaseConnector(env_path=env_path)

                with patch(
                    "database_connector.pd.read_sql", return_value=expected_df
                ) as mock_read_sql:
                    result_df = connector.execute_query(query, params={"id": 1})

            self.assertIs(result_df, expected_df)
            self.assertEqual(str(mock_read_sql.call_args.args[0]), query)
            self.assertIs(mock_read_sql.call_args.args[1], connector.engine)
            self.assertEqual(mock_read_sql.call_args.kwargs["params"], {"id": 1})

    def test_test_connection_runs_probe_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = write_env_file(
                Path(tmp_dir),
                [
                    "DB_USERNAME=test_user",
                    "DB_PASSWORD=test_password",
                    "DB_HOST=localhost",
                    "DB_PORT=25010",
                    "DB_NAME=test_db",
                ],
            )

            fake_connection = MagicMock(name="connection")
            fake_context_manager = MagicMock(name="context_manager")
            fake_context_manager.__enter__.return_value = fake_connection

            fake_engine = MagicMock(name="engine")
            fake_engine.connect.return_value = fake_context_manager

            with patch.dict(os.environ, {}, clear=True):
                with patch("database_connector.create_engine", return_value=fake_engine):
                    connector = DatabaseConnector(env_path=env_path)

            self.assertTrue(connector.test_connection())
            self.assertEqual(
                str(fake_connection.execute.call_args.args[0]),
                "SELECT 1 FROM SYSIBM.SYSDUMMY1",
            )


class TestDatabaseConnectorIntegration(unittest.TestCase):
    def test_connection_and_data_fetch(self) -> None:
        if os.getenv("RUN_DB_INTEGRATION_TESTS") != "1":
            self.skipTest("Set RUN_DB_INTEGRATION_TESTS=1 to run integration DB tests.")

        connector = DatabaseConnector()
        try:
            try:
                self.assertTrue(connector.test_connection())
                df = connector.execute_query(
                    "SELECT * FROM IEPLANE.FLIGHTS FETCH FIRST 10 ROWS ONLY"
                )
            except SQLAlchemyError as error:
                self.skipTest(f"DB integration unavailable: {error}")

            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df.columns), 0)
        finally:
            connector.dispose()


if __name__ == "__main__":
    unittest.main()
