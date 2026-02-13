import os
import platform
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

class DatabaseConnector:
    def __init__(self):
        load_dotenv()
        self.username = os.getenv("DB_USERNAME")
        self.password = os.getenv("DB_PASSWORD")
        self.host = os.getenv("DB_HOST", "52.211.123.34")
        self.port = os.getenv("DB_PORT", "25010")
        self.dbname = os.getenv("DB_NAME", "IEMASTER")
        
        self.engine = None
        self._init_connection()

    def _init_connection(self):
        # Handle IBM DB2 driver loading on Windows
        if platform.system() == 'Windows':
            import site
            import glob
            
            # Try to find the clidriver/bin directory in site-packages
            site_packages = site.getsitepackages()
            clidriver_path = None
            
            for sp in site_packages:
                possible_path = os.path.join(sp, 'clidriver', 'bin')
                if os.path.exists(possible_path):
                    clidriver_path = possible_path
                    break
                    
            if clidriver_path:
                os.add_dll_directory(clidriver_path)
            else:
                # Fallback: check if we are in a venv and can find it relative to that
                venv_clidriver = os.path.join(os.getcwd(), '.venv', 'Lib', 'site-packages', 'clidriver', 'bin')
                if os.path.exists(venv_clidriver):
                    os.add_dll_directory(venv_clidriver)

        connection_string = f"db2+ibm_db://{self.username}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        self.engine = create_engine(connection_string)

    def get_query_as_df(self, query):
        """Executes a SQL query and returns the result as a pandas DataFrame."""
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(query, connection)
                return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()

# Usage Example:
# db = DatabaseConnector()
# df = db.get_query_as_df("SELECT * FROM IEPLANE.FLIGHTS FETCH FIRST 5 ROWS ONLY")
# print(df)
