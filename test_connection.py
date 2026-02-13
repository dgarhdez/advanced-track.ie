from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
dbname = os.getenv("DB_NAME")

# Basic validation
if not all([username, password, host, port, dbname]):
    print("Error: One or more environment variables are missing. Please check your .env file.")
    exit(1)

connection_string = f"db2+ibm_db://{username}:{password}@{host}:{port}/{dbname}"

try:
    print(f"Attempting to connect to {host}:{port}/{dbname} as {username}...")
    engine = create_engine(connection_string)
    
    # Test query
    query = "SELECT * FROM IEPLANE.FLIGHTS FETCH FIRST 5 ROWS ONLY"
    df = pd.read_sql(query, engine)
    
    print("\nConnection successful!")
    print("Sample data:")
    print(df)
except Exception as e:
    print(f"\nConnection failed: {e}")
