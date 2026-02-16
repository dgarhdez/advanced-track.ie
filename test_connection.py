import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

import pandas as pd

# Load environment variables
load_dotenv()

username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")

# Option 1: Build connection URL using SQLAlchemy's URL helper
connection_url = URL.create(
    "db2+ibm_db",
    username=username,
    password=password,
    host=host,
    port=port,
    database=database
)
engine = create_engine(connection_url)


# Test connection
try:
    with engine.connect() as connection:
        print("Successfully connected to the database!")
except Exception as e:
    print(f"Error connecting to database: {e}")


query = "SELECT * FROM IEPLANE.FLIGHTS FETCH FIRST 10 ROWS ONLY"
df = pd.read_sql(query, engine)

print(df.head())