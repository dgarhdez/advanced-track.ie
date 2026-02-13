from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import pandas as pd
import platform

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
        print(f"Added clidriver path: {clidriver_path}")
    else:
        # Fallback: check if we are in a venv and can find it relative to that
        # (This handles the case where site.getsitepackages() might act differently in venv)
        venv_clidriver = os.path.join(os.getcwd(), '.venv', 'Lib', 'site-packages', 'clidriver', 'bin')
        if os.path.exists(venv_clidriver):
             os.add_dll_directory(venv_clidriver)
             print(f"Added clidriver path from venv: {venv_clidriver}")


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
