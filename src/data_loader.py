import pandas as pd
from config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Data Loaded Successfully")
    print("Columns:", df.columns)
    return df