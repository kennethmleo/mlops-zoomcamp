from mage_ai.data_preparation.decorators import custom
import pandas as pd
from datetime import date
from pathlib import Path

# ---- Your original helper (unchanged) ----
def read_dataframe(year, month):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df

def _next_year_month(year, month):
    return (year, month + 1) if month < 12 else (year + 1, 1)

@data_loader
def load_data(**kwargs):
    """
    Inputs (pipeline variables):
      - year, month  → the TRAIN month (we’ll pass “two months ago” via the trigger)
    Returns:
      dict with df_train, df_val and meta about months used.
    """
    year = int(kwargs.get('year'))
    month = int(kwargs.get('month'))

    # Validation is always the month after training:
    val_year, val_month = _next_year_month(year, month)

    print(f"[load_data] train={year}-{month:02d}, val={val_year}-{val_month:02d}")
    df_train = read_dataframe(year=year, month=month)
    df_val = read_dataframe(year=val_year, month=val_month)

    return {
        'df_train': df_train,
        'df_val': df_val,
        'train_year': year,
        'train_month': month,
        'val_year': val_year,
        'val_month': val_month,
    }
