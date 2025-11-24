import pandas as pd
from haversine import haversine
from sklearn.cluster import KMeans
from src.utils.time import robust_hour_of_iso_date


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    peak_hours=df['event_hour'].value_counts()
    peak_hours=peak_hours[peak_hours>9500]
    df['is_peak_hour']=df['event_hour'].apply(lambda x:1 if x in peak_hours else 0)
    return df

def week(df:pd.DataFrame) -> pd.DataFrame:
    df['datetime_timestamp'] = pd.to_datetime(df['event_timestamp'], errors='coerce')
    df['week']=df['datetime_timestamp'].dt.weekday
    return df

def customer_cluster(df, kmeans_pickup=None):
    pickup_locations = df[['pickup_latitude', 'pickup_longitude']].dropna()

    if kmeans_pickup is None:
        kmeans_pickup = KMeans(n_clusters=5, random_state=42)
        pickup_clusters = kmeans_pickup.fit_predict(pickup_locations)
    else:
        pickup_clusters = kmeans_pickup.predict(pickup_locations)

    df.loc[pickup_locations.index, 'customer_cluster_label'] = pickup_clusters
    return df, kmeans_pickup

def driver_cluster(df, kmeans_driver=None):
    driver_locations = df[['driver_latitude', 'driver_longitude']].dropna()

    if kmeans_driver is None:
        kmeans_driver = KMeans(n_clusters=5, random_state=42)
        driver_clusters = kmeans_driver.fit_predict(driver_locations)
    else:
        driver_clusters = kmeans_driver.predict(driver_locations)

    df.loc[driver_locations.index, 'driver_cluster_label'] = driver_clusters
    return df, kmeans_driver


def driver_historical_completed_bookings(
    df: pd.DataFrame, historical_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Adds a feature to the DataFrame that represents the number of historical bookings
    completed by each driver. For test data, it uses precomputed historical data.

    Args:
        df (pd.DataFrame): Input DataFrame.
        historical_data (pd.DataFrame, optional): Precomputed historical data for test data.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'historical_completed_bookings'.
    """
    if historical_data is None:
        # Compute historical completed bookings from the current DataFrame (training data)
        completed_bookings = df[df["is_completed"] == 1]
        historical_counts = completed_bookings.groupby("driver_id").size().reset_index(
            name="historical_completed_bookings"
        )
    else:
        # Use precomputed historical data for test data
        historical_counts = historical_data[["driver_id", "historical_completed_bookings"]]


    historical_counts=historical_counts.drop_duplicates(subset=['driver_id'], keep='last')

    # Merge the counts back to the original DataFrame
    df = df.merge(historical_counts, on="driver_id", how="left")

    # Fill NaN values with 0 for drivers with no completed bookings
    df["historical_completed_bookings"] = df["historical_completed_bookings"].fillna(0).astype(int)
    return df
