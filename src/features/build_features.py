import pandas as pd
from src.utils.store import AssignmentStore
from sklearn.model_selection import train_test_split
from src.utils.store import AssignmentStore

from src.features.transformations import (
    driver_distance_to_pickup,
    driver_historical_completed_bookings,
    hour_of_day,
    customer_cluster,
    driver_cluster,
    week,
)



def main():
    store = AssignmentStore()

    dataset = store.get_processed("dataset.csv")
    dataset = apply_feature_engineering(dataset)

    store.put_processed("transformed_dataset.csv", dataset)


def apply_feature_engineering(df: pd.DataFrame, historical_data: pd.DataFrame = None,cust_cluster=None,dr_cluster=None) -> pd.DataFrame:
    store = AssignmentStore()
    df = driver_distance_to_pickup(df)
    df= hour_of_day(df)
    df = driver_historical_completed_bookings(df, historical_data)
    df,pickup_cluster = customer_cluster(df,cust_cluster)
    if(cust_cluster==None): store.put_model("pickup_cluster.pkl", pickup_cluster)
    df,driver_location=driver_cluster(df,dr_cluster)
    if(dr_cluster==None): store.put_model("driver_cluster.pkl", driver_location)
    return df


if __name__ == "__main__":
    main()
