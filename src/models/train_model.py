from sklearn.model_selection import train_test_split, RandomizedSearchCV,GridSearchCV,StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier
from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    df = store.get_processed("transformed_dataset.csv")
    df_train, df_test = train_test_split(df, test_size=config["test_size"])

    scaler = StandardScaler()
    df_train.loc[:, config["features"]] = scaler.fit_transform(df_train[config["features"]])
    df_test.loc[:, config["features"]] = scaler.transform(df_test[config["features"]])
    df_train.loc[:, config["target"]] = df_train[config["target"]].astype(int)
    df_test.loc[:, config["target"]] = df_test[config["target"]].astype(int)

    xg_estimator = XGBClassifier(**config["xgboost"])

    if config.get("gridsearch", {}).get("enable", False):
        param_grid = config["param_grid"]

        grid = GridSearchCV(
            estimator=xg_estimator,
            param_grid=param_grid,
            scoring=config["gridsearch"]["scoring"],
            cv = StratifiedKFold(n_splits=config["gridsearch"]["cv"], shuffle=True, random_state=42)
            n_jobs=config["gridsearch"]["n_jobs"],
            verbose=config["gridsearch"]["verbose"]
        )
        model_wrapper = SklearnClassifier(grid, config["features"], config["target"])
        model_wrapper.train(df_train)
        print("\nBest hyperparameters:", grid.best_params_)
        model_wrapper.estimator = grid.best_estimator_
    else:
        model_wrapper = SklearnClassifier(xg_estimator, config["features"], config["target"])
        model_wrapper.train(df_train)


    metrics = model_wrapper.evaluate(df_test)

    store.put_model("saved_model.pkl", model_wrapper)
    store.put_metrics("metrics.json", metrics)


if __name__ == "__main__":
    main()
