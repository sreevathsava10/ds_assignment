from abc import ABC, abstractmethod
from typing import Dict, List
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, f1_score, precision_recall_curve

class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame):
        y_true = df_test[self.target].values
        probs = self.clf.predict_proba(df_test[self.features].values)
        pos_class_index = self.clf.classes_.tolist().index(1)
        y_proba = probs[:, pos_class_index]
        thresholds = np.linspace(0.1, 0.9, 17)
        best_threshold = 0.5
        best_f1 = -1

        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            f1 = f1_score(y_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        y_pred = (y_proba >= best_threshold).astype(int)
        class_report = classification_report(y_true, y_pred, output_dict=True)

        return {
            "classification_report": class_report,
            "best_threshold": float(best_threshold),
            "best_f1": float(best_f1)
        }



    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features].values)[:, 1]
