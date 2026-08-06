from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "features" / "training_dataset_clean.csv"

METADATA = [
    "patient_id",
    "window_index",
    "start_seconds",
    "number_of_beats",
    "label",
]

data = pd.read_csv(DATA_PATH)

X = data.drop(columns=METADATA)
y = data["label"]
groups = data["patient_id"]

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
}

if HAS_XGBOOST:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

gkf = GroupKFold(n_splits=5)

results = []

for name, model in models.items():

    acc = []
    f1 = []
    auc = []

    for train_idx, test_idx in gkf.split(X, y, groups):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        acc.append(accuracy_score(y_test, pred))
        f1.append(f1_score(y_test, pred))
        auc.append(roc_auc_score(y_test, prob))

    results.append(
        {
            "Model": name,
            "Accuracy": sum(acc) / len(acc),
            "F1": sum(f1) / len(f1),
            "ROC-AUC": sum(auc) / len(auc),
        }
    )

results = pd.DataFrame(results)

print(results.sort_values("ROC-AUC", ascending=False))