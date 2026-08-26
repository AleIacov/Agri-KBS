import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


NUMERIC_FEATURES = [
    "temperature",
    "humidity",
    "rainfall",
    "leaf_wetness",
    "wind_speed",
    "night_temperature"
]

BASE_FEATURES = NUMERIC_FEATURES + [
    "crop"
]

ONTOLOGY_FEATURES = BASE_FEATURES + [
    "bk_high_risk",
    "bk_medium_risk"
]


def build_models():

    models = {}

    # LOGISTIC REGRESSION

    logistic = Pipeline([
        (
            "preprocessor",
            ColumnTransformer(
                transformers=[
                    (
                        "numeric",
                        StandardScaler(),
                        NUMERIC_FEATURES
                    ),
                    (
                        "categorical",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                        ["crop"]
                    )
                ]
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    logistic_grid = {
        "classifier__C": [
            0.1,
            1.0,
            10.0
        ]
    }

    models[
        "Logistic Regression"
    ] = (
        logistic,
        logistic_grid
    )

    # RANDOM FOREST

    random_forest = Pipeline([
        (
            "preprocessor",
            ColumnTransformer(
                transformers=[
                    (
                        "categorical",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                        ["crop"]
                    )
                ],
                remainder="passthrough"
            )
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42
            )
        )
    ])

    random_forest_grid = {
        "classifier__n_estimators": [
            100,
            200
        ],
        "classifier__max_depth": [
            3,
            5,
            None
        ],
        "classifier__min_samples_leaf": [
            1,
            3
        ]
    }

    models[
        "Random Forest"
    ] = (
        random_forest,
        random_forest_grid
    )

    return models


def evaluate_configuration(
    X,
    y,
    model,
    parameter_grid,
    outer_splits=5,
    outer_repeats=3,
    inner_splits=3
):

    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=42
    )

    scores = []

    for fold_id, (
        train_idx,
        test_idx
    ) in enumerate(
        outer_cv.split(X, y),
        start=1
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        inner_cv = RepeatedStratifiedKFold(
            n_splits=inner_splits,
            n_repeats=1,
            random_state=fold_id
        )

        grid = GridSearchCV(
            estimator=model,
            param_grid=parameter_grid,
            scoring="f1",
            cv=inner_cv,
            n_jobs=-1
        )

        grid.fit(
            X_train,
            y_train
        )

        predictions = grid.predict(
            X_test
        )

        scores.append({
            "fold": fold_id,

            "accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "best_params": str(
                grid.best_params_
            )
        })

    return pd.DataFrame(
        scores
    )


def run_ml_experiments(df):

    print(
        "\n=========================================="
    )

    print(
        "FASE 2 - MACHINE LEARNING"
    )

    print(
        "=========================================="
    )

    datasets = {

        "Raw": BASE_FEATURES,

        "OntoBK": ONTOLOGY_FEATURES

    }

    y = df[
        "disease"
    ]

    models = build_models()

    summary_rows = []
    fold_rows = []

    for data_name, features in datasets.items():

        X = df[
            features
        ]

        for model_name, (
            model,
            parameter_grid
        ) in models.items():

            print(
                f"\nModello: {model_name}"
            )

            print(
                f"Dataset: {data_name}"
            )

            results = evaluate_configuration(
                X,
                y,
                model,
                parameter_grid
            )

            results[
                "model"
            ] = model_name

            results[
                "dataset"
            ] = data_name

            fold_rows.append(
                results
            )

            for metric in [
                "accuracy",
                "precision",
                "recall",
                "f1"
            ]:

                summary_rows.append({

                    "model": model_name,

                    "dataset": data_name,

                    "metric": metric,

                    "mean": results[
                        metric
                    ].mean(),

                    "std": results[
                        metric
                    ].std()

                })

    summary = pd.DataFrame(
        summary_rows
    )

    folds = pd.concat(
        fold_rows,
        ignore_index=True
    )

    print(
        "\nRISULTATI MEDI"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return summary, folds


def train_final_model(
    df,
    model_name="Random Forest"
):

    # addestro il modello finale su tutto il dataset usando i parametri ottimali

    X = df[
        ONTOLOGY_FEATURES
    ]

    y = df[
        "disease"
    ]

    models = build_models()

    model, parameter_grid = models[
        model_name
    ]

    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    search.fit(
        X,
        y
    )

    print(
        "\nModello finale:",
        model_name
    )

    print(
        "Parametri scelti:",
        search.best_params_
    )

    return search.best_estimator_