from pathlib import Path

import pandas as pd

from src.dataset import (
    generate_dataset
)

from src.ontology import (
    apply_ontology_reasoning
)

from src.ml import (
    run_ml_experiments,
    train_final_model,
    ONTOLOGY_FEATURES
)

from src.csp import (
    run_csp
)


RESULTS_DIR = Path(
    "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)


def main():

    print(
        "=" * 60
    )

    print(
        "AGRI-KBS"
    )

    print(
        "Knowledge-Based System "
        "per l'agricoltura di precisione"
    )

    print(
        "=" * 60
    )

  
    # FASE 0: dataset storico


    print(
        "\nGenerazione dataset storico..."
    )

    historical_data = generate_dataset(
        n_samples=600,
        random_state=42
    )

    historical_data.to_csv(
        RESULTS_DIR
        / "dataset_raw.csv",
        index=False
    )

    print(
        "Dataset storico:",
        historical_data.shape
    )

    # FASE 1: KNOWLEDGE BASE
 

    historical_enriched = (
        apply_ontology_reasoning(
            historical_data
        )
    )

    historical_enriched.to_csv(
        RESULTS_DIR
        / "dataset_ontoBK.csv",
        index=False
    )


    # FASE 2: MACHINE LEARNING
   

    summary, folds = (
        run_ml_experiments(
            historical_enriched
        )
    )

    summary.to_csv(
        RESULTS_DIR
        / "ml_summary.csv",
        index=False
    )

    folds.to_csv(
        RESULTS_DIR
        / "ml_fold_results.csv",
        index=False
    )


    # MODELLO FINALE
   

    final_model = (
        train_final_model(
            historical_enriched,
            model_name="Random Forest"
        )
    )

    # Nuove osservazioni operative
   
    print(
        "\nGenerazione nuove osservazioni operative..."
    )

    operational_data = generate_dataset(
        n_samples=20,
        random_state=2026
    )

    operational_enriched = (
        apply_ontology_reasoning(
            operational_data
        )
    )

    # Predizione della malattia


    operational_enriched[
        "predicted_disease"
    ] = final_model.predict(
        operational_enriched[
            ONTOLOGY_FEATURES
        ]
    )

    operational_enriched[
        "predicted_probability"
    ] = final_model.predict_proba(
        operational_enriched[
            ONTOLOGY_FEATURES
        ]
    )[:, 1]

    operational_enriched.to_csv(
        RESULTS_DIR
        / "operational_predictions.csv",
        index=False
    )

    predicted_zones = (
        operational_enriched[
            operational_enriched[
                "predicted_disease"
            ] == 1
        ]
        .sort_values(
            "predicted_probability",
            ascending=False
        )
        ["zone_id"]
        .tolist()
    )

    # FASE 3: CSP
    
    predicted_zones = (
        predicted_zones[:4]
    )

    print(
        "\nZone considerate a rischio:"
    )

    print(
        predicted_zones
    )

    if len(
        predicted_zones
    ) >= 2:

        csp_results = run_csp(
            operational_enriched,
            predicted_zones
        )

        pd.DataFrame([{

            "backtracking_solution":
                str(
                    csp_results[
                        "backtracking_solution"
                    ]
                ),

            "backtracking_nodes":
                csp_results[
                    "backtracking_nodes"
                ],

            "backtracking_time":
                csp_results[
                    "backtracking_time"
                ],

            "backtracking_cost":
                csp_results[
                    "backtracking_cost"
                ],

            "min_conflicts_solution":
                str(
                    csp_results[
                        "min_conflicts_solution"
                    ]
                ),

            "min_conflicts_steps":
                csp_results[
                    "min_conflicts_steps"
                ],

            "min_conflicts_time":
                csp_results[
                    "min_conflicts_time"
                ],

            "min_conflicts_cost":
                csp_results[
                    "min_conflicts_cost"
                ]

        }]).to_csv(
            RESULTS_DIR
            / "csp_summary.csv",
            index=False
        )

    else:

        print(
            "\nNon sono state individuate "
            "abbastanza zone a rischio "
            "per costruire il CSP."
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "PROGETTO TERMINATO"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()