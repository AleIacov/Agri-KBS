
# REALIZZAZIONE GRAFICI: vengono letti automaticamente i file CSV generati nella cartella "results" e creaTI i relativi grafici.


import ast
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

# Definizione dei percorsi
OUTPUT_DIR = Path("results_graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("results")

def main():
    # Verifica che i CSV esistano prima di procedere
    ml_csv = RESULTS_DIR / "ml_summary.csv"
    csp_csv = RESULTS_DIR / "csp_summary.csv"

    if not ml_csv.exists() or not csp_csv.exists():
        print("Errore: Impossibile trovare i file CSV dei risultati.")
        print("Assicurati di aver eseguito prima 'main.py' per generare i dati.")
        return

    # LETTURA DATI ML DA CSV

    ml_df = pd.read_csv(ml_csv)
    
    # Preparo la struttura dati per i grafici
    configs = [
        ("Logistic Regression", "Raw", "Logistic Regression\nRaw"),
        ("Random Forest", "Raw", "Random Forest\nRaw"),
        ("Logistic Regression", "OntoBK", "Logistic Regression\nOntoBK"),
        ("Random Forest", "OntoBK", "Random Forest\nOntoBK")
    ]
    
    ml_records = []
    for model, dataset, label in configs:
        record = {"Configurazione": label}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            row = ml_df[(ml_df["model"] == model) & 
                        (ml_df["dataset"] == dataset) & 
                        (ml_df["metric"] == metric)]
            
            if not row.empty:
                record[metric.capitalize()] = row["mean"].values[0]
                record[f"{metric.capitalize()}_std"] = row["std"].values[0]
            else:
                record[metric.capitalize()] = 0.0
                record[f"{metric.capitalize()}_std"] = 0.0
                
        ml_records.append(record)

    ml = pd.DataFrame(ml_records)

    # GRAFICO 1 - Confronto dei quattro modelli/scenari

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(ml))
    width = 0.18

    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        ax.bar(
            x + (i - 1.5) * width,
            ml[metric],
            width,
            yerr=ml[f"{metric}_std"],
            capsize=3,
            label=metric,
            zorder=3  
        )

    ax.set_title("Confronto delle prestazioni dei modelli")
    ax.set_ylabel("Valore medio della metrica")
    ax.set_xticks(x)
    ax.set_xticklabels(ml["Configurazione"])
    ax.set_ylim(0.65, 1.00) 
    
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.grid(axis='y', which='major', color='gray', linestyle='-', linewidth=0.6, alpha=0.7, zorder=0)
    ax.grid(axis='y', which='minor', color='gray', linestyle='--', linewidth=0.3, alpha=0.5, zorder=0)

    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_confronto_prestazioni_ML.png", dpi=200)
    plt.close(fig)


    # GRAFICO 2 - Effetto della Knowledge Base sulla Random Forest

    rf = ml.iloc[[1, 3]].copy()
    rf["Scenario"] = ["Raw", "OntoBK"]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(rf))
    width = 0.18

    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        ax.bar(
            x + (i - 1.5) * width,
            rf[metric],
            width,
            yerr=rf[f"{metric}_std"],
            capsize=3,
            label=metric,
            zorder=3
        )

    ax.set_title("Effetto della Background Knowledge sulla Random Forest")
    ax.set_ylabel("Valore medio della metrica")
    ax.set_xticks(x)
    ax.set_xticklabels(rf["Scenario"])
    ax.set_ylim(0.65, 1.00) 
    
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.grid(axis='y', which='major', color='gray', linestyle='-', linewidth=0.6, alpha=0.7, zorder=0)
    ax.grid(axis='y', which='minor', color='gray', linestyle='--', linewidth=0.3, alpha=0.5, zorder=0)

    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_effetto_KB_Random_Forest.png", dpi=200)
    plt.close(fig)


    # LETTURA DATI CSP DA CSV E GRAFICI 3, 4, 5

    csp_df = pd.read_csv(csp_csv)
    csp_row = csp_df.iloc[0]

    csp_methods = ["Backtracking + Forward Checking", "Min-Conflicts"]
    csp_costs = [csp_row["backtracking_cost"], csp_row["min_conflicts_cost"]]
    csp_times = [csp_row["backtracking_time"], csp_row["min_conflicts_time"]]

    bt_sol_str = csp_row["backtracking_solution"]
    mc_sol_str = csp_row["min_conflicts_solution"]
    
    bt_sol = ast.literal_eval(bt_sol_str) if isinstance(bt_sol_str, str) else bt_sol_str
    mc_sol = ast.literal_eval(mc_sol_str) if isinstance(mc_sol_str, str) else mc_sol_str
    
    zones = list(bt_sol.keys())
    bt_slots = [bt_sol[z] for z in zones]
    mc_slots = [mc_sol[z] for z in zones]

    # --- GRAFICO 3 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(csp_methods, csp_costs, zorder=3)
    ax.set_title("Confronto del costo delle soluzioni CSP")
    ax.set_ylabel("Costo della soluzione")
    ax.set_ylim(0, max(csp_costs) + 5)
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

    for bar, value in zip(bars, csp_costs):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_costo_soluzioni_CSP.png", dpi=200)
    plt.close(fig)

    # --- GRAFICO 4 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(csp_methods, csp_times, zorder=3)
    ax.set_title("Confronto del tempo di esecuzione del CSP")
    ax.set_ylabel("Tempo (secondi)")
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

    for bar, value in zip(bars, csp_times):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(csp_times) * 0.03, f"{value:.6f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_tempo_esecuzione_CSP.png", dpi=200)
    plt.close(fig)

    # --- GRAFICO 5 ---
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(zones))
    width = 0.36

    b1 = ax.bar(x - width / 2, bt_slots, width, label="Backtracking + Forward Checking", zorder=3)
    b2 = ax.bar(x + width / 2, mc_slots, width, label="Min-Conflicts", zorder=3)

    ax.set_title("Slot temporali assegnati alle zone a rischio")
    ax.set_ylabel("Slot temporale (ora)")
    ax.set_xticks(x)
    ax.set_xticklabels(zones)
    ax.set_yticks([8, 10, 12, 14, 16, 18])
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    ax.legend()

    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{int(value)}", ha="center")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_piano_intervento_CSP.png", dpi=200)
    plt.close(fig)

    print("Grafici aggiornati dinamicamente e salvati nella cartella:", OUTPUT_DIR)

if __name__ == "__main__":
    main()