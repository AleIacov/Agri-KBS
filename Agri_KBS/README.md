# Agri-KBS

## Predizione e Pianificazione per l'Agricoltura di Precisione

Il progetto implementa un Knowledge-Based System (KBS)
per supportare il monitoraggio del rischio fitosanitario
e la pianificazione degli interventi agricoli.

La pipeline integra tre principali tematiche:

1. Rappresentazione della conoscenza mediante ontologie;
2. Apprendimento supervisionato;
3. Ragionamento con vincoli e ricerca locale.

## Architettura

Dati ambientali
        |
        v
Knowledge Base
        |
        v
Reasoner (HermiT)
        |
        v
Feature semantiche
        |
        v
Machine Learning
        |
        v
Zone a rischio
        |
        v
CSP
        |
        v
Piano di intervento

## Installazione

Installare le dipendenze:

pip install -r requirements.txt

È inoltre necessario disporre di un ambiente Java funzionante 
per l'esecuzione del reasoner HermiT integrato in Owlready2.

Verificare la corretta installazione di Java digitando nel terminale:

java -version

## Esecuzione

Dal terminale della directory principale eseguire il comando:

python main.py

## Generazione dei Grafici

Per garantire la totale trasparenza e riproducibilità dei risultati riportati nella documentazione, il progetto include uno script dedicato alla visualizzazione dei dati.

Lo script legge dinamicamente i risultati salvati nei file CSV (`ml_summary.csv` e `csp_summary.csv`) e genera automaticamente i grafici a barre con le relative deviazioni standard.

Per generare i grafici, dopo aver eseguito `main.py`, lanciare il seguente comando dal terminale:

python src/generate_graphs.py

## Output

I risultati dell'esecuzione vengono salvati nella directory:

results/

Tra gli output prodotti in formato CSV troverai:

- dataset_raw.csv (Dati grezzi generati)
- dataset_ontoBK.csv (Dataset arricchito con le deduzioni semantiche)
- ml_summary.csv (Risultati aggregati di Accuratezza, Precisione, Recall e F1)
- ml_fold_results.csv (Dettaglio di tutti i fold della Nested CV)
- operational_predictions.csv (Predizioni operative sulle nuove zone da analizzare)
- csp_summary.csv (Costi e prestazioni degli algoritmi Backtracking e Min-Conflicts)

L'ontologia popolata con gli individui e le inferenze viene salvata in:

ontology/agriculture.owl