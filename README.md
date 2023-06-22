# IntrusionDetectorsTraining

**Repository  della tesi: Nicolò Polazzi, "Addestramento di rilevatori di intrusioni tramite generazione di attacchi: approcci dallo stato dell’arte", Corso di Laurea in Informatica, AA. 2022-2023. Relatore: Andrea Ceccarelli.**

## Obiettivo e approccio

Lo scopo è quello di generare l'effetto degli attacchi zero-day, in moda da arricchire un dataset composto da dati normali. In teoria, se riusciamo a modellare l'effetto degli attacchi, allora la presenza degli zero-day nel momento di test porterà a un comportamento del sistema che  è stato già osservato nell'attimo del training. Nel caso in cui si riuscisse ad ottenre quanto detto prima, la pericolosità degli attacchi sconosciuti sarebbe altamente ridotta, perché i loro effetti sono equivalenti a quelli di attacchi conosciuti.

L'assunzione di base è che gli attacchi hanno un effetto visibile su un sistema target, per esempio è possibile catalogarli come anomalie guardando i dati raccolti. Questa assunzione è la base di ogni lavoro nel settore dell'anomaly detection.
In generale, l'approccio è quello di modellare i dati normali in un iperpiano per scoprire la varietà dei dati normali, e, da questa, identificare i data points fuori dalla varietà. Ogni anomalia sarà descritta da questi data points: attacchi, conosciuti o non, sono descritti da questi data points.
Di conseguenza, se i data points nella varietà dei dati normali e anomali possono essere identificate, possiamo allenare un Intrusion Detection System in un ambiente supervisionato senza il rischio di attacchi sconosciuti.

## Installazione

Per essere in grado di riprodurre gli esperimenti è necessario installare le seguenti componenti:

- [ALAD](https://github.com/houssamzenati/Adversarially-Learned-Anomaly-Detection);
- [ARN](https://github.com/arnwg/arn);
- i dataset CICIDS18 e ADFANet;
- l'enviroment conda attacchi.yml fornito in questo github.

## Configurazione

E' necessario impostare i PATH corretti; questi dovrebbero essere directories dove avete i permessi di lettura e scrittura.

## Esecuzione

Tutti i comandi devono essere eseguiti nell'ambiente **attacchi** tramite il comando `conda activate attacchi`.
Consiglio di procedere nell' ordine seguente:

1. Esecuzione di ALAD(eGAN): spostarsi nella cartella dove si è installato ALAD dove si potrà eseguire ALAD sui dataset ADFANet e CICIDS18, per esempio, con i comandi:
```bash
python3 main.py gan adfa run --nb_epochs=50 --label=1 --w=0.1 --m='cross-e' --d=2 --rd=42
```
```bash
python3 main.py gan cicids run --nb_epochs=50 --label=1 --w=0.1 --m='cross-e' --d=2 --rd=42
```

2. Esecuzione di ARN: è sufficiente eseguire i notebook *ADFA-ARN_ADFA_REV_Generation.ipynb* e *ADFA-ARN_CICIDS_REV_Generation.ipynb*.


3. Eseguire i notebook *zero-day attack generation-ADFA.ipynb* e *zero-day attack generation-CICIDS.ipynb*.

L'esecuzione di ALAD e ARN produce un file numpy contenente gli attacchi generati.

Tutti gli algoritmi scrivono i risultati nei file *adfa_competitors.csv* e *cicids_competitors.csv*.