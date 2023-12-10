# IntrusionDetectorsTraining

**Thesis repository: Nicolò Polazzi, "Addestramento di rilevatori di intrusioni tramite generazione di attacchi: approcci dallo stato dell’arte",  Computer Science Bachelor's Degree, AY 2022-2023. Supervisor: Andrea Ceccarelli.**

## Objective and Method

The objective is to generate the effects of zero-day attacks to enrich a dataset composed of normal data. Theoretically, if we are able to generate attack effects that cover the entire input space, zero-days at test time will lead to a system behaviour that has already been observed at training time. Clearly, if the above is achieved, the dangerousness of unknown attacks is greatly reduced, because their effect is equivalent to the effect of known attacks.

The basic assumption is that attacks have a visible effect on a target system, for example, they can be categorized as anomalies by examining collected data. This assumption constitutes the foundation of all work in the field of anomaly detection.
In general, the approach is to model normal data on a hyperplane to discovery the variety of normal data and, from this, identify data points outside the variety. Each anomaly will be described by these data points: attacks, whether known or unknown, are characterized by these data points. Consequently, if the data points within the variety of normal and anomalous data can be identified, we can train an Intrusion Detection System in a supervised environment without the risk of unknown attacks.

## Installation instructions

In order to reproduce the experiments, it is necessary to install the following components:

- [ALAD](https://github.com/houssamzenati/Adversarially-Learned-Anomaly-Detection);
- [ARN](https://github.com/arnwg/arn);
- CICIDS18 and ADFANet datasets;
- conda environment _attacchi.yml_ provided in this GitHub repository.

## Configuration

It is necessary to set the correct PATHs. These should be directories where you have read and write permissions.

## Execution

All commands must be executed in the **attacchi** environment using the following command:`conda activate attacchi`.
I recommend you to proceed in the following order:

1. ALAD(eGAN) execution.
   
   Move to the folder where ALAD is installed, where you can run ALAD on datasets ADFANet and CICIDS18. E.g., you may use the following commands:

```bash
python3 main.py gan adfa run --nb_epochs=50 --label=1 --w=0.1 --m='cross-e' --d=2 --rd=42
```
```bash
python3 main.py gan cicids run --nb_epochs=50 --label=1 --w=0.1 --m='cross-e' --d=2 --rd=42
```

2. ARN execution.
   
   It is sufficient to execute *ADFA-ARN_ADFA_REV_Generation.ipynb* and *ADFA-ARN_CICIDS_REV_Generation.ipynb* notebooks.


4. Collect the results.
   
   Run  *zero-day attack generation-ADFA.ipynb* and *zero-day attack generation-CICIDS.ipynb* notebooks.

## Output

The execution of ALAD and ARN produces a numpy file containing the generated attacks.

All algorithms write the results to the files *adfa_competitors.csv* e *cicids_competitors.csv*.
