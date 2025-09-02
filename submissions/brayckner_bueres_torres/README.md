# Computer Network Traffic Anomaly Detection using XGBoost

## What it does
Detects malicious network traffic patterns using XGBoost to handle binary classification, identifying potential DDoS attacks, port scans, and other network anomalies in real-time traffic data.

## How to run
- Have conda installed: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
- In the terminal cd into: `submissions/brayckner_bueres_torres/`
- Then run the command: `conda env create -f environment.yml`. This will install all the dependencies to run this project.
- Then run the command: `conda activate georgia-tech-ai-oa`
- Open `submissions/brayckner_bueres_torres/network_anomaly_detection.ipynb`
- Ensure the notebook is using the conda env `georgia-tech-ai-oa` 
- Run full notebook.

## HuggingFace Integration
- Uses HuggingFace `datasets` for data loading the `abmallick/network-traffic-anomaly` dataset for this project.