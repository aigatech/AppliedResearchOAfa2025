# Network Traffic Classification Model for Anomaly Detection using XGBoost

## What it does
Implements a supervised learning approach using XGBoost to perform binary classification on network traffic data. The model detects malicious patterns in real time, such as potential DDoS attacks, port scans, and other network anomalies, enabling proactive network defense.

## How to run
- Have conda installed: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
- In the terminal cd into: `submissions/brayckner_bueres_torres/`
- Then run the command: `conda env create -f environment.yml`. This will install all the dependencies to run this project.
- Then run the command: `conda activate georgia-tech-ai-oa`
- Open `submissions/brayckner_bueres_torres/network_anomaly_detection.ipynb`
- Ensure the notebook is using the conda env `georgia-tech-ai-oa` 
- Run full notebook.

## Note: For readability within Github, I included the Jupyter notebook file converted to a python script
File name: `submissions/brayckner_bueres_torres/anom_detection_script.py`

## HuggingFace Integration
- Uses HuggingFace `datasets` for data loading the `abmallick/network-traffic-anomaly` dataset for this project.