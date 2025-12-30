# Multimodal Insider Threat Detection System

This project implements an early-warning insider threat detection engine using multimodal enterprise telemetry from the CERT dataset.

## What it does
The system predicts whether a user will become an insider threat up to 7 days before data exfiltration, using:
- Login behavior
- File access
- USB usage
- Web uploads
- Email activity
- Temporal behavior modeling (LSTM)

## Model
A 7-day LSTM sequence model learns behavioral drift and attack build-up.

## Performance
On 1.39M user-days:
- Precision (insider): ~94%
- Recall (insider): ~99%
- F1-score: ~97%

## Run
pip install -r requirements.txt
python train.py


## Dataset
Models were trained on enterprise-scale CERT logs (~90GB).  
For reproducibility, we provide a compact extracted feature set.

This project demonstrates how modern SOC platforms detect malicious insiders before data theft occurs.
