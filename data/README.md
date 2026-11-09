# IMU Gesture Dataset

This directory contains IMU recordings collected using the MPU-6050 for
gesture-recognition development.

## Directory structure

- `raw/`: Finalized sessions eligible for model training and evaluation
- `pilot/`: Preliminary or nonconforming sessions excluded from normal models
- `processed/`: Generated windows or other derived data

## Finalized collection configuration

- Sampling rate: 100 Hz
- Trial duration: 2 seconds
- Expected samples per complete trial: 200
- Trials per class per session: 10
- Active gesture classes: 6
- Additional class: idle
- Accelerometer range: ±2g
- Gyroscope range: ±250 degrees/second
- Acceleration units: m/s², including gravity
- Gyroscope units: degrees/second

## CSV columns

- `participant_id`
- `session_id`
- `gesture`
- `trial_id`
- `timestamp_ns`
- `sequence_number`
- `ax`, `ay`, `az`
- `gx`, `gy`, `gz`

## Dataset inclusion policy

Collection scripts and model-training code should read from `data/raw/` by
default.

Data under `data/pilot/` must not be included in final training, validation, or
testing unless a documented experiment explicitly requests it.

P001/S01 was collected at approximately 50 Hz and is retained under
`data/pilot/` for visualization and pipeline testing. P001 must be recollected
at 100 Hz before being included in the finalized dataset.

## Privacy

Participant IDs must be pseudonymous. Do not store participant names or other
identifying information in this repository.