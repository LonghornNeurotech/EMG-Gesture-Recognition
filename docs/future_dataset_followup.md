# Future IMU Dataset Follow-Up

## Current-semester decision

The P002-P006 100 Hz recordings are accepted for the current prototype,
model-development, and showcase work.

P001/S01 remains under `data/pilot/50hz/` and is not part of the finalized
100 Hz dataset.

The QC analysis identified recordings that may benefit from future review,
but these findings do not currently create automatic trial-exclusion
requirements. The current ML pipeline may use all trials under `data/raw/`.

## Candidates for future review or recollection

If the dataset is expanded or used for higher-confidence evaluation, consider:

- Recollecting P001 at 100 Hz.
- Reviewing or recollecting P002 idle trial 1.
- Reviewing P003 forearm-raise trials.
- Reviewing P003 forearm-outward trial 8.
- Reviewing P004 forearm-outward trials.
- Collecting additional sessions to measure placement and session variability.

These are follow-up recommendations, not requirements for the current
semester prototype.

## Future collection improvements

- Provide an unrecorded practice repetition before each gesture.
- Demonstrate the intended movement plane, range, and endpoint.
- Ask participants to perform a smooth movement over most of the
  two-second recording period.
- Return to neutral only after recording stops.
- Record sensor position and orientation consistently.
- Preserve original recordings when collecting additional sessions.
- Evaluate any exclusion policy before adding it to the model-loading code.
