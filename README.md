# Budget-Constrained SEIR Recalibration

This course project studies when a forecasting surrogate should be recalibrated when only six oracle updates are available. The true process is a stochastic SEIR simulator; the surrogate is deterministic and receives the current true transmission rate only when a policy selects `RECALIBRATE`.

## Experiments

The single experiment entry point compares DQN, error thresholds `0.003` and `0.005`, and a fixed schedule.

- **Fixed trajectory:** the existing 160-day slow-rise, sudden-drop, sudden-rise, slow-decrease beta trajectory. The fixed baseline recalibrates on days `40, 60, 80, 100, 120, 140`.
- **Random trajectory:** four beta changes are generated independently for every episode. Episodes end at epidemic extinction or the 240-day safety cap. The fixed baseline recalibrates on days `30, 60, 90, 120, 150, 180`.

Random event gaps are uniform integers from 2 through 30 days. Event duration first selects a short or long group with probability 0.5 each. Short durations use a truncated normal with mean 3 and standard deviation 1.22 on `[1, 10]`; long durations use mean 25 and standard deviation 7.4 on `[11, 40]`. Samples are rounded to integer days. New beta values are uniform on `[0.15, 0.85]` and differ from the previous target by at least 0.10.

## Training and testing

DQN uses three training seeds (`123, 456, 789`), 500 training episodes per seed, and no validation split. Testing uses 100 paired episodes. Every policy sees the same epidemic seeds and, for the random tier, the same held-out trajectory seeds.

An action enters replay only after its complete seven-day forecast path is observed. The training error is:

```text
0.3 * one-step normalized SEIR MAE + 0.7 * complete seven-day path normalized SEIR MAE
```

## Running

```bash
python3 -m pip install -r requirements.txt
python3 -m experiments.run_course_experiments
```

Run one tier or a small smoke experiment:

```bash
python3 -m experiments.run_course_experiments --tiers fixed
python3 -m experiments.run_course_experiments --training-episodes 2 --evaluation-episodes 2 --training-seeds 123 --output-dir results/smoke
```

## Outputs

Each tier writes to `results/course_project/<tier>/`:

- model checkpoints and per-seed training logs
- training and test trajectory manifests
- step-level test trajectories
- aligned seven-day `S/E/I/R` forecast paths
- episode, aggregate, and paired-difference summaries
- prediction, beta tracking, SEIR trajectory, budget, and recalibration-timing figures

For variable-length random episodes, average daily one-step error, mean complete-path error, and mean mixed error are the primary metrics. Cumulative error is retained together with episode length.
