# Budget-Constrained Recalibration RL Prototype

This code studies when to recalibrate a surrogate forecasting model under a finite computational budget. There are no interventions and the objective is forecast accuracy only.

This branch treats each simulation as one epidemic episode. An episode begins with introduced infections and ends when active infections fall below an extinction threshold for several consecutive days. `max_steps` is only a safety cap. The previous fixed-horizon behavior can still be selected with `termination_mode: fixed_horizon`.

## Model

The true simulator is a stochastic discrete-time SEIR model:

```text
S -> E -> I -> R
```

Daily transitions are:

```text
new_exposed    ~ Binomial(S, 1 - exp(-beta_t * I / N))
new_infectious ~ Binomial(E, sigma)
new_recovered  ~ Binomial(I, gamma)
```

The transmission rate has a surge period:

```text
beta_t = beta_surge during surge_start <= t <= surge_end
beta_t = beta_base otherwise
```

The surrogate is a deterministic SEIR model with `beta_hat`, `sigma_hat`, and `gamma_hat`. In the initial oracle-style recalibration setting, recalibration sets `beta_hat` to the true current beta before the true simulator advances. This avoids future-state leakage.

Default simulator settings:

```text
population = 100000
initial_exposed = 20
initial_infectious = 10
beta_base = 0.30
beta_surge = 0.65
sigma = 0.1923
gamma = 0.1429
episode_length = 120 days for the beta schedule reference
surge_start = day 45
surge_end = day 75
```

Default surrogate settings:

```text
beta_hat = 0.30
sigma_hat = 0.1923
gamma_hat = 0.1429
fit_window = 14 days
```

## Environment

The environment is Gym-like but does not depend on Gymnasium.

Actions:

```text
0 = WAIT
1 = RECALIBRATE
```

The reward is negative forecast error:

```text
reward = -error
```

The default error metric is normalized one-step MAE over `S,E,I,R`. The config also supports `mse` and `infectious_mse`.

Default state features use the `D_raw_mean_trend` setting: normalized `S,E,I,R`, the last `K` forecast errors, rolling mean forecast error, rolling error trend, remaining budget fraction, and days since recalibration scaled by `max_steps`. Here `K = error_history_length`, which defaults to 14 days and can be changed in `configs/default.yaml`.

State-choice experiments are documented in [docs/state_choice_experiments.md](docs/state_choice_experiments.md), and a full project handoff is available in [docs/project_handoff.md](docs/project_handoff.md). The focused confirmation favored `D_raw_mean_trend` for robustness; the implementation default uses the nearby two-week setting `K=14`, while `A_raw_errors, K=10` remains the strongest simple alternative.

Default environment settings:

```text
termination_mode = epidemic
horizon = 120 days if fixed_horizon mode is selected
max_steps = 365
extinction_threshold = 1
extinction_patience = 7
min_steps_before_extinction_check = 14
forecast_horizon = 1 day
recalibration_budget = 8
error_metric = mae
error_history_length = 14
state_features = S/N, E/N, I/N, R/N, last 14 errors, rolling_mean_error, rolling_error_trend, remaining_budget, days_since_recalibration
```

Active infections are defined as:

```text
active_t = E_t + I_t
```

In epidemic mode, an episode terminates when:

```text
active_t <= extinction_threshold
for extinction_patience consecutive days
and t >= min_steps_before_extinction_check
```

The episode also terminates if:

```text
t >= max_steps
```

The termination reason is logged as `extinction`, `max_steps`, or `fixed_horizon`.

At each environment step:

1. The policy observes the current state vector.
2. The policy chooses `WAIT` or `RECALIBRATE`.
3. If recalibration is chosen and budget remains, the surrogate updates `beta_hat` from the current true beta.
4. The surrogate forecasts from the current true state.
5. The true stochastic simulator advances one day.
6. Forecast error is computed against the realized next true state.
7. The environment updates rolling error history and previous active/infectious counts for optional features.
8. The environment checks extinction or safety-cap termination.
9. The environment returns next state, reward, done flag, and diagnostics.

## Methods

- Fixed schedule: recalibrate every `k` days if budget remains.
- Threshold: recalibrate when recent forecast error exceeds a threshold if budget remains.
- DQN: learn a binary wait/recalibrate policy with replay buffer, target network, epsilon-greedy exploration, and an MLP Q-network.

### Fixed Schedule Policy

`FixedSchedulePolicy(interval, start_day=0, cooldown=0)` recalibrates every `interval` days as long as budget remains. It does not look at forecast error, beta, or epidemic state. This makes it a simple calendar-based baseline.

Default evaluation intervals:

```text
7, 14, 21, 28 days
```

### Threshold Policy

`ThresholdPolicy(error_threshold, use_trend=False, trend_threshold=None, cooldown=0)` recalibrates when rolling forecast error exceeds `error_threshold`, as long as budget remains.

If `use_trend=True`, the policy can also recalibrate when the recent error trend increases by more than `trend_threshold`. The default experiments use the direct error threshold only.

Default evaluation thresholds:

```text
0.001, 0.002, 0.003, 0.005
```

### DQN Policy

The DQN agent approximates action-values for the two actions `WAIT` and `RECALIBRATE`.

Network:

```text
input_dim = length of environment state vector
hidden layers = [64, 64]
output_dim = 2
```

Training details:

```text
episodes = 250
batch_size = 64
replay_capacity = 20000
gamma = 0.98
learning_rate = 0.001
min_replay_size = 500
target_update_interval = 200 steps
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 8000
optimizer = Adam
loss = Huber loss
```

DQN training loop:

1. Reset the environment with a deterministic episode seed.
2. Select an action with epsilon-greedy exploration.
3. Step the environment.
4. Store `(state, action, reward, next_state, done)` in replay memory.
5. Update the Q-network from a sampled replay batch when enough data are available.
6. Periodically copy online network weights into the target network.
7. Save the trained checkpoint and episode-level training logs.

### Evaluation Protocol

Policy evaluation runs multiple random seeds and logs every variable-length epidemic episode. The default evaluation uses 50 episodes per policy. Summary tables report mean and standard deviation across seeds.

Because episodes can have different lengths, the code reports both:

```text
cumulative_forecast_error = sum of daily errors over the episode
average_daily_forecast_error = cumulative error / episode length
```

Cumulative error measures total episode burden. Average daily error normalizes for different outbreak durations.

Budget sweep evaluates budgets:

```text
0, 1, 3, 5, 10, 20
```

Robustness experiments vary:

```text
beta_base: lower and higher baseline transmission
surge timing: baseline, early surge, late surge
surge magnitude: mild surge, severe surge
initial conditions: low and high initial exposed/infectious counts
```

Observation-noise controls are not implemented yet.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Commands

Train DQN:

```bash
python3 -m experiments.train_dqn --config configs/default.yaml
```

Evaluate fixed schedule, threshold, and DQN policies:

```bash
python3 -m experiments.evaluate_policies --config configs/default.yaml --model results/dqn_model.pt
```

Run budget sweep:

```bash
python3 -m experiments.run_budget_sweep --config configs/default.yaml --model results/dqn_model.pt
```

Run error-history sweep:

```bash
python3 -m experiments.run_error_history_sweep --config configs/default.yaml --history-lengths 1 3 5 7 9 11 13 --seeds 123 456 789
```

Run error-feature ablation:

```bash
python3 -m experiments.run_error_feature_ablation --config configs/default.yaml --history-lengths 7 11 --episodes 250
```

Run robustness experiment:

```bash
python3 -m experiments.run_robustness --config configs/default.yaml --model results/dqn_model.pt --seeds 101 202 303
```

## Outputs

CSV results are saved under `results/`.

- `dqn_training_log.csv`
- `evaluation_summary.csv`
- `episode_logs/*.csv`
- `policy_trajectories.csv`
- `policy_episode_summary.csv`
- `policy_comparison.csv`
- `budget_sweep_trajectories.csv`
- `budget_sweep_summary.csv`
- `error_history_sweep_detailed.csv`
- `error_history_sweep_summary.csv`
- `error_history_sweep_new3_detailed.csv`
- `error_history_sweep_new3_summary.csv`
- `error_history_sweep_six_seed_detailed.csv`
- `error_history_sweep_six_seed_summary.csv`
- `error_feature_ablation_detailed.csv`
- `error_feature_ablation_summary.csv`
- `robustness_trajectories.csv`
- `robustness_summary.csv`
- `robustness_comparison.csv`

Diagnostic PNG figures are saved under `results/figures_minimal_state/` by default.

- `true_seir_curves.png`
- `true_beta_vs_surrogate_beta.png`
- `forecast_error_over_time.png`
- `policy_error_comparison.png`
- `cumulative_error_by_policy.png`
- `budget_usage_by_policy.png`
- `dqn_training_reward.png`
- `dqn_training_loss.png`
- `dqn_epsilon.png`
- `budget_sweep_cumulative_error.png`
- `budget_sweep_budget_used.png`
- `error_history_sweep_cumulative_error.png`
- `error_history_sweep_new3_cumulative_error.png`
- `error_history_sweep_six_seed_cumulative_error.png`
- `error_feature_ablation_cumulative_error.png`
- `robustness_cumulative_error.png`
- `episode_length_distribution.png`
- `average_daily_error_by_policy.png`
- `normalized_recalibration_timing.png`
- `representative_episode_diagnostics.png`

The normalized recalibration timing figure marks the beta-surge window with horizontal reference lines and light shading. The representative episode diagnostics figure keeps only `dqn`, `threshold_0.003`, and `fixed_schedule_14` so recalibration behavior can be compared against the known synthetic regime change without an overcrowded legend.

The current default config writes figures to `results/figures_minimal_state/`.

The main reported quantities are cumulative forecast error, average daily forecast error, episode length, budget allowed, budget used, budget utilization, recalibration counts/times, termination reason, final active infections, final `S,E,I,R`, final epidemic size, and robustness summaries across seeds or scenarios.

## Project Structure

```text
configs/default.yaml
envs/recalibration_env.py
envs/true_simulator.py
envs/surrogate_model.py
agents/dqn_agent.py
agents/replay_buffer.py
agents/networks.py
baselines/fixed_schedule.py
baselines/threshold.py
experiments/train_dqn.py
experiments/evaluate_policies.py
experiments/run_budget_sweep.py
experiments/run_robustness.py
diagnostics/plots.py
diagnostics/metrics.py
utils/seed.py
utils/config.py
```

## Reproducibility

Random seeds are set for Python, NumPy, and PyTorch through `utils/seed.py`. The simulator receives deterministic per-episode seeds during training and evaluation.

## Code Quality Notes

The code is intentionally small and modular. It is runnable from the command line, uses short comments where helpful, and avoids dependencies on the original Hierarchical-RL application logic. Generated CSV, PNG, and checkpoint files are ignored by git.
