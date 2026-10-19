# CliffWalking Q-learning

Kaylie Skinner  
CPSC 4420 — Assignment 2

## Overview

This project implements tabular Q-learning for Gymnasium's
CliffWalking-v1 environment. The agent learns to reach the goal
while avoiding the cliff.

Experiments compare learning rate, discount factor, and
exploration decay while changing one parameter at a time.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

Run these commands from the project folder, in order:

```bash
python q_learning.py
python experiments.py
python evaluate.py
python record_video.py
```

1. Train the baseline and save its reward plot.
2. Run all three hyperparameter experiments.
3. Evaluate saved agents over 50 episodes with exploration disabled.
4. Record the selected agent as an MP4 video.

Generated files are saved in `results/`.

## Baseline Configuration

| Parameter | Value |
|---|---:|
| Learning rate (alpha) | 0.1 |
| Discount factor (gamma) | 0.99 |
| Initial epsilon | 1.0 |
| Epsilon decay per episode | 0.995 |
| Minimum epsilon | 0.05 |
| Training episodes | 3000 |
| Maximum steps per episode | 500 |
| Training random seed | 42 |

Q-values start at zero. Training uses epsilon-greedy action
selection, with random tie-breaking among maximizing actions.
Evaluation uses a greedy policy with deterministic tie-breaking.

## Experiments

Each setting is trained from a fresh Q-table using seed 42.
All settings other than the tested parameter remain at baseline.

| Experiment | Values |
|---|---|
| Learning rate | 0.1, 0.5, 0.9 |
| Discount factor | 0.50, 0.90, 0.99 |
| Exploration decay | 0.99, 0.995, 0.999 |

Comparison plots show a 100-episode moving average of training
reward. Higher rewards (less negative values) are better.

## Results

Baseline average training reward improved from -2593.54 in the
first 100 episodes to -30.58 in the last 100 episodes.

All tested agents achieved the following greedy evaluation results:

- Mean reward: -13
- Success rate: 100% over 50 episodes
- Mean steps: 13
- Mean cliff falls: 0

The selected agent used alpha=0.1, gamma=0.9, and
epsilon_decay=0.995. Evaluation results tied, so the final
100-episode training average was used to break the tie.
The selected agent matched the baseline's evaluation performance.

The environment and greedy policy are deterministic, so evaluation
episodes repeat the same route. Experiments use one training seed;
small training differences should not be treated as definitive
evidence that one configuration is generally superior.

## Output Files

- `baseline_training.png`: baseline training rewards
- `learning_rate_comparison.png`: learning-rate comparison
- `discount_factor_comparison.png`: discount-factor comparison
- `exploration_decay_comparison.png`: exploration-decay comparison
- `evaluation_summary.csv`: evaluation metrics for all agents
- `best_agent.json`: selected configuration and results
- `trained_agent.mp4`: video of the selected policy
- `.npy` files: saved Q-tables and training rewards

## Environment Reference

https://gymnasium.farama.org/environments/toy_text/cliff_walking/