import csv
import json

import gymnasium as gym
import numpy as np

from q_learning import BASELINE, RESULTS


def evaluate(q_table, episodes=50, max_steps=500):
    env = gym.make("CliffWalking-v1")

    rewards = []
    steps = []
    successes = []
    falls = []

    for episode in range(episodes):
        state, _ = env.reset(seed=10000 + episode)

        total_reward = 0
        cliff_falls = 0
        success = False

        for step in range(1, max_steps + 1):
            # Greedy action selection: no random exploration
            action = int(np.argmax(q_table[state]))

            state, reward, terminated, truncated, _ = env.step(
                action
            )

            total_reward += reward

            if reward == -100:
                cliff_falls += 1

            if terminated:
                success = True

            if terminated or truncated:
                break

        rewards.append(total_reward)
        steps.append(step)
        successes.append(success)
        falls.append(cliff_falls)

    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(steps)),
        "mean_cliff_falls": float(np.mean(falls)),
    }


def main():
    rows = []

    paths = sorted(
        path for path in RESULTS.glob("*_q.npy")
        if path.stem != "best_q"
    )

    for path in paths:
        name = path.stem.removesuffix("_q")
        q_table = np.load(path)

        metrics = evaluate(q_table)

        training_rewards = np.load(
            RESULTS / f"{name}_rewards.npy"
        )

        row = {
            "agent": name,
            **metrics,
            "last_100_training_reward": float(
                training_rewards[-100:].mean()
            ),
        }
        rows.append(row)

        print(
            f"{name}: "
            f"reward={metrics['mean_reward']:.2f}, "
            f"success={metrics['success_rate']:.0%}, "
            f"steps={metrics['mean_steps']:.2f}, "
            f"falls={metrics['mean_cliff_falls']:.2f}"
        )

    # Prefer higher evaluation reward, then higher success,
    # then fewer steps. Break remaining ties using training reward.
    best = max(
        rows,
        key=lambda row: (
            row["mean_reward"],
            row["success_rate"],
            -row["mean_steps"],
            row["last_100_training_reward"],
        ),
    )

    best_name = best["agent"]
    best_q = np.load(RESULTS / f"{best_name}_q.npy")
    np.save(RESULTS / "best_q.npy", best_q)

    best_settings = BASELINE.copy()

    if best_name != "baseline":
        parameter, value = best_name.rsplit("_", 1)
        best_settings[parameter] = float(value)

    with open(RESULTS / "evaluation_summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(RESULTS / "best_agent.json", "w") as file:
        json.dump(
            {
                "agent": best_name,
                "settings": best_settings,
                "evaluation_episodes": 50,
                "results": best,
            },
            file,
            indent=2,
        )

    print("\nSelected agent:", best_name)
    print("Settings:", best_settings)


if __name__ == "__main__":
    main()