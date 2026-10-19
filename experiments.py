import csv

import matplotlib.pyplot as plt
import numpy as np

from q_learning import BASELINE, RESULTS, moving_average, train


def compare_parameter(parameter, values, title, filename):
    summary = []

    plt.figure(figsize=(10, 5))

    for value in values:
        # Change only the parameter being tested.
        settings = BASELINE.copy()
        settings[parameter] = value

        q_table, rewards = train(**settings)

        prefix = f"{parameter}_{value}"
        np.save(RESULTS / f"{prefix}_q.npy", q_table)
        np.save(RESULTS / f"{prefix}_rewards.npy", rewards)

        smoothed = moving_average(rewards)

        plt.plot(
            np.arange(100, len(rewards) + 1),
            smoothed,
            label=f"{parameter} = {value}",
        )

        first_100 = float(rewards[:100].mean())
        last_100 = float(rewards[-100:].mean())

        summary.append({
            parameter: value,
            "first_100_mean_reward": first_100,
            "last_100_mean_reward": last_100,
        })

        print(
            f"{parameter}={value}: "
            f"first 100 = {first_100:.2f}, "
            f"last 100 = {last_100:.2f}"
        )

    plt.title(title)
    plt.xlabel("Training episode")
    plt.ylabel("Reward (100-episode average)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / f"{filename}.png", dpi=200)
    plt.close()

    with open(
        RESULTS / f"{filename}_summary.csv", "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    compare_parameter(
        parameter="alpha",
        values=[0.1, 0.5, 0.9],
        title="Effect of Learning Rate on Q-learning",
        filename="learning_rate_comparison",
    )

    compare_parameter(
        parameter="gamma",
        values=[0.50, 0.90, 0.99],
        title="Effect of Discount Factor on Q-learning",
        filename="discount_factor_comparison",
    )

    compare_parameter(
        parameter="epsilon_decay",
        values=[0.99, 0.995, 0.999],
        title="Effect of Exploration Decay on Q-learning",
        filename="exploration_decay_comparison",
    )