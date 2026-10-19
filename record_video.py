import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from q_learning import RESULTS


def main():
    q_table = np.load(RESULTS / "best_q.npy")

    env = gym.make(
        "CliffWalking-v1",
        render_mode="rgb_array",
    )

    state, _ = env.reset(seed=42)
    frames = [env.render().copy()]
    total_reward = 0
    success = False

    for step in range(1, 501):
        action = int(np.argmax(q_table[state]))

        state, reward, terminated, truncated, _ = env.step(
            action
        )

        total_reward += reward
        frames.append(env.render().copy())

        if terminated or truncated:
            success = terminated
            break

    env.close()

    # Pause briefly at the start and end.
    video_frames = (
        [frames[0]] * 5
        + frames
        + [frames[-1]] * 10
    )

    output = RESULTS / "trained_agent.mp4"

    imageio.mimsave(
        output,
        video_frames,
        fps=4,
        codec="libx264",
        macro_block_size=1,
    )

    print("Video saved to:", output)
    print("Reached goal:", success)
    print("Total reward:", total_reward)
    print("Steps:", step)


if __name__ == "__main__":
    main()