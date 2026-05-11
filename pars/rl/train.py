import os
import multiprocessing
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from tsp_env import TSPEnv
from data_loader import get_consistent_data


def make_env(matrix, goals):
    """
    Wrapper to create an environment instance for a subprocess.
    """

    def _init():
        # Each worker gets the exact same matrix object passed from the main process
        return TSPEnv(matrix, must_visit_nodes=goals)

    return _init


if __name__ == "__main__":
    # 1. Load data and config ONCE in the main process
    # Because this is called before SubprocVecEnv, 'matrix' is defined once
    matrix, goals, config = get_consistent_data()

    train_params = config['train_params']
    ppo_params = config['ppo_params']
    num_cpu = train_params.get('num_cpu', 8)

    # 2. Setup Vectorized Environment
    # This sends the 'matrix' we just generated to all 8 CPU workers
    print(f"--- Environment Setup ---")
    print(f"Goal Cities: {goals}")
    print(f"Launching {num_cpu} parallel training workers...")

    env = SubprocVecEnv([make_env(matrix, goals) for _ in range(num_cpu)])

    # 3. Configure Neural Network
    policy_kwargs = dict(
        net_arch=dict(
            pi=[config['network_arch']['num_neurons']] * config['network_arch']['num_layers'],
            vf=[config['network_arch']['num_neurons']] * config['network_arch']['num_layers']
        )
    )

    # 4. Initialize MaskablePPO
    # Checks config for tensorboard to avoid the ImportError
    use_tb = train_params.get('use_tensorboard', False)

    model = MaskablePPO(
        env=env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=train_params['log_dir'] if use_tb else None,
        **ppo_params
    )

    # 5. Training
    print(f"Starting training for {train_params['total_timesteps']} steps...")
    try:
        model.learn(total_timesteps=train_params['total_timesteps'])

        # 6. Save results
        model.save(train_params['save_path'])
        print(f"Success! Model saved as: {train_params['save_path']}")

    except KeyboardInterrupt:
        print("\nManual stop detected. Saving model before exit...")
        model.save(f"{train_params['save_path']}_interrupted")
    finally:
        env.close()