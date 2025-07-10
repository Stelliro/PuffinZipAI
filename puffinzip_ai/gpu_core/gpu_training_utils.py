import logging
import numpy as np

CUPY_AVAILABLE = False
cp = None
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger("PuffinZipAI_GPU_TrainingUtils")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def batch_update_q_table_gpu(q_table_gpu,
                             batch_states_cpu,
                             batch_actions_cpu,
                             batch_rewards_cpu,
                             learning_rate: float,
                             discount_factor: float,
                             gpu_id: int = 0):
    if not CUPY_AVAILABLE or q_table_gpu is None or not isinstance(q_table_gpu, cp.ndarray):
        logger.debug(
            "batch_update_q_table_gpu: CuPy not available or q_table_gpu invalid. Update skipped/deferred to CPU.")
        return False

    batch_size = len(batch_states_cpu)
    if batch_size == 0:
        logger.debug("batch_update_q_table_gpu: Empty batch. Nothing to update.")
        return True

    try:
        with cp.cuda.Device(gpu_id):
            states_np = np.asarray(batch_states_cpu, dtype=np.int32)
            actions_np = np.asarray(batch_actions_cpu, dtype=np.int32)
            rewards_np = np.asarray(batch_rewards_cpu, dtype=q_table_gpu.dtype)

            states_gpu = cp.asarray(states_np)
            actions_gpu = cp.asarray(actions_np)
            rewards_gpu = cp.asarray(rewards_np)

            current_q_values_gpu = q_table_gpu[states_gpu, actions_gpu]

            td_error_gpu = rewards_gpu - current_q_values_gpu
            new_q_values_gpu = current_q_values_gpu + learning_rate * td_error_gpu

            q_table_gpu[states_gpu, actions_gpu] = new_q_values_gpu

        logger.debug(f"Batch Q-table update on GPU {gpu_id} completed for {batch_size} experiences.")
        return True

    except Exception as e:
        logger.error(f"Error during batch Q-table update on GPU {gpu_id}: {e}", exc_info=True)
        return False


def get_batch_actions_gpu(q_table_gpu,
                          batch_states_cpu,
                          exploration_rate: float,
                          action_space_size: int,
                          gpu_id: int = 0):
    if not CUPY_AVAILABLE or q_table_gpu is None or not isinstance(q_table_gpu, cp.ndarray):
        logger.debug("get_batch_actions_gpu: CuPy not available or q_table_gpu invalid. Cannot get actions.")
        return None

    batch_size = len(batch_states_cpu)
    if batch_size == 0:
        return np.array([], dtype=np.int32)

    try:
        with cp.cuda.Device(gpu_id):
            states_gpu = cp.asarray(np.asarray(batch_states_cpu, dtype=np.int32))

            q_values_for_batch_states_gpu = q_table_gpu[states_gpu, :]

            greedy_actions_gpu = cp.argmax(q_values_for_batch_states_gpu, axis=1)

            random_actions_cpu = np.random.randint(0, action_space_size, size=batch_size, dtype=np.int32)
            exploration_decisions_cpu = np.random.rand(batch_size) < exploration_rate

            exploration_decisions_gpu = cp.asarray(exploration_decisions_cpu)
            random_actions_gpu = cp.asarray(random_actions_cpu)

            chosen_actions_gpu = cp.where(exploration_decisions_gpu, random_actions_gpu, greedy_actions_gpu)

            chosen_actions_cpu = cp.asnumpy(chosen_actions_gpu)

        logger.debug(
            f"Batch actions ({batch_size}) retrieved using GPU Q-table (exploration rate: {exploration_rate:.3f}).")
        return chosen_actions_cpu.astype(np.int32)

    except Exception as e:
        logger.error(f"Error during batch action selection on GPU {gpu_id}: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    print("\n--- Testing GPU Training Utilities ---")

    if CUPY_AVAILABLE:
        print("\n--- Simulating GPU Batch Q-Table Update ---")
        state_space_size_test = 100
        action_space_size_test = 3
        batch_size_test = 10
        lr_test = 0.1
        gamma_test = 0.9
        gpu_id_test = 0

        try:
            num_gpus_test = cp.cuda.runtime.getDeviceCount()
            if num_gpus_test == 0:
                print("No CUDA devices for test. Skipping CuPy dependent parts.")
                CUPY_AVAILABLE = False
            else:
                gpu_id_test = 0 if gpu_id_test >= num_gpus_test else gpu_id_test
                with cp.cuda.Device(gpu_id_test):
                    q_table_gpu_test = cp.random.rand(state_space_size_test, action_space_size_test,
                                                      dtype=cp.float32) * 10
                    print(f"Initial dummy Q-table created on GPU {gpu_id_test}. Shape: {q_table_gpu_test.shape}")

                    test_states = np.random.randint(0, state_space_size_test, size=batch_size_test)
                    test_actions = np.random.randint(0, action_space_size_test, size=batch_size_test)
                    test_rewards = np.random.rand(batch_size_test).astype(np.float32) * 5

                    q_before_gpu = q_table_gpu_test[cp.asarray(test_states), cp.asarray(
                        test_actions)].copy()

                    update_success = batch_update_q_table_gpu(q_table_gpu_test,
                                                              test_states, test_actions, test_rewards,
                                                              lr_test, gamma_test, gpu_id=gpu_id_test)

                    if update_success:
                        print("Batch Q-table update attempted on GPU.")
                        q_after_gpu = q_table_gpu_test[cp.asarray(test_states), cp.asarray(test_actions)]

                        q_before_cpu = cp.asnumpy(q_before_gpu)
                        q_after_cpu = cp.asnumpy(q_after_gpu)

                        print("Sample Q-value changes (CPU representations):")
                        for i in range(min(3, batch_size_test)):
                            print(
                                f"  Item {i} (S:{test_states[i]}, A:{test_actions[i]}, R:{test_rewards[i]:.2f}): Q_before={q_before_cpu[i]:.3f} -> Q_after={q_after_cpu[i]:.3f}")

                        if not cp.allclose(q_before_gpu, q_after_gpu):
                            print("Q-values changed after update: PASS")
                        else:
                            print("Q-values did NOT change significantly after update: CHECK LOGIC/PARAMS")
                    else:
                        print("Batch Q-table update on GPU reported failure or was skipped.")

        except Exception as e_main_test:
            print(f"Error in CuPy test block: {e_main_test}")

        print("\n--- Simulating GPU Batch Action Selection ---")
        if 'q_table_gpu_test' in locals() and isinstance(q_table_gpu_test, cp.ndarray) and CUPY_AVAILABLE and num_gpus_test > 0:
            with cp.cuda.Device(gpu_id_test):
                exploration_rate_test = 0.1
                batch_states_for_action = np.random.randint(0, state_space_size_test, size=batch_size_test)

                chosen_actions_on_cpu = get_batch_actions_gpu(q_table_gpu_test, batch_states_for_action,
                                                              exploration_rate_test, action_space_size_test,
                                                              gpu_id=gpu_id_test)
                if chosen_actions_on_cpu is not None:
                    print(
                        f"Batch actions selected (len: {len(chosen_actions_on_cpu)}): {chosen_actions_on_cpu[:10]}...")
                    greedy_check_actions_gpu = cp.argmax(q_table_gpu_test[cp.asarray(batch_states_for_action), :],
                                                         axis=1)
                    greedy_check_actions_cpu = cp.asnumpy(greedy_check_actions_gpu)

                    num_explored = np.sum(chosen_actions_on_cpu != greedy_check_actions_cpu)
                    print(
                        f"  Number of actions that differed from pure greedy (explored): {num_explored} out of {batch_size_test}")
                    print(f"  (Expected roughly {batch_size_test * exploration_rate_test:.1f} with exploration)")
                else:
                    print("Failed to get batch actions from GPU.")
        else:
            print("Skipping GPU Batch Action Selection test as q_table_gpu_test is not available or no CUDA device.")
    else:
        print("CuPy not available. GPU training utility tests will be skipped.")

    print("\n--- GPU Training Utilities Test Finished ---")