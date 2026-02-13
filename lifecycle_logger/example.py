from lifecycle_logger import LifecycleLogger
import time

LOG_FILE = "fine_grained_log.txt"

# --- RUN 1 ---
print("Starting Run 1...")
logger = LifecycleLogger(LOG_FILE)

logger.start_experiment("Neural_Net_v1")

# Preprocessing
logger.start_preprocessing()
time.sleep(0.123) # Simulate fast work
logger.end_preprocessing()

# Training
logger.start_training()
time.sleep(0.5)   # Simulate training
logger.end_training()

# Evaluation
logger.start_evaluation()
time.sleep(0.2)   # Simulate eval
logger.end_evaluation()

logger.end_experiment()

# CLEANUP: Delete the object to free resources
del logger


# --- RUN 2 (Simulating a separate process or later execution) ---
print("Starting Run 2...")
logger_2 = LifecycleLogger(LOG_FILE)

logger_2.start_experiment("Neural_Net_v2_Optimized")

# Maybe this run skips preprocessing and goes straight to training
logger_2.start_training()
time.sleep(0.35) # Faster training
logger_2.end_training()

logger_2.start_evaluation()
time.sleep(0.15)
logger_2.end_evaluation()

logger_2.end_experiment()
del logger_2

print(f"Done. Check {LOG_FILE}")


"""
[2023-10-27 15:30:01.100200] === START: Neural_Net_v1 ===
[2023-10-27 15:30:01.100500] >> Preprocessing: STARTED
[2023-10-27 15:30:01.223800] >> Preprocessing: ENDED   (Duration: 0.123300s)
[2023-10-27 15:30:01.224000] >> Training:      STARTED
[2023-10-27 15:30:01.725100] >> Training:      ENDED   (Duration: 0.501100s)
[2023-10-27 15:30:01.725500] >> Evaluation:    STARTED
[2023-10-27 15:30:01.926000] >> Evaluation:    ENDED   (Duration: 0.200500s)
[2023-10-27 15:30:01.926200] === END: Experiment (Total Time: 0.826000s) ===

[2023-10-27 15:30:05.500100] === START: Neural_Net_v2_Optimized ===
[2023-10-27 15:30:05.500400] >> Training:      STARTED
[2023-10-27 15:30:05.851000] >> Training:      ENDED   (Duration: 0.350600s)
[2023-10-27 15:30:05.851200] >> Evaluation:    STARTED
[2023-10-27 15:30:06.001500] >> Evaluation:    ENDED   (Duration: 0.150300s)
[2023-10-27 15:30:06.001800] === END: Experiment (Total Time: 0.501700s) ===
"""