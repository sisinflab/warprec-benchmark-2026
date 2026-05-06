import logging
import uuid
import time
from datetime import datetime

class LifecycleLogger:
    def __init__(self, filename):
        """
        Initializes a logger with microsecond precision and duration tracking.
        """
        # Unique name to ensure we can create/delete objects without conflicts
        self.logger_name = f"lifecycle_{uuid.uuid4()}"
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Internal dictionary to store start times for duration calculation
        self._timers = {}

        # Setup File Handler
        self.handler = logging.FileHandler(filename)
        
        # Format: [Timestamp] Message
        # We use a simple format here because we inject the precise time in the message
        formatter = logging.Formatter('%(message)s')
        self.handler.setFormatter(formatter)
        
        self.logger.addHandler(self.handler)

    def _get_time_str(self):
        """Returns current time with microsecond precision."""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    def _log(self, message):
        """Internal helper to log with the timestamp."""
        self.logger.info(f"[{self._get_time_str()}] {message}")

    # --- 1. Experiment Lifecycle ---
    def start_experiment(self, exp_name="Experiment"):
        self._timers['experiment'] = time.perf_counter()
        self._log(f"=== START: {exp_name} ===")

    def end_experiment(self):
        duration = time.perf_counter() - self._timers.get('experiment', time.perf_counter())
        self._log(f"=== END: Experiment (Total Time: {duration:.6f}s) ===\n")
        self.close()

    # --- 2. Preprocessing ---
    def start_preprocessing(self):
        self._timers['preprocessing'] = time.perf_counter()
        self._log(">> Preprocessing: STARTED")

    def end_preprocessing(self):
        start_time = self._timers.get('preprocessing', time.perf_counter())
        duration = time.perf_counter() - start_time
        self._log(f">> Preprocessing: ENDED   (Duration: {duration:.6f}s)")

    # --- 3. Training ---
    def start_training(self):
        self._timers['training'] = time.perf_counter()
        self._log(">> Training:      STARTED")

    def end_training(self):
        start_time = self._timers.get('training', time.perf_counter())
        duration = time.perf_counter() - start_time
        self._log(f">> Training:      ENDED   (Duration: {duration:.6f}s)")

    # --- 4. Evaluation ---
    def start_evaluation(self):
        self._timers['evaluation'] = time.perf_counter()
        self._log(">> Evaluation:    STARTED")

    def end_evaluation(self):
        start_time = self._timers.get('evaluation', time.perf_counter())
        duration = time.perf_counter() - start_time
        self._log(f">> Evaluation:    ENDED   (Duration: {duration:.6f}s)")

    # --- Cleanup ---
    def close(self):
        """Detaches the handler so the file is released."""
        if self.handler:
            self.handler.close()
            self.logger.removeHandler(self.handler)