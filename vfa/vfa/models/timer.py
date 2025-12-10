import time
import subprocess
import threading
import numpy as np
from typing import Dict, List, Optional
import os

class RegistrationTimer:
    """Timer class for measuring wall clock time and GPU utilization during registration.
    
    This class provides functionality to:
    1. Measure wall clock time for the full registration loop
    2. Track GPU utilization metrics during the process
    3. Compute statistics like mean, max GPU utilization
    """
    
    def __init__(self, gpu_id=None, gpu_sampling_rate: float = 0.25):
        """Initialize the timer.
        
        Args:
            gpu_sampling_rate: How often to sample GPU metrics in seconds (default: 0.1s)
        """
        self.gpu_sampling_rate = gpu_sampling_rate
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.gpu_utilization: List[float] = []
        self.gpu_memory: List[float] = []
        self._stop_sampling = False
        self._sampling_thread: Optional[threading.Thread] = None
        self.gpu_id = gpu_id
        print(f"Starting timer with GPU ID: {self.gpu_id}")

    def _sample_gpu_metrics(self):
        """Continuously sample GPU metrics in a separate thread."""
        while not self._stop_sampling:
                # Get GPU utilization and memory usage using nvidia-smi
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            # Parse the output for each GPU line
            lines = result.stdout.strip().split('\n')
            if self.gpu_id is not None:
                cuda_devices = str(self.gpu_id)
            else:
                cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            assert cuda_devices is not None
            if cuda_devices:
                gpu_indices = [int(x) for x in cuda_devices.split(',')]
                utils = []
                mems = []
                for i, line in enumerate(lines):
                    if i in gpu_indices:
                        util, mem = map(float, line.split(','))
                        utils.append(util)
                        mems.append(mem)
                # Take average utilization and sum of memory across selected GPUs
                self.gpu_utilization.append(np.mean(utils))
                self.gpu_memory.append(np.sum(mems))
            else:
                # If CUDA_VISIBLE_DEVICES not set, use first GPU
                util, mem = map(float, lines[0].split(','))
                self.gpu_utilization.append(util)
                self.gpu_memory.append(mem)
            
            time.sleep(self.gpu_sampling_rate)

    def start(self):
        """Start the timer and begin GPU metric collection."""
        # Clear previous measurements
        self.gpu_utilization = []
        self.gpu_memory = []
        self._stop_sampling = False
        
        # Start GPU sampling in a separate thread
        self._sampling_thread = threading.Thread(target=self._sample_gpu_metrics)
        self._sampling_thread.daemon = True
        self._sampling_thread.start()
        
        # Record start time
        self.start_time = time.perf_counter()

    def stop(self) -> Dict[str, float]:
        """Stop the timer and return timing and GPU statistics.
        
        Returns:
            Dictionary containing timing and GPU statistics:
            - wall_time: Total wall clock time in seconds
            - gpu_util_mean: Mean GPU utilization percentage
            - gpu_util_max: Maximum GPU utilization percentage
            - gpu_memory_mean: Mean GPU memory usage in MB
            - gpu_memory_max: Maximum GPU memory usage in MB
        """
        self.end_time = time.perf_counter()
        
        # Stop GPU sampling
        self._stop_sampling = True
        if self._sampling_thread is not None:
            self._sampling_thread.join()
        
        # Calculate statistics
        wall_time = self.end_time - self.start_time
        
        # Handle case where no GPU metrics were collected
        if not self.gpu_utilization:
            return {
                'wall_time': wall_time,
                'gpu_util_mean': 0.0,
                'gpu_util_max': 0.0,
                'gpu_memory_mean': 0.0,
                'gpu_memory_max': 0.0
            }
        
        gpu_util_mean = np.mean(self.gpu_utilization)
        gpu_util_max = np.max(self.gpu_utilization)
        gpu_memory_mean = np.mean(self.gpu_memory)
        gpu_memory_max = np.max(self.gpu_memory)
        
        return {
            'wall_time': wall_time,
            'gpu_util_mean': gpu_util_mean,
            'gpu_util_max': gpu_util_max,
            'gpu_memory_mean': gpu_memory_mean,
            'gpu_memory_max': gpu_memory_max
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False  # Don't suppress exceptions
