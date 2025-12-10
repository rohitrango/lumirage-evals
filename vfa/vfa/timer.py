import os
import time
import logging
from contextlib import contextmanager
import torch
from tabulate import tabulate

class Timer:
    def __init__(self, sync=True):
        """Initialize the Timer class to track execution time and GPU memory usage"""
        torch.cuda.reset_peak_memory_stats()
        self.contexts = {}
        self.start_peak_memory = torch.cuda.max_memory_allocated()
        self.max_peak_memory = self.start_peak_memory
        self.sync = sync
        self.wall_time_start = time.perf_counter()

        # Check if CUDA is available and get the device
        if torch.cuda.is_available():
            cuda_device = os.getenv('CUDA_VISIBLE_DEVICES')
            if cuda_device is not None:
                cuda_device = int(cuda_device.split(',')[0])
                print(f"Using GPU: {cuda_device}")
                # torch.cuda.set_device(f"cuda:{cuda_device}")
                self.cuda_available = True
                # Reset peak stats at initialization
                torch.cuda.reset_peak_memory_stats()
                self.start_peak_memory = torch.cuda.max_memory_allocated()
            else:
                self.cuda_available = False
                logging.warning("CUDA_VISIBLE_DEVICES not set. GPU memory tracking will be disabled.")
                raise ValueError("CUDA VISIBLE DEVICES not set")
        else:
            self.cuda_available = False
            logging.warning("CUDA not available. GPU memory tracking will be disabled.")

    @contextmanager
    def get_context(self, name, sync=None):
        """Context manager to measure execution time and GPU memory usage for a block of code
        
        Args:
            name (str): Name of the context/block being measured
        """
        # Initialize context stats if not exists
        if name not in self.contexts:
            self.contexts[name] = {
                'times': [],
                'memory_before': [],
                'memory_after': [],
                'memory_diff': []
            }
        
        # Record starting time and memory
        start_time = time.perf_counter()
        if self.sync or sync:
            torch.cuda.synchronize()
        if self.cuda_available:
            # before resetting, keep track of the current peak memory
            self.max_peak_memory = max(self.max_peak_memory, torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated() if self.cuda_available else 0
        
        try:
            yield
        finally:
            # Record ending time and memory
            if self.sync or sync:
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            end_memory = torch.cuda.memory_allocated() if self.cuda_available else 0
            
            # Calculate time and memory differences
            time_diff = end_time - start_time
            memory_diff = end_memory - start_memory if self.cuda_available else 0
            
            # Store measurements
            self.contexts[name]['times'].append(time_diff)
            if self.cuda_available:
                self.contexts[name]['memory_before'].append(start_memory)
                self.contexts[name]['memory_after'].append(end_memory)
                self.contexts[name]['memory_diff'].append(memory_diff)

    def get_stats(self):
        """Print statistics for all contexts and return peak memory usage and table format
        
        Returns:
            tuple: (peak_diff, table_str) where peak_diff is peak memory usage in bytes 
                  and table_str is the formatted table string
        """
        # Prepare table headers and rows
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        headers = ["Context", "Calls", "Total Time (s)", "Time %", "Avg Time (s)"]
        if self.cuda_available:
            headers.extend(["Avg Memory (MB)", "Max Memory (MB)"])
        
        # First calculate total time across all contexts for percentage
        total_time_all = sum(sum(stats['times']) for stats in self.contexts.values())
        
        rows = []
        for name, stats in self.contexts.items():
            avg_time = sum(stats['times']) / len(stats['times'])
            total_time = sum(stats['times'])
            time_percentage = (total_time / total_time_all * 100) if total_time_all > 0 else 0
            
            row = [
                name,
                len(stats['times']),
                f"{total_time:.4f}",
                f"{time_percentage:.2f}%",
                f"{avg_time:.4f}"
            ]
            
            if self.cuda_available:
                avg_memory_diff = sum(stats['memory_diff']) / len(stats['memory_diff'])
                max_memory_diff = max(stats['memory_diff'])
                row.extend([
                    f"{avg_memory_diff/1024/1024:.2f}",
                    f"{max_memory_diff/1024/1024:.2f}"
                ])
            
            rows.append(row)
        
        # Create table string
        table_str = tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".4f")
        
        print("\n=== Timer Statistics ===")
        print(f"Wall time: {end_time - self.wall_time_start:.2f} seconds")
        print(table_str)
        
        # Get peak memory usage and reset stats
        current_peak = max(self.max_peak_memory, torch.cuda.max_memory_allocated()) if self.cuda_available else 0
        peak_diff = current_peak - self.start_peak_memory if self.cuda_available else 0
        
        if self.cuda_available:
            print(f"\nPeak memory usage: {peak_diff/1024/1024:.2f} MB, starting peak memory: {self.start_peak_memory/1024/1024:.2f} MB, total peak memory: {current_peak/1024/1024:.2f} MB")
        
        # Reset context stats
        self.contexts.clear()
        
        # Reset CUDA peak stats if available
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
            self.start_peak_memory = torch.cuda.max_memory_allocated()
        
        return peak_diff, table_str
