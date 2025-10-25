import time
import os
from datetime import datetime


class ExecutionLogger:
    def __init__(self, log_file="execution_log.txt"):
        self.log_file = log_file
        self.start_times = {}
        self.logs = []
        
    def start_phase(self, phase_name):
        self.start_times[phase_name] = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Starting: {phase_name}"
        print(log_entry)
        self.logs.append(log_entry)
        
    def end_phase(self, phase_name, details=""):
        if phase_name not in self.start_times:
            return
        
        elapsed = time.time() - self.start_times[phase_name]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Completed: {phase_name} | Time: {elapsed:.2f}s"
        if details:
            log_entry += f" | {details}"
        print(log_entry)
        self.logs.append(log_entry)
        
        del self.start_times[phase_name]
        return elapsed
    
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.logs.append(log_entry)
    
    def save(self):
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("JUMBLED FRAMES RECONSTRUCTION - EXECUTION LOG\n")
            f.write("=" * 80 + "\n\n")
            for log in self.logs:
                f.write(log + "\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("Log saved at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 80 + "\n")
        print(f"\nExecution log saved to: {self.log_file}")
