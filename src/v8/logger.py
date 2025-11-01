#!/usr/bin/env python3
"""
Execution Logger for V8 Pipeline
Tracks execution time and performance metrics for each step.
"""

import time
import os
from datetime import datetime
from pathlib import Path


class V8ExecutionLogger:
    """Logger specifically designed for V8 motion-based optimization pipeline tracking."""
    
    def __init__(self, log_file="execution_log_v8.txt"):
        self.log_file = log_file
        self.start_times = {}
        self.logs = []
        self.metrics = {
            'steps': {},
            'total_frames': 0,
            'detection_rate': 0,
            'cost_type': 'hybrid',
            'optimization_method': '2-opt'
        }
        self.pipeline_start = None
        
    def start_pipeline(self):
        """Mark the start of the complete V8 pipeline."""
        self.pipeline_start = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = "=" * 80
        log_entry = f"""
{header}
V8 MOTION-BASED OPTIMIZATION - EXECUTION LOG
{header}
Started: {timestamp}
Approach: Motion Model + Hybrid Cost (Motion+SSIM) + 2-opt Optimization
{header}
"""
        print(log_entry)
        self.logs.append(log_entry)
        
    def start_step(self, step_name, description=""):
        """Record the start time of a pipeline step."""
        self.start_times[step_name] = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n[{timestamp}] 🚀 Starting: {step_name}"
        if description:
            log_entry += f"\n    Description: {description}"
        print(log_entry)
        self.logs.append(log_entry)
        
    def end_step(self, step_name, details="", metrics=None):
        """
        Record the completion of a pipeline step.
        
        Args:
            step_name: Name of the step
            details: Additional details about completion
            metrics: Dict of step-specific metrics
        """
        if step_name not in self.start_times:
            return
        
        elapsed = time.time() - self.start_times[step_name]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format time nicely
        if elapsed < 60:
            time_str = f"{elapsed:.2f}s"
        else:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.1f}s"
        
        log_entry = f"[{timestamp}] ✅ Completed: {step_name} | Time: {time_str}"
        
        if details:
            log_entry += f"\n    {details}"
        
        if metrics:
            log_entry += "\n    Metrics:"
            for key, value in metrics.items():
                log_entry += f"\n      • {key}: {value}"
                
        print(log_entry)
        self.logs.append(log_entry)
        
        # Store step metrics
        self.metrics['steps'][step_name] = {
            'duration': elapsed,
            'metrics': metrics or {}
        }
        
        del self.start_times[step_name]
        return elapsed
    
    def log(self, message, level="INFO"):
        """
        Add a timestamped log message.
        
        Args:
            message: The message to log
            level: Log level (INFO, WARNING, ERROR, SUCCESS)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add emoji based on level
        emoji_map = {
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'SUCCESS': '✅'
        }
        emoji = emoji_map.get(level, 'ℹ️')
        
        log_entry = f"[{timestamp}] {emoji} {message}"
        print(log_entry)
        self.logs.append(log_entry)
    
    def set_final_metrics(self, total_frames, detection_rate, cost_alpha=0.3, cost_beta=0.7):
        """Set the final V8 pipeline metrics."""
        self.metrics['total_frames'] = total_frames
        self.metrics['detection_rate'] = detection_rate
        self.metrics['cost_alpha'] = cost_alpha
        self.metrics['cost_beta'] = cost_beta
    
    def end_pipeline(self, success=True):
        """Mark the end of the complete V8 pipeline."""
        if self.pipeline_start is None:
            return
        
        total_time = time.time() - self.pipeline_start
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format total time
        if total_time < 60:
            time_str = f"{total_time:.2f}s"
        else:
            mins = int(total_time // 60)
            secs = total_time % 60
            time_str = f"{mins}m {secs:.1f}s"
        
        header = "=" * 80
        status = "✅ SUCCESS" if success else "❌ FAILED"
        
        summary = f"""
{header}
V8 MOTION-BASED OPTIMIZATION - {status}
{header}
Completed: {timestamp}
Total Execution Time: {time_str}

STEP BREAKDOWN:
"""
        for step_name, data in self.metrics['steps'].items():
            duration = data['duration']
            if duration < 60:
                dur_str = f"{duration:.2f}s"
            else:
                mins = int(duration // 60)
                secs = duration % 60
                dur_str = f"{mins}m {secs:.1f}s"
            summary += f"  • {step_name}: {dur_str}\n"
        
        if self.metrics['total_frames'] > 0:
            summary += f"""
FINAL METRICS:
  • Total Frames: {self.metrics['total_frames']}
  • Detection Rate: {self.metrics['detection_rate']:.1f}%
  • Cost Function: Hybrid (alpha={self.metrics.get('cost_alpha', 0.3)}, beta={self.metrics.get('cost_beta', 0.7)})
  • Optimization: 2-opt local search
"""
        
        summary += f"\n{header}\n"
        
        print(summary)
        self.logs.append(summary)
    
    def save(self):
        """Save the complete log to file."""
        # Ensure we're in the project root (go up 2 levels from src/v8/)
        project_root = Path(__file__).parent.parent.parent
        log_path = project_root / self.log_file
        
        with open(log_path, 'w', encoding='utf-8') as f:
            for log in self.logs:
                f.write(log + "\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Log saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n📄 Execution log saved to: {log_path}")
        return str(log_path)


# Example usage
if __name__ == "__main__":
    # Test the logger
    logger = V8ExecutionLogger()
    
    logger.start_pipeline()
    
    logger.start_step("Step 1: YOLO+ByteTrack Detection", "YOLOv8x person detection")
    time.sleep(1)
    logger.end_step("Step 1: YOLO+ByteTrack Detection", 
                    "Person tracking data extracted",
                    metrics={'detection_rate': '100%', 'frames': 300})
    
    logger.start_step("Step 2: Motion Model Building", "Building motion features")
    time.sleep(1)
    logger.end_step("Step 2: Motion Model Building",
                    "Motion model with velocity created",
                    metrics={'features': 'position, velocity, size'})
    
    logger.start_step("Step 3: Hybrid Cost Calculation", "Computing motion + SSIM costs")
    time.sleep(1)
    logger.end_step("Step 3: Hybrid Cost Calculation",
                    "Cost matrix computed",
                    metrics={'alpha': 0.3, 'beta': 0.7, 'matrix_size': '300x300'})
    
    logger.start_step("Step 4: 2-opt Optimization", "Finding optimal order")
    time.sleep(1)
    logger.end_step("Step 4: 2-opt Optimization",
                    "Optimal order found",
                    metrics={'method': '2-opt', 'iterations': 'converged'})
    
    logger.start_step("Step 5: Video Reconstruction", "Assembling final video")
    time.sleep(1)
    logger.end_step("Step 5: Video Reconstruction",
                    "Video created successfully",
                    metrics={'file_size': '~60MB', 'resolution': '1920x1080'})
    
    logger.set_final_metrics(
        total_frames=300,
        detection_rate=100.0,
        cost_alpha=0.3,
        cost_beta=0.7
    )
    
    logger.end_pipeline(success=True)
    logger.save()
