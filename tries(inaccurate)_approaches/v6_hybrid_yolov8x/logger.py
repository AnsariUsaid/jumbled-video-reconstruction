#!/usr/bin/env python3
"""
Execution Logger for V6 Hybrid Pipeline
Tracks execution time and performance metrics for each stage.
"""

import time
import os
from datetime import datetime
from pathlib import Path


class V6ExecutionLogger:
    """Logger specifically designed for V6 Hybrid pipeline tracking."""
    
    def __init__(self, log_file="execution_log_v6.txt"):
        self.log_file = log_file
        self.start_times = {}
        self.logs = []
        self.metrics = {
            'stages': {},
            'total_frames': 0,
            'detection_rate': 0,
            'avg_step_distance': 0,
            'jump_count': 0
        }
        self.pipeline_start = None
        
    def start_pipeline(self):
        """Mark the start of the complete V6 pipeline."""
        self.pipeline_start = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = "=" * 80
        log_entry = f"""
{header}
V6 HYBRID PIPELINE - EXECUTION LOG
{header}
Started: {timestamp}
Approach: CNN (ResNet50) + YOLOv8x Spatial Optimization
{header}
"""
        print(log_entry)
        self.logs.append(log_entry)
        
    def start_stage(self, stage_name, description=""):
        """Record the start time of a pipeline stage."""
        self.start_times[stage_name] = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n[{timestamp}] 🚀 Starting: {stage_name}"
        if description:
            log_entry += f"\n    Description: {description}"
        print(log_entry)
        self.logs.append(log_entry)
        
    def end_stage(self, stage_name, details="", metrics=None):
        """
        Record the completion of a pipeline stage.
        
        Args:
            stage_name: Name of the stage
            details: Additional details about completion
            metrics: Dict of stage-specific metrics (e.g., {'frames': 300, 'detection_rate': 100})
        """
        if stage_name not in self.start_times:
            return
        
        elapsed = time.time() - self.start_times[stage_name]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format time nicely
        if elapsed < 60:
            time_str = f"{elapsed:.2f}s"
        else:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.1f}s"
        
        log_entry = f"[{timestamp}] ✅ Completed: {stage_name} | Time: {time_str}"
        
        if details:
            log_entry += f"\n    {details}"
        
        if metrics:
            log_entry += "\n    Metrics:"
            for key, value in metrics.items():
                log_entry += f"\n      • {key}: {value}"
                
        print(log_entry)
        self.logs.append(log_entry)
        
        # Store stage metrics
        self.metrics['stages'][stage_name] = {
            'duration': elapsed,
            'metrics': metrics or {}
        }
        
        del self.start_times[stage_name]
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
    
    def set_final_metrics(self, total_frames, detection_rate, avg_step, jump_count):
        """Set the final V6 pipeline metrics."""
        self.metrics['total_frames'] = total_frames
        self.metrics['detection_rate'] = detection_rate
        self.metrics['avg_step_distance'] = avg_step
        self.metrics['jump_count'] = jump_count
    
    def end_pipeline(self, success=True):
        """Mark the end of the complete V6 pipeline."""
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
V6 HYBRID PIPELINE - {status}
{header}
Completed: {timestamp}
Total Execution Time: {time_str}

STAGE BREAKDOWN:
"""
        for stage_name, data in self.metrics['stages'].items():
            duration = data['duration']
            if duration < 60:
                dur_str = f"{duration:.2f}s"
            else:
                mins = int(duration // 60)
                secs = duration % 60
                dur_str = f"{mins}m {secs:.1f}s"
            summary += f"  • {stage_name}: {dur_str}\n"
        
        if self.metrics['total_frames'] > 0:
            summary += f"""
FINAL METRICS:
  • Total Frames: {self.metrics['total_frames']}
  • Detection Rate: {self.metrics['detection_rate']:.1f}% ({int(self.metrics['total_frames'] * self.metrics['detection_rate'] / 100)}/{self.metrics['total_frames']})
  • Avg Step Distance: {self.metrics['avg_step_distance']:.2f} pixels
  • Jump Count: {self.metrics['jump_count']}
  • Jump Rate: {(self.metrics['jump_count'] / (self.metrics['total_frames'] - 1) * 100):.2f}%
"""
        
        summary += f"\n{header}\n"
        
        print(summary)
        self.logs.append(summary)
    
    def save(self):
        """Save the complete log to file."""
        # Ensure we're in the project root
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
    logger = V6ExecutionLogger()
    
    logger.start_pipeline()
    
    logger.start_stage("Stage 1: CNN Semantic Ordering", "Using ResNet50 for feature extraction")
    time.sleep(1)
    logger.end_stage("Stage 1: CNN Semantic Ordering", 
                     "Semantically ordered frames created",
                     metrics={'similarity': '99.6%', 'frames': 300})
    
    logger.start_stage("Stage 2: YOLOv8x Detection", "Detecting person in each frame")
    time.sleep(1)
    logger.end_stage("Stage 2: YOLOv8x Detection",
                     "Person tracking data extracted",
                     metrics={'detection_rate': '100%', 'frames_detected': 300})
    
    logger.start_stage("Stage 3: Spatial Re-ordering", "Nearest-neighbor optimization")
    time.sleep(1)
    logger.end_stage("Stage 3: Spatial Re-ordering",
                     "Optimal spatial path created",
                     metrics={'avg_step': '3.5px', 'jumps': 1})
    
    logger.start_stage("Stage 4: Video Reconstruction", "Assembling final video")
    time.sleep(1)
    logger.end_stage("Stage 4: Video Reconstruction",
                     "Video created successfully",
                     metrics={'file_size': '63MB', 'resolution': '1920x1080'})
    
    logger.set_final_metrics(
        total_frames=300,
        detection_rate=100.0,
        avg_step=3.5,
        jump_count=1
    )
    
    logger.end_pipeline(success=True)
    logger.save()
