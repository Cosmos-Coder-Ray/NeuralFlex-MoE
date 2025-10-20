"""
Performance profiling and monitoring utilities.
Helps identify bottlenecks and optimize training/inference.
"""

import time
import functools
from contextlib import contextmanager
from typing import Dict, Optional, Callable
import torch

# Profiling tools - import what's available
try:
    import nvitop
    from nvitop import Device
    HAS_NVITOP = True
except ImportError:
    HAS_NVITOP = False

try:
    from torch.profiler import profile, ProfilerActivity, schedule
    from torch.profiler import tensorboard_trace_handler
    HAS_TORCH_PROFILER = True
except ImportError:
    HAS_TORCH_PROFILER = False

try:
    import memray
    HAS_MEMRAY = True
except ImportError:
    HAS_MEMRAY = False

try:
    import scalene
    HAS_SCALENE = True
except ImportError:
    HAS_SCALENE = False


class PerformanceMonitor:
    """
    Simple performance monitor that tracks execution time and memory.
    Useful for finding slow parts of your code.
    """
    
    def __init__(self):
        self.timings = {}
        self.memory_stats = {}
        self.call_counts = {}
    
    def reset(self):
        """Clear all collected stats"""
        self.timings.clear()
        self.memory_stats.clear()
        self.call_counts.clear()
    
    @contextmanager
    def measure(self, name: str):
        """
        Context manager for timing code blocks.
        
        Usage:
            with monitor.measure("forward_pass"):
                output = model(input)
        """
        start_time = time.perf_counter()
        
        # Track GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()
        else:
            start_mem = 0
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_mem = torch.cuda.memory_allocated()
                mem_used = (end_mem - start_mem) / 1024**2  # Convert to MB
            else:
                mem_used = 0
            
            elapsed = time.perf_counter() - start_time
            
            # Update stats
            if name not in self.timings:
                self.timings[name] = []
                self.memory_stats[name] = []
                self.call_counts[name] = 0
            
            self.timings[name].append(elapsed)
            self.memory_stats[name].append(mem_used)
            self.call_counts[name] += 1
    
    def get_summary(self) -> Dict:
        """Get a summary of all measurements"""
        summary = {}
        
        for name in self.timings:
            times = self.timings[name]
            mems = self.memory_stats[name]
            
            summary[name] = {
                "calls": self.call_counts[name],
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "avg_memory_mb": sum(mems) / len(mems) if mems else 0,
            }
        
        return summary
    
    def print_summary(self):
        """Print a nice formatted summary"""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("Performance Summary")
        print("="*70)
        
        for name, stats in sorted(summary.items(), key=lambda x: x[1]["total_time"], reverse=True):
            print(f"\n{name}:")
            print(f"  Calls: {stats['calls']}")
            print(f"  Total time: {stats['total_time']:.3f}s")
            print(f"  Avg time: {stats['avg_time']*1000:.2f}ms")
            print(f"  Min/Max: {stats['min_time']*1000:.2f}ms / {stats['max_time']*1000:.2f}ms")
            if stats['avg_memory_mb'] > 0:
                print(f"  Avg memory: {stats['avg_memory_mb']:.1f}MB")
        
        print("="*70 + "\n")


def profile_function(name: Optional[str] = None):
    """
    Decorator to automatically profile a function.
    
    Usage:
        @profile_function("my_function")
        def my_function(x):
            return x * 2
    """
    def decorator(func: Callable):
        func_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = getattr(wrapper, '_monitor', None)
            if monitor is None:
                # Create a global monitor if none exists
                monitor = PerformanceMonitor()
                wrapper._monitor = monitor
            
            with monitor.measure(func_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class GPUMonitor:
    """Monitor GPU usage in real-time using nvitop"""
    
    def __init__(self):
        self.available = HAS_NVITOP and torch.cuda.is_available()
        if self.available:
            self.devices = Device.all()
        else:
            self.devices = []
    
    def get_stats(self) -> Dict:
        """Get current GPU stats"""
        if not self.available:
            return {"error": "nvitop not available or no GPU"}
        
        stats = {}
        for i, device in enumerate(self.devices):
            stats[f"gpu_{i}"] = {
                "name": device.name(),
                "memory_used": device.memory_used(),
                "memory_total": device.memory_total(),
                "memory_percent": device.memory_percent(),
                "gpu_utilization": device.gpu_utilization(),
                "temperature": device.temperature(),
            }
        
        return stats
    
    def print_stats(self):
        """Print GPU stats in a readable format"""
        stats = self.get_stats()
        
        if "error" in stats:
            print(stats["error"])
            return
        
        print("\n" + "="*70)
        print("GPU Status")
        print("="*70)
        
        for gpu_id, info in stats.items():
            print(f"\n{gpu_id.upper()}: {info['name']}")
            print(f"  Memory: {info['memory_used']/1024:.1f}GB / {info['memory_total']/1024:.1f}GB ({info['memory_percent']:.1f}%)")
            print(f"  Utilization: {info['gpu_utilization']:.1f}%")
            print(f"  Temperature: {info['temperature']}°C")
        
        print("="*70 + "\n")


class TorchProfiler:
    """
    Wrapper around PyTorch's built-in profiler.
    Generates detailed traces for TensorBoard.
    """
    
    def __init__(self, output_dir: str = "./logs/profiler"):
        self.output_dir = output_dir
        self.available = HAS_TORCH_PROFILER
    
    @contextmanager
    def profile_context(self, use_cuda: bool = True):
        """
        Context manager for profiling.
        
        Usage:
            with profiler.profile_context():
                model(input)
        """
        if not self.available:
            print("Warning: torch.profiler not available")
            yield
            return
        
        activities = [ProfilerActivity.CPU]
        if use_cuda and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=tensorboard_trace_handler(self.output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            yield prof
    
    def profile_model(self, model, sample_input, num_iterations: int = 10):
        """
        Profile a model with sample input.
        Results can be viewed in TensorBoard.
        """
        if not self.available:
            print("Warning: torch.profiler not available")
            return
        
        model.eval()
        
        with self.profile_context() as prof:
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = model(sample_input)
                prof.step()
        
        print(f"✓ Profiling complete. View results with:")
        print(f"  tensorboard --logdir={self.output_dir}")


def benchmark_inference(model, tokenizer, prompts: list, num_runs: int = 10):
    """
    Benchmark inference speed across multiple prompts.
    Returns tokens/second and latency stats.
    """
    monitor = PerformanceMonitor()
    gpu_monitor = GPUMonitor()
    
    model.eval()
    total_tokens = 0
    
    print(f"Running benchmark with {len(prompts)} prompts, {num_runs} runs each...")
    
    for run in range(num_runs):
        for i, prompt in enumerate(prompts):
            with monitor.measure(f"prompt_{i}"):
                inputs = tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=100,
                        do_sample=False
                    )
                
                total_tokens += outputs.shape[1]
    
    # Calculate stats
    summary = monitor.get_summary()
    total_time = sum(s["total_time"] for s in summary.values())
    tokens_per_second = total_tokens / total_time
    
    print(f"\n{'='*70}")
    print(f"Benchmark Results")
    print(f"{'='*70}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {tokens_per_second:.1f} tokens/second")
    print(f"Average latency: {total_time / (len(prompts) * num_runs) * 1000:.1f}ms per prompt")
    print(f"{'='*70}\n")
    
    # Show GPU stats
    gpu_monitor.print_stats()
    
    return {
        "tokens_per_second": tokens_per_second,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "detailed_stats": summary
    }


# Global monitor instance for convenience
_global_monitor = PerformanceMonitor()

def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor"""
    return _global_monitor
