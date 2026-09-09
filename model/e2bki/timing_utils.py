"""
时间测量工具

用于测量各个模块的耗时，帮助定位性能瓶颈
"""

import time
import torch
from collections import defaultdict


class TimingProfiler:
    """时间分析器，用于测量各个模块的耗时"""
    
    def __init__(self, enabled=True, silent=False, sync_cuda=False):
        self.enabled = enabled
        self.silent = silent  # If True, print_summary() does nothing (for profiling without log spam)
        self.sync_cuda = sync_cuda  # If True, synchronize CUDA before/after timing points
        self.timings = defaultdict(list)  # {module_name: [time1, time2, ...]}
        self.current_timings = {}  # {module_name: start_time}
        
    def start(self, module_name):
        """开始计时"""
        if not self.enabled:
            return
        
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.current_timings[module_name] = time.time()
    
    def end(self, module_name):
        """结束计时并记录"""
        if not self.enabled:
            return
        
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if module_name in self.current_timings:
            elapsed = time.time() - self.current_timings[module_name]
            self.timings[module_name].append(elapsed)
            del self.current_timings[module_name]
            return elapsed
        return 0.0
    
    def get_summary(self):
        """获取时间统计摘要"""
        if not self.enabled or not self.timings:
            return {}
        
        summary = {}
        total_time = 0.0
        
        for module_name, times in self.timings.items():
            if times:
                total = sum(times)
                mean = total / len(times)
                summary[module_name] = {
                    'total': total,
                    'mean': mean,
                    'count': len(times),
                    'min': min(times),
                    'max': max(times)
                }
                total_time += total
        
        # 添加百分比
        if total_time > 0:
            for module_name in summary:
                summary[module_name]['percentage'] = (summary[module_name]['total'] / total_time) * 100
        
        summary['_total'] = total_time
        
        return summary
    
    def print_summary(self, title="Timing Summary"):
        """打印时间统计摘要"""
        if not self.enabled or getattr(self, 'silent', False):
            return
        
        summary = self.get_summary()
        if not summary:
            return
        
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        
        # 按总时间排序
        sorted_modules = sorted(
            [(k, v) for k, v in summary.items() if k != '_total'],
            key=lambda x: x[1]['total'],
            reverse=True
        )
        
        total_time = summary['_total']
        
        print(f"{'Module':<40} {'Total (s)':<12} {'Mean (s)':<12} {'Count':<8} {'%':<8}")
        print("-" * 80)
        
        for module_name, stats in sorted_modules:
            print(f"{module_name:<40} {stats['total']:<12.3f} {stats['mean']:<12.3f} "
                  f"{stats['count']:<8} {stats['percentage']:<8.1f}%")
        
        print("-" * 80)
        print(f"{'TOTAL':<40} {total_time:<12.3f}")
        print("=" * 80)
    
    def reset(self):
        """重置所有计时数据"""
        self.timings.clear()
        self.current_timings.clear()
