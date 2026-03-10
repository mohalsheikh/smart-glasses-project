# src/utils/telemetry_viz.py
"""
Enhanced Telemetry Visualization Module
Generates comprehensive graphs and dashboards from JSONL telemetry data.

Supports visualization of:
- Frame metrics (FPS, latency, detections)
- Voice interactions (timing, success rates)
- Safety events (obstacles, guidance)
- AI operations (scene AI, OCR timing)
- Speech output (TTS patterns)
- System health (memory, CPU)
- Error tracking

Usage:
    python -m src.utils.telemetry_viz telemetry/runs/run_YYYYMMDD_HHMMSS.jsonl
    python -m src.utils.telemetry_viz --latest
    python -m src.utils.telemetry_viz --all
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class TelemetryData:
    """Container for parsed telemetry data."""
    meta: Dict[str, Any] = field(default_factory=dict)
    frames: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    voice: List[Dict[str, Any]] = field(default_factory=list)
    safety: List[Dict[str, Any]] = field(default_factory=list)
    ai: List[Dict[str, Any]] = field(default_factory=list)
    speech: List[Dict[str, Any]] = field(default_factory=list)
    system: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Derived metrics (computed after loading)
    frame_indices: List[int] = field(default_factory=list)
    fps_values: List[float] = field(default_factory=list)
    capture_ms: List[float] = field(default_factory=list)
    detect_ms: List[float] = field(default_factory=list)
    loop_total_ms: List[float] = field(default_factory=list)
    n_detections: List[int] = field(default_factory=list)
    top_labels: List[str] = field(default_factory=list)
    top_confs: List[float] = field(default_factory=list)
    timestamps_ms: List[int] = field(default_factory=list)
    
    # Label frequency tracking
    all_labels: List[str] = field(default_factory=list)
    
    # Voice metrics
    voice_listen_ms: List[float] = field(default_factory=list)
    voice_process_ms: List[float] = field(default_factory=list)
    voice_total_ms: List[float] = field(default_factory=list)
    
    # Safety metrics
    safety_severities: List[int] = field(default_factory=list)
    safety_types: List[str] = field(default_factory=list)
    
    # AI metrics
    ai_latencies: List[float] = field(default_factory=list)
    ai_operations: List[str] = field(default_factory=list)
    
    # System metrics
    cpu_percent: List[float] = field(default_factory=list)
    memory_percent: List[float] = field(default_factory=list)


def load_telemetry(path: Path) -> TelemetryData:
    """Load and parse a JSONL telemetry file."""
    data = TelemetryData()
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            rec_type = record.get('type', '')
            
            if rec_type == 'meta':
                data.meta.update(record)
            elif rec_type == 'frame':
                data.frames.append(record)
            elif rec_type == 'event':
                data.events.append(record)
            elif rec_type == 'voice':
                data.voice.append(record)
            elif rec_type == 'safety':
                data.safety.append(record)
            elif rec_type == 'ai':
                data.ai.append(record)
            elif rec_type == 'speech':
                data.speech.append(record)
            elif rec_type == 'system':
                data.system.append(record)
            elif rec_type == 'error':
                data.errors.append(record)
    
    # Extract time series data from frames
    for frame in data.frames:
        data.frame_indices.append(frame.get('frame_idx', 0))
        data.fps_values.append(frame.get('fps', 0.0))
        data.capture_ms.append(frame.get('capture_ms', 0.0))
        data.detect_ms.append(frame.get('detect_ms', 0.0))
        data.loop_total_ms.append(frame.get('loop_total_ms', 0.0))
        data.n_detections.append(frame.get('n_detections', 0))
        data.top_labels.append(frame.get('top_label', ''))
        data.top_confs.append(frame.get('top_conf', 0.0))
        data.timestamps_ms.append(frame.get('ts_ms', 0))
        
        # Collect all labels from dets_compact
        dets_compact = frame.get('dets_compact', [])
        for det in dets_compact:
            label = det.get('label', '')
            if label:
                data.all_labels.append(label)
    
    # Extract voice metrics
    for v in data.voice:
        if v.get('listen_ms'):
            data.voice_listen_ms.append(v['listen_ms'])
        if v.get('process_ms'):
            data.voice_process_ms.append(v['process_ms'])
        if v.get('total_ms'):
            data.voice_total_ms.append(v['total_ms'])
    
    # Extract safety metrics
    for s in data.safety:
        data.safety_severities.append(s.get('severity', 0))
        data.safety_types.append(s.get('event_type', 'unknown'))
    
    # Extract AI metrics
    for a in data.ai:
        data.ai_latencies.append(a.get('latency_ms', 0.0))
        data.ai_operations.append(a.get('operation', 'unknown'))
    
    # Extract system metrics
    for sys in data.system:
        data.cpu_percent.append(sys.get('cpu_percent', 0.0))
        data.memory_percent.append(sys.get('memory_percent', 0.0))
    
    return data


def get_latest_telemetry_file(telemetry_dir: Path = Path('telemetry/runs')) -> Optional[Path]:
    """Get the most recent telemetry file."""
    if not telemetry_dir.exists():
        return None
    
    files = sorted(telemetry_dir.glob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def get_all_telemetry_files(telemetry_dir: Path = Path('telemetry/runs')) -> List[Path]:
    """Get all telemetry files sorted by modification time."""
    if not telemetry_dir.exists():
        return []
    
    return sorted(telemetry_dir.glob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)


class TelemetryVisualizer:
    """Generate visualizations from telemetry data."""
    
    def __init__(self, data: TelemetryData, output_dir: Path = Path('telemetry/graphs')):
        self.data = data
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        self.colors = {
            'primary': '#2563eb',      # Blue
            'secondary': '#16a34a',    # Green
            'warning': '#eab308',      # Yellow
            'danger': '#dc2626',       # Red
            'neutral': '#6b7280',      # Gray
            'purple': '#9333ea',       # Purple
            'orange': '#ea580c',       # Orange
            'cyan': '#06b6d4',         # Cyan
            'pink': '#ec4899',         # Pink
        }
        
    def _setup_style(self):
        """Apply consistent styling to plots."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        
    def generate_fps_graph(self, save: bool = True) -> Optional[Path]:
        """Generate FPS over time graph."""
        if not HAS_MATPLOTLIB or not self.data.frames:
            return None
            
        self._setup_style()
        fig, ax = plt.subplots(figsize=(12, 5))
        
        frames = self.data.frame_indices
        fps = self.data.fps_values
        
        # Plot FPS line
        ax.plot(frames, fps, color=self.colors['primary'], linewidth=1.5, alpha=0.8)
        
        # Add moving average
        if HAS_NUMPY and len(fps) > 10:
            window = min(30, len(fps) // 5)
            fps_smooth = np.convolve(fps, np.ones(window)/window, mode='valid')
            frames_smooth = frames[window-1:]
            ax.plot(frames_smooth, fps_smooth, color=self.colors['danger'], 
                   linewidth=2, label=f'{window}-frame moving avg')
        
        # Average line
        avg_fps = sum(fps) / len(fps) if fps else 0
        ax.axhline(y=avg_fps, color=self.colors['secondary'], linestyle='--', 
                   linewidth=2, label=f'Average: {avg_fps:.1f} FPS')
        
        ax.set_xlabel('Frame Index', fontsize=12)
        ax.set_ylabel('FPS', fontsize=12)
        ax.set_title('📊 Frames Per Second Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'fps_over_time.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None
        
    def generate_latency_graph(self, save: bool = True) -> Optional[Path]:
        """Generate latency breakdown graph."""
        if not HAS_MATPLOTLIB or not self.data.frames:
            return None
            
        self._setup_style()
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        frames = self.data.frame_indices
        
        # Top plot: stacked area for latency components
        ax1 = axes[0]
        
        if HAS_NUMPY:
            capture = np.array(self.data.capture_ms)
            detect = np.array(self.data.detect_ms)
            
            ax1.fill_between(frames, 0, capture, alpha=0.6, 
                           color=self.colors['primary'], label='Capture')
            ax1.fill_between(frames, capture, capture + detect, alpha=0.6,
                           color=self.colors['secondary'], label='Detection')
        else:
            ax1.plot(frames, self.data.capture_ms, color=self.colors['primary'], 
                    label='Capture', linewidth=1)
            ax1.plot(frames, self.data.detect_ms, color=self.colors['secondary'],
                    label='Detection', linewidth=1)
        
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_title('⏱️ Processing Latency Breakdown', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        
        # Bottom plot: total loop time
        ax2 = axes[1]
        ax2.plot(frames, self.data.loop_total_ms, color=self.colors['purple'], 
                linewidth=1.5, alpha=0.8)
        
        avg_loop = sum(self.data.loop_total_ms) / len(self.data.loop_total_ms) if self.data.loop_total_ms else 0
        ax2.axhline(y=avg_loop, color=self.colors['danger'], linestyle='--',
                   linewidth=2, label=f'Average: {avg_loop:.1f}ms')
        
        # Target frame time line (e.g., 33ms for 30 FPS)
        ax2.axhline(y=33.3, color=self.colors['warning'], linestyle=':',
                   linewidth=2, label='30 FPS target')
        
        ax2.set_xlabel('Frame Index', fontsize=12)
        ax2.set_ylabel('Total Loop Time (ms)', fontsize=12)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'latency_breakdown.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None
        
    def generate_detection_count_graph(self, save: bool = True) -> Optional[Path]:
        """Generate detection count over time."""
        if not HAS_MATPLOTLIB or not self.data.frames:
            return None
            
        self._setup_style()
        fig, ax = plt.subplots(figsize=(12, 5))
        
        frames = self.data.frame_indices
        counts = self.data.n_detections
        
        ax.bar(frames, counts, color=self.colors['primary'], alpha=0.7, width=1.0)
        
        # Moving average
        if HAS_NUMPY and len(counts) > 10:
            window = min(30, len(counts) // 5)
            counts_smooth = np.convolve(counts, np.ones(window)/window, mode='valid')
            frames_smooth = frames[window-1:]
            ax.plot(frames_smooth, counts_smooth, color=self.colors['danger'],
                   linewidth=2.5, label=f'{window}-frame moving avg')
        
        avg_count = sum(counts) / len(counts) if counts else 0
        ax.axhline(y=avg_count, color=self.colors['secondary'], linestyle='--',
                  linewidth=2, label=f'Average: {avg_count:.1f}')
        
        ax.set_xlabel('Frame Index', fontsize=12)
        ax.set_ylabel('Number of Detections', fontsize=12)
        ax.set_title('🎯 Object Detections Per Frame', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'detection_count.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None
        
    def generate_label_frequency_graph(self, top_n: int = 15, save: bool = True) -> Optional[Path]:
        """Generate bar chart of most frequent detection labels."""
        if not HAS_MATPLOTLIB or not self.data.all_labels:
            return None
            
        self._setup_style()
        fig, ax = plt.subplots(figsize=(12, 6))
        
        label_counts = Counter(self.data.all_labels)
        top_labels = label_counts.most_common(top_n)
        
        if not top_labels:
            return None
            
        labels = [l[0] for l in top_labels]
        counts = [l[1] for l in top_labels]
        
        # Horizontal bar chart with gradient colors
        colors = [plt.cm.Blues(0.4 + 0.5 * i / len(labels)) for i in range(len(labels))]
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, counts, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Top to bottom
        ax.set_xlabel('Detection Count', fontsize=12)
        ax.set_title(f'🏷️ Top {top_n} Detected Objects', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'label_frequency.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None
        
    def generate_confidence_distribution(self, save: bool = True) -> Optional[Path]:
        """Generate histogram of detection confidence scores."""
        if not HAS_MATPLOTLIB or not self.data.top_confs:
            return None
            
        self._setup_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Filter out zero confidences
        confs = [c for c in self.data.top_confs if c > 0]
        
        if not confs:
            return None
        
        ax.hist(confs, bins=20, range=(0, 1), color=self.colors['primary'], 
               alpha=0.7, edgecolor='white', linewidth=1.2)
        
        avg_conf = sum(confs) / len(confs)
        ax.axvline(x=avg_conf, color=self.colors['danger'], linestyle='--',
                  linewidth=2, label=f'Average: {avg_conf:.2f}')
        
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('📈 Detection Confidence Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'confidence_distribution.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None

    def generate_voice_timing_graph(self, save: bool = True) -> Optional[Path]:
        """Generate voice interaction timing analysis."""
        if not HAS_MATPLOTLIB or not self.data.voice:
            return None
        
        self._setup_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Timing breakdown
        ax1 = axes[0]
        
        listen_times = self.data.voice_listen_ms
        process_times = self.data.voice_process_ms
        
        if listen_times or process_times:
            categories = []
            values = []
            colors_list = []
            
            if listen_times:
                categories.append('Listen/Transcribe')
                values.append(sum(listen_times) / len(listen_times))
                colors_list.append(self.colors['primary'])
            
            if process_times:
                categories.append('Process/Response')
                values.append(sum(process_times) / len(process_times))
                colors_list.append(self.colors['secondary'])
            
            bars = ax1.bar(categories, values, color=colors_list, alpha=0.8)
            
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{val:.0f}ms', ha='center', va='bottom', fontsize=11)
            
            ax1.set_ylabel('Average Time (ms)', fontsize=12)
            ax1.set_title('🎤 Voice Interaction Timing', fontsize=14, fontweight='bold')
        
        # Right: Total time distribution
        ax2 = axes[1]
        
        if self.data.voice_total_ms:
            ax2.hist(self.data.voice_total_ms, bins=15, color=self.colors['purple'],
                    alpha=0.7, edgecolor='white')
            avg_total = sum(self.data.voice_total_ms) / len(self.data.voice_total_ms)
            ax2.axvline(x=avg_total, color=self.colors['danger'], linestyle='--',
                       linewidth=2, label=f'Average: {avg_total:.0f}ms')
            ax2.set_xlabel('Total Response Time (ms)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Voice Response Time Distribution', fontsize=12, fontweight='bold')
            ax2.legend()
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'voice_timing.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None

    def generate_safety_analysis_graph(self, save: bool = True) -> Optional[Path]:
        """Generate safety events analysis."""
        if not HAS_MATPLOTLIB or not self.data.safety:
            return None
        
        self._setup_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Event types pie chart
        ax1 = axes[0]
        
        type_counts = Counter(self.data.safety_types)
        if type_counts:
            labels = list(type_counts.keys())
            sizes = list(type_counts.values())
            colors_list = [self.colors['primary'], self.colors['secondary'], 
                          self.colors['warning'], self.colors['danger'],
                          self.colors['purple'], self.colors['orange']][:len(labels)]
            
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                               colors=colors_list, startangle=90)
            ax1.set_title('🚨 Safety Event Types', fontsize=14, fontweight='bold')
        
        # Right: Severity distribution
        ax2 = axes[1]
        
        if self.data.safety_severities:
            severity_counts = Counter(self.data.safety_severities)
            severity_labels = ['Info (0)', 'Near (1)', 'Close (2)', 'Danger (3)']
            severity_values = [severity_counts.get(i, 0) for i in range(4)]
            colors_list = [self.colors['neutral'], self.colors['warning'],
                          self.colors['orange'], self.colors['danger']]
            
            bars = ax2.bar(severity_labels, severity_values, color=colors_list, alpha=0.8)
            
            for bar, val in zip(bars, severity_values):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(val), ha='center', va='bottom', fontsize=11)
            
            ax2.set_ylabel('Event Count', fontsize=12)
            ax2.set_title('Safety Event Severity', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'safety_analysis.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None

    def generate_ai_performance_graph(self, save: bool = True) -> Optional[Path]:
        """Generate AI operation performance analysis."""
        if not HAS_MATPLOTLIB or not self.data.ai:
            return None
        
        self._setup_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Operation type breakdown
        ax1 = axes[0]
        
        op_latencies = defaultdict(list)
        for a in self.data.ai:
            op = a.get('operation', 'unknown')
            lat = a.get('latency_ms', 0)
            op_latencies[op].append(lat)
        
        if op_latencies:
            ops = list(op_latencies.keys())
            avg_latencies = [sum(v)/len(v) for v in op_latencies.values()]
            
            colors_list = [self.colors['primary'], self.colors['secondary'],
                          self.colors['purple'], self.colors['orange'],
                          self.colors['cyan']][:len(ops)]
            
            bars = ax1.barh(ops, avg_latencies, color=colors_list, alpha=0.8)
            
            for bar, val in zip(bars, avg_latencies):
                ax1.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                        f'{val:.0f}ms', va='center', fontsize=10)
            
            ax1.set_xlabel('Average Latency (ms)', fontsize=12)
            ax1.set_title('🤖 AI Operation Latency', fontsize=14, fontweight='bold')
        
        # Right: Latency distribution
        ax2 = axes[1]
        
        if self.data.ai_latencies:
            ax2.hist(self.data.ai_latencies, bins=20, color=self.colors['purple'],
                    alpha=0.7, edgecolor='white')
            avg_lat = sum(self.data.ai_latencies) / len(self.data.ai_latencies)
            ax2.axvline(x=avg_lat, color=self.colors['danger'], linestyle='--',
                       linewidth=2, label=f'Average: {avg_lat:.0f}ms')
            ax2.set_xlabel('Latency (ms)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('AI Latency Distribution', fontsize=12, fontweight='bold')
            ax2.legend()
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'ai_performance.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None

    def generate_system_health_graph(self, save: bool = True) -> Optional[Path]:
        """Generate system health metrics."""
        if not HAS_MATPLOTLIB or not self.data.system:
            return None
        
        self._setup_style()
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        indices = list(range(len(self.data.cpu_percent)))
        
        # Top: CPU usage
        ax1 = axes[0]
        ax1.plot(indices, self.data.cpu_percent, color=self.colors['primary'],
                linewidth=1.5, alpha=0.8)
        ax1.fill_between(indices, self.data.cpu_percent, alpha=0.3, color=self.colors['primary'])
        ax1.set_ylabel('CPU %', fontsize=12)
        ax1.set_title('💻 System Resource Usage', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 100)
        
        if self.data.cpu_percent:
            avg_cpu = sum(self.data.cpu_percent) / len(self.data.cpu_percent)
            ax1.axhline(y=avg_cpu, color=self.colors['danger'], linestyle='--',
                       label=f'Average: {avg_cpu:.1f}%')
            ax1.legend(loc='upper right')
        
        # Bottom: Memory usage
        ax2 = axes[1]
        ax2.plot(indices, self.data.memory_percent, color=self.colors['secondary'],
                linewidth=1.5, alpha=0.8)
        ax2.fill_between(indices, self.data.memory_percent, alpha=0.3, color=self.colors['secondary'])
        ax2.set_xlabel('Sample Index', fontsize=12)
        ax2.set_ylabel('Memory %', fontsize=12)
        ax2.set_ylim(0, 100)
        
        if self.data.memory_percent:
            avg_mem = sum(self.data.memory_percent) / len(self.data.memory_percent)
            ax2.axhline(y=avg_mem, color=self.colors['danger'], linestyle='--',
                       label=f'Average: {avg_mem:.1f}%')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'system_health.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None

    def generate_event_timeline(self, save: bool = True) -> Optional[Path]:
        """Generate timeline of all events."""
        if not HAS_MATPLOTLIB or not self.data.events:
            return None
            
        self._setup_style()
        
        # Group events by name
        event_groups: Dict[str, List[Dict]] = defaultdict(list)
        for event in self.data.events:
            name = event.get('name', 'unknown')
            event_groups[name].append(event)
        
        if not event_groups:
            return None
            
        fig, ax = plt.subplots(figsize=(14, max(4, len(event_groups) * 0.5)))
        
        colors_list = list(self.colors.values())
        event_names = list(event_groups.keys())
        
        for i, (event_name, events) in enumerate(event_groups.items()):
            timestamps = [e.get('ts_ms', 0) for e in events]
            y_vals = [i] * len(timestamps)
            color = colors_list[i % len(colors_list)]
            ax.scatter(timestamps, y_vals, c=color, s=50, alpha=0.7, label=event_name)
        
        ax.set_yticks(range(len(event_names)))
        ax.set_yticklabels(event_names)
        ax.set_xlabel('Timestamp (ms)', fontsize=12)
        ax.set_title('📅 Event Timeline', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'event_timeline.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None

    def generate_error_summary(self, save: bool = True) -> Optional[Path]:
        """Generate error summary visualization."""
        if not HAS_MATPLOTLIB or not self.data.errors:
            return None
        
        self._setup_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        error_types = Counter(e.get('error_type', 'Unknown') for e in self.data.errors)
        
        if not error_types:
            return None
        
        types = list(error_types.keys())
        counts = list(error_types.values())
        
        colors_list = [self.colors['danger']] * len(types)
        
        bars = ax.barh(types, counts, color=colors_list, alpha=0.8)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   str(count), va='center', fontsize=11)
        
        ax.set_xlabel('Error Count', fontsize=12)
        ax.set_title('❌ Error Summary', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'error_summary.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None
    
    def generate_dashboard(self, save: bool = True) -> Optional[Path]:
        """Generate a comprehensive dashboard with all metrics."""
        if not HAS_MATPLOTLIB or not self.data.frames:
            return None
            
        self._setup_style()
        
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25)
        
        # Title
        run_id = self.data.meta.get('run_id', 'Unknown')
        model = self.data.meta.get('model', 'Unknown')
        fig.suptitle(f'📊 Smart Glasses Telemetry Dashboard\nRun: {run_id} | Model: {model}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        frames = self.data.frame_indices
        
        # Row 1: FPS, Latency, Detection Count
        # 1. FPS over time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(frames, self.data.fps_values, color=self.colors['primary'], linewidth=1)
        avg_fps = sum(self.data.fps_values) / len(self.data.fps_values) if self.data.fps_values else 0
        ax1.axhline(y=avg_fps, color=self.colors['danger'], linestyle='--', label=f'Avg: {avg_fps:.1f}')
        ax1.set_title('FPS Over Time', fontweight='bold')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('FPS')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_ylim(bottom=0)
        
        # 2. Latency
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(frames, self.data.loop_total_ms, color=self.colors['purple'], linewidth=1, alpha=0.8)
        ax2.axhline(y=33.3, color=self.colors['warning'], linestyle=':', label='30 FPS target')
        avg_latency = sum(self.data.loop_total_ms) / len(self.data.loop_total_ms) if self.data.loop_total_ms else 0
        ax2.axhline(y=avg_latency, color=self.colors['danger'], linestyle='--', label=f'Avg: {avg_latency:.1f}ms')
        ax2.set_title('Loop Latency', fontweight='bold')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Time (ms)')
        ax2.legend(loc='upper right', fontsize=8)
        
        # 3. Detection count
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(frames, self.data.n_detections, color=self.colors['secondary'], alpha=0.7, width=1)
        avg_det = sum(self.data.n_detections) / len(self.data.n_detections) if self.data.n_detections else 0
        ax3.axhline(y=avg_det, color=self.colors['danger'], linestyle='--', label=f'Avg: {avg_det:.1f}')
        ax3.set_title('Detections Per Frame', fontweight='bold')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Count')
        ax3.legend(loc='upper right', fontsize=8)
        
        # Row 2: Confidence, Label frequency (spans 2 cols)
        # 4. Confidence distribution
        ax4 = fig.add_subplot(gs[1, 0])
        confs = [c for c in self.data.top_confs if c > 0]
        if confs:
            ax4.hist(confs, bins=20, range=(0, 1), color=self.colors['primary'], alpha=0.7, edgecolor='white')
            avg_conf = sum(confs) / len(confs)
            ax4.axvline(x=avg_conf, color=self.colors['danger'], linestyle='--', label=f'Avg: {avg_conf:.2f}')
        ax4.set_title('Confidence Distribution', fontweight='bold')
        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Frequency')
        ax4.legend(loc='upper right', fontsize=8)
        
        # 5. Label frequency (spans 2 columns)
        ax5 = fig.add_subplot(gs[1, 1:])
        label_counts = Counter(self.data.all_labels)
        top_labels = label_counts.most_common(10)
        if top_labels:
            labels = [l[0] for l in top_labels]
            counts = [l[1] for l in top_labels]
            bars = ax5.barh(range(len(labels)), counts, color=self.colors['orange'], alpha=0.8)
            ax5.set_yticks(range(len(labels)))
            ax5.set_yticklabels(labels)
            ax5.invert_yaxis()
            for bar, count in zip(bars, counts):
                ax5.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                        f'{count:,}', va='center', fontsize=9)
        ax5.set_title('Top 10 Detected Objects', fontweight='bold')
        ax5.set_xlabel('Count')
        
        # Row 3: Voice, AI, Safety
        # 6. Voice timing (if data exists)
        ax6 = fig.add_subplot(gs[2, 0])
        if self.data.voice_total_ms:
            ax6.hist(self.data.voice_total_ms, bins=15, color=self.colors['cyan'], alpha=0.7, edgecolor='white')
            avg_voice = sum(self.data.voice_total_ms) / len(self.data.voice_total_ms)
            ax6.axvline(x=avg_voice, color=self.colors['danger'], linestyle='--', label=f'Avg: {avg_voice:.0f}ms')
            ax6.legend(loc='upper right', fontsize=8)
        ax6.set_title('Voice Response Time', fontweight='bold')
        ax6.set_xlabel('Time (ms)')
        ax6.set_ylabel('Frequency')
        
        # 7. AI latency (if data exists)
        ax7 = fig.add_subplot(gs[2, 1])
        if self.data.ai_latencies:
            ax7.hist(self.data.ai_latencies, bins=15, color=self.colors['purple'], alpha=0.7, edgecolor='white')
            avg_ai = sum(self.data.ai_latencies) / len(self.data.ai_latencies)
            ax7.axvline(x=avg_ai, color=self.colors['danger'], linestyle='--', label=f'Avg: {avg_ai:.0f}ms')
            ax7.legend(loc='upper right', fontsize=8)
        ax7.set_title('AI Operation Latency', fontweight='bold')
        ax7.set_xlabel('Latency (ms)')
        ax7.set_ylabel('Frequency')
        
        # 8. Safety severity (if data exists)
        ax8 = fig.add_subplot(gs[2, 2])
        if self.data.safety_severities:
            severity_counts = Counter(self.data.safety_severities)
            severity_labels = ['Info', 'Near', 'Close', 'Danger']
            severity_values = [severity_counts.get(i, 0) for i in range(4)]
            colors_list = [self.colors['neutral'], self.colors['warning'],
                          self.colors['orange'], self.colors['danger']]
            ax8.bar(severity_labels, severity_values, color=colors_list, alpha=0.8)
        ax8.set_title('Safety Event Severity', fontweight='bold')
        ax8.set_xlabel('Severity Level')
        ax8.set_ylabel('Count')
        
        # Row 4: System health + summary stats
        # 9. CPU/Memory (if data exists)
        ax9 = fig.add_subplot(gs[3, 0:2])
        if self.data.cpu_percent:
            indices = list(range(len(self.data.cpu_percent)))
            ax9.plot(indices, self.data.cpu_percent, color=self.colors['primary'], 
                    linewidth=1.5, label='CPU %', alpha=0.8)
            ax9.plot(indices, self.data.memory_percent, color=self.colors['secondary'],
                    linewidth=1.5, label='Memory %', alpha=0.8)
            ax9.legend(loc='upper right', fontsize=8)
            ax9.set_ylim(0, 100)
        ax9.set_title('System Resources', fontweight='bold')
        ax9.set_xlabel('Sample')
        ax9.set_ylabel('Usage %')
        
        # 10. Summary stats
        ax10 = fig.add_subplot(gs[3, 2])
        ax10.axis('off')
        
        stats_text = f"""
Summary Statistics
──────────────────
Total Frames: {len(self.data.frames):,}
Total Events: {len(self.data.events):,}
Voice Interactions: {len(self.data.voice):,}
Safety Events: {len(self.data.safety):,}
AI Operations: {len(self.data.ai):,}
Errors: {len(self.data.errors):,}

Performance
──────────────────
Avg FPS: {avg_fps:.1f}
Avg Latency: {avg_latency:.1f}ms
Avg Detections: {avg_det:.1f}
Unique Objects: {len(Counter(self.data.all_labels)):,}
"""
        ax10.text(0.1, 0.9, stats_text, transform=ax10.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            path = self.output_dir / 'dashboard.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        return None
    
    def generate_all(self) -> Dict[str, Optional[Path]]:
        """Generate all available graphs."""
        results = {}
        
        print("📊 Generating graphs...")
        
        results['fps'] = self.generate_fps_graph()
        if results['fps']:
            print(f"  ✓ FPS graph: {results['fps']}")
            
        results['latency'] = self.generate_latency_graph()
        if results['latency']:
            print(f"  ✓ Latency graph: {results['latency']}")
            
        results['detection_count'] = self.generate_detection_count_graph()
        if results['detection_count']:
            print(f"  ✓ Detection count: {results['detection_count']}")
            
        results['label_frequency'] = self.generate_label_frequency_graph()
        if results['label_frequency']:
            print(f"  ✓ Label frequency: {results['label_frequency']}")
            
        results['confidence'] = self.generate_confidence_distribution()
        if results['confidence']:
            print(f"  ✓ Confidence distribution: {results['confidence']}")

        results['voice_timing'] = self.generate_voice_timing_graph()
        if results['voice_timing']:
            print(f"  ✓ Voice timing: {results['voice_timing']}")

        results['safety_analysis'] = self.generate_safety_analysis_graph()
        if results['safety_analysis']:
            print(f"  ✓ Safety analysis: {results['safety_analysis']}")

        results['ai_performance'] = self.generate_ai_performance_graph()
        if results['ai_performance']:
            print(f"  ✓ AI performance: {results['ai_performance']}")

        results['system_health'] = self.generate_system_health_graph()
        if results['system_health']:
            print(f"  ✓ System health: {results['system_health']}")
            
        results['events'] = self.generate_event_timeline()
        if results['events']:
            print(f"  ✓ Event timeline: {results['events']}")

        results['errors'] = self.generate_error_summary()
        if results['errors']:
            print(f"  ✓ Error summary: {results['errors']}")
            
        results['dashboard'] = self.generate_dashboard()
        if results['dashboard']:
            print(f"  ✓ Dashboard: {results['dashboard']}")
        
        return results


def generate_summary_stats(data: TelemetryData) -> Dict[str, Any]:
    """Generate summary statistics from telemetry data."""
    stats = {
        'run_id': data.meta.get('run_id', 'unknown'),
        'model': data.meta.get('model', 'unknown'),
        'total_frames': len(data.frames),
        'total_events': len(data.events),
        'total_voice': len(data.voice),
        'total_safety': len(data.safety),
        'total_ai': len(data.ai),
        'total_errors': len(data.errors),
    }
    
    if data.fps_values:
        stats['fps_avg'] = sum(data.fps_values) / len(data.fps_values)
        stats['fps_min'] = min(data.fps_values)
        stats['fps_max'] = max(data.fps_values)
    
    if data.loop_total_ms:
        stats['latency_avg_ms'] = sum(data.loop_total_ms) / len(data.loop_total_ms)
        stats['latency_max_ms'] = max(data.loop_total_ms)
    
    if data.n_detections:
        stats['detections_avg'] = sum(data.n_detections) / len(data.n_detections)
        stats['detections_total'] = sum(data.n_detections)
    
    confs = [c for c in data.top_confs if c > 0]
    if confs:
        stats['confidence_avg'] = sum(confs) / len(confs)
    
    if data.all_labels:
        label_counts = Counter(data.all_labels)
        stats['unique_labels'] = len(label_counts)
        stats['top_5_labels'] = label_counts.most_common(5)
    
    if data.voice_total_ms:
        stats['voice_avg_ms'] = sum(data.voice_total_ms) / len(data.voice_total_ms)
    
    if data.ai_latencies:
        stats['ai_avg_ms'] = sum(data.ai_latencies) / len(data.ai_latencies)
    
    return stats


def print_summary(stats: Dict[str, Any]):
    """Print formatted summary statistics."""
    print("\n" + "="*60)
    print("📊 TELEMETRY SUMMARY")
    print("="*60)
    print(f"Run ID: {stats.get('run_id', 'N/A')}")
    print(f"Model: {stats.get('model', 'N/A')}")
    print(f"Total Frames: {stats.get('total_frames', 0):,}")
    print(f"Total Events: {stats.get('total_events', 0):,}")
    print()
    
    print("📈 Activity Counts:")
    print(f"   Voice Interactions: {stats.get('total_voice', 0):,}")
    print(f"   Safety Events: {stats.get('total_safety', 0):,}")
    print(f"   AI Operations: {stats.get('total_ai', 0):,}")
    print(f"   Errors: {stats.get('total_errors', 0):,}")
    print()
    
    if 'fps_avg' in stats:
        print("⏱️  Performance:")
        print(f"   FPS: {stats['fps_avg']:.1f} avg | {stats.get('fps_min', 0):.1f} min | {stats.get('fps_max', 0):.1f} max")
    
    if 'latency_avg_ms' in stats:
        print(f"   Latency: {stats['latency_avg_ms']:.1f}ms avg | {stats.get('latency_max_ms', 0):.1f}ms max")
    
    if 'voice_avg_ms' in stats:
        print(f"   Voice Response: {stats['voice_avg_ms']:.0f}ms avg")
    
    if 'ai_avg_ms' in stats:
        print(f"   AI Latency: {stats['ai_avg_ms']:.0f}ms avg")
    
    print()
    if 'detections_avg' in stats:
        print("🎯 Detections:")
        print(f"   Per frame: {stats['detections_avg']:.1f} avg")
        print(f"   Total: {stats.get('detections_total', 0):,}")
    
    if 'confidence_avg' in stats:
        print(f"   Confidence: {stats['confidence_avg']:.2f} avg")
    
    if 'unique_labels' in stats:
        print(f"   Unique objects: {stats['unique_labels']}")
        print("   Top 5:")
        for label, count in stats.get('top_5_labels', []):
            print(f"      • {label}: {count:,}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Telemetry Visualization Tool')
    parser.add_argument('file', nargs='?', help='Path to JSONL telemetry file')
    parser.add_argument('--latest', action='store_true', help='Use the latest telemetry file')
    parser.add_argument('--all', action='store_true', help='Process all telemetry files')
    parser.add_argument('--output', '-o', type=str, default='telemetry/graphs',
                       help='Output directory for graphs')
    parser.add_argument('--stats-only', action='store_true', help='Only print stats, no graphs')
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB and not args.stats_only:
        print("⚠️  matplotlib not installed. Install with: pip install matplotlib")
        print("   Running in stats-only mode...")
        args.stats_only = True
    
    files_to_process: List[Path] = []
    
    if args.all:
        files_to_process = get_all_telemetry_files()
        if not files_to_process:
            print("❌ No telemetry files found in telemetry/runs/")
            return
    elif args.latest or args.file is None:
        latest = get_latest_telemetry_file()
        if latest is None:
            print("❌ No telemetry files found in telemetry/runs/")
            return
        files_to_process = [latest]
    else:
        path = Path(args.file)
        if not path.exists():
            print(f"❌ File not found: {args.file}")
            return
        files_to_process = [path]
    
    print(f"📁 Processing {len(files_to_process)} telemetry file(s)...\n")
    
    for file_path in files_to_process:
        print(f"📄 Loading: {file_path}")
        data = load_telemetry(file_path)
        
        stats = generate_summary_stats(data)
        print_summary(stats)
        
        if not args.stats_only:
            # Create run-specific output directory
            run_id = data.meta.get('run_id', file_path.stem)
            output_dir = Path(args.output) / run_id
            
            viz = TelemetryVisualizer(data, output_dir)
            results = viz.generate_all()
            
            successful = sum(1 for v in results.values() if v is not None)
            print(f"\n✅ Generated {successful} graphs in {output_dir}/")
        
        print()


if __name__ == '__main__':
    main()