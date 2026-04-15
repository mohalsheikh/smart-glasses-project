# src/utils/telemetry_dashboard.py
"""
Enhanced Interactive HTML Dashboard Generator for Smart Glasses Telemetry

Generates a standalone HTML file with Plotly.js charts - no server required.
Includes all metrics: frames, voice, AI, safety, speech, system health, errors.

Usage:
    python -m src.utils.telemetry_dashboard telemetry/runs/run_YYYYMMDD_HHMMSS.jsonl
    python -m src.utils.telemetry_dashboard --latest
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict
from datetime import datetime

# Import from telemetry_viz for data loading
try:
    from src.utils.telemetry_viz import (
        TelemetryData,
        load_telemetry,
        get_latest_telemetry_file,
    )
except ImportError:
    # If running standalone, define minimal versions
    from dataclasses import dataclass, field
    
    @dataclass
    class TelemetryData:
        meta: Dict[str, Any] = field(default_factory=dict)
        frames: List[Dict[str, Any]] = field(default_factory=list)
        events: List[Dict[str, Any]] = field(default_factory=list)
        voice: List[Dict[str, Any]] = field(default_factory=list)
        safety: List[Dict[str, Any]] = field(default_factory=list)
        ai: List[Dict[str, Any]] = field(default_factory=list)
        speech: List[Dict[str, Any]] = field(default_factory=list)
        system: List[Dict[str, Any]] = field(default_factory=list)
        errors: List[Dict[str, Any]] = field(default_factory=list)
        frame_indices: List[int] = field(default_factory=list)
        fps_values: List[float] = field(default_factory=list)
        capture_ms: List[float] = field(default_factory=list)
        detect_ms: List[float] = field(default_factory=list)
        loop_total_ms: List[float] = field(default_factory=list)
        n_detections: List[int] = field(default_factory=list)
        top_labels: List[str] = field(default_factory=list)
        top_confs: List[float] = field(default_factory=list)
        timestamps_ms: List[int] = field(default_factory=list)
        all_labels: List[str] = field(default_factory=list)
        voice_listen_ms: List[float] = field(default_factory=list)
        voice_process_ms: List[float] = field(default_factory=list)
        voice_total_ms: List[float] = field(default_factory=list)
        safety_severities: List[int] = field(default_factory=list)
        safety_types: List[str] = field(default_factory=list)
        ai_latencies: List[float] = field(default_factory=list)
        ai_operations: List[str] = field(default_factory=list)
        cpu_percent: List[float] = field(default_factory=list)
        memory_percent: List[float] = field(default_factory=list)


def generate_html_dashboard(data: TelemetryData, output_path: Path) -> Path:
    """Generate an interactive HTML dashboard from telemetry data."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract data for charts
    run_id = data.meta.get('run_id', 'unknown')
    model = data.meta.get('model', 'unknown')
    
    # Calculate summary stats
    total_frames = len(data.frames)
    avg_fps = sum(data.fps_values) / len(data.fps_values) if data.fps_values else 0
    avg_latency = sum(data.loop_total_ms) / len(data.loop_total_ms) if data.loop_total_ms else 0
    avg_detections = sum(data.n_detections) / len(data.n_detections) if data.n_detections else 0
    
    confs = [c for c in data.top_confs if c > 0]
    avg_confidence = sum(confs) / len(confs) if confs else 0
    
    label_counts = Counter(data.all_labels)
    unique_labels = len(label_counts)
    top_labels = label_counts.most_common(10)
    
    # Voice stats
    total_voice = len(data.voice)
    avg_voice_time = sum(data.voice_total_ms) / len(data.voice_total_ms) if data.voice_total_ms else 0
    
    # Safety stats
    total_safety = len(data.safety)
    
    # AI stats
    total_ai = len(data.ai)
    avg_ai_latency = sum(data.ai_latencies) / len(data.ai_latencies) if data.ai_latencies else 0
    
    # System stats
    avg_cpu = sum(data.cpu_percent) / len(data.cpu_percent) if data.cpu_percent else 0
    avg_memory = sum(data.memory_percent) / len(data.memory_percent) if data.memory_percent else 0
    
    # Error stats
    total_errors = len(data.errors)
    error_types = Counter(e.get('error_type', 'Unknown') for e in data.errors)
    
    # Speech stats
    total_speech = len(data.speech)
    speech_sources = Counter(s.get('source', 'unknown') for s in data.speech)
    
    # Prepare chart data
    frames_json = json.dumps(data.frame_indices)
    fps_json = json.dumps(data.fps_values)
    latency_json = json.dumps(data.loop_total_ms)
    capture_json = json.dumps(data.capture_ms)
    detect_json = json.dumps(data.detect_ms)
    detections_json = json.dumps(data.n_detections)
    confidence_json = json.dumps([c for c in data.top_confs if c > 0])
    
    top_labels_names = json.dumps([l[0] for l in top_labels])
    top_labels_counts = json.dumps([l[1] for l in top_labels])
    
    # Voice timing data
    voice_total_json = json.dumps(data.voice_total_ms)
    voice_listen_json = json.dumps(data.voice_listen_ms)
    
    # AI operation data
    ai_by_operation = defaultdict(list)
    for a in data.ai:
        op = a.get('operation', 'unknown')
        lat = a.get('latency_ms', 0)
        ai_by_operation[op].append(lat)
    
    ai_ops_names = json.dumps(list(ai_by_operation.keys()))
    ai_ops_avg = json.dumps([sum(v)/len(v) if v else 0 for v in ai_by_operation.values()])
    ai_latencies_json = json.dumps(data.ai_latencies)
    
    # Safety data
    safety_severity_counts = Counter(data.safety_severities)
    safety_severity_labels = json.dumps(['Info (0)', 'Near (1)', 'Close (2)', 'Danger (3)'])
    safety_severity_values = json.dumps([safety_severity_counts.get(i, 0) for i in range(4)])
    
    safety_type_counts = Counter(data.safety_types)
    safety_type_labels = json.dumps(list(safety_type_counts.keys()) if safety_type_counts else ['No events'])
    safety_type_values = json.dumps(list(safety_type_counts.values()) if safety_type_counts else [0])
    
    # System health data
    system_indices = json.dumps(list(range(len(data.cpu_percent))))
    cpu_json = json.dumps(data.cpu_percent)
    memory_json = json.dumps(data.memory_percent)
    
    # Error data
    error_type_labels = json.dumps(list(error_types.keys()) if error_types else ['No errors'])
    error_type_values = json.dumps(list(error_types.values()) if error_types else [0])
    
    # Speech source data
    speech_source_labels = json.dumps(list(speech_sources.keys()) if speech_sources else ['No speech'])
    speech_source_values = json.dumps(list(speech_sources.values()) if speech_sources else [0])
    
    # Event timeline data
    event_groups = defaultdict(list)
    for event in data.events:
        name = event.get('name', 'unknown')
        ts = event.get('ts_ms', 0)
        event_groups[name].append(ts)
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Glasses Telemetry Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #94a3b8;
            font-size: 1.1rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: rgba(30, 41, 59, 0.8);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(148, 163, 184, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }}
        
        .stat-card .icon {{
            font-size: 1.8rem;
            margin-bottom: 8px;
        }}
        
        .stat-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #f8fafc;
        }}
        
        .stat-card .label {{
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 4px;
        }}
        
        .stat-card.highlight {{
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
            border-color: rgba(59, 130, 246, 0.3);
        }}
        
        .stat-card.warning {{
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(239, 68, 68, 0.2));
            border-color: rgba(245, 158, 11, 0.3);
        }}
        
        .stat-card.success {{
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.2));
            border-color: rgba(34, 197, 94, 0.3);
        }}
        
        .section-title {{
            font-size: 1.3rem;
            font-weight: 600;
            margin: 30px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(148, 163, 184, 0.2);
            color: #f1f5f9;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .chart-container {{
            background: rgba(30, 41, 59, 0.8);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }}
        
        .chart-container h3 {{
            font-size: 1.1rem;
            margin-bottom: 15px;
            color: #f1f5f9;
        }}
        
        .chart-container.full-width {{
            grid-column: 1 / -1;
        }}
        
        .chart-container.small {{
            min-width: 300px;
        }}
        
        .small-charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .no-data {{
            text-align: center;
            padding: 40px;
            color: #64748b;
            font-style: italic;
        }}
        
        .no-data .icon {{
            font-size: 3rem;
            margin-bottom: 10px;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(148, 163, 184, 0.1);
            color: #64748b;
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            .small-charts-grid {{
                grid-template-columns: 1fr;
            }}
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Smart Glasses Telemetry</h1>
        <p class="subtitle">Run: {run_id} | Model: {model}</p>
    </div>
    
    <!-- Primary Stats -->
    <div class="stats-grid">
        <div class="stat-card highlight">
            <div class="icon">🎬</div>
            <div class="value">{total_frames:,}</div>
            <div class="label">Total Frames</div>
        </div>
        <div class="stat-card highlight">
            <div class="icon">⚡</div>
            <div class="value">{avg_fps:.1f}</div>
            <div class="label">Average FPS</div>
        </div>
        <div class="stat-card">
            <div class="icon">⏱️</div>
            <div class="value">{avg_latency:.1f}ms</div>
            <div class="label">Avg Latency</div>
        </div>
        <div class="stat-card">
            <div class="icon">🎯</div>
            <div class="value">{avg_detections:.1f}</div>
            <div class="label">Avg Detections</div>
        </div>
        <div class="stat-card success">
            <div class="icon">📈</div>
            <div class="value">{avg_confidence*100:.0f}%</div>
            <div class="label">Avg Confidence</div>
        </div>
        <div class="stat-card">
            <div class="icon">🏷️</div>
            <div class="value">{unique_labels}</div>
            <div class="label">Unique Objects</div>
        </div>
    </div>
    
    <!-- Activity Stats -->
    <div class="stats-grid">
        <div class="stat-card {'success' if total_voice > 0 else ''}">
            <div class="icon">🎤</div>
            <div class="value">{total_voice}</div>
            <div class="label">Voice Interactions</div>
        </div>
        <div class="stat-card {'success' if total_ai > 0 else ''}">
            <div class="icon">🤖</div>
            <div class="value">{total_ai}</div>
            <div class="label">AI Operations</div>
        </div>
        <div class="stat-card {'warning' if total_safety > 0 else ''}">
            <div class="icon">🚨</div>
            <div class="value">{total_safety}</div>
            <div class="label">Safety Events</div>
        </div>
        <div class="stat-card">
            <div class="icon">🔊</div>
            <div class="value">{total_speech}</div>
            <div class="label">Speech Outputs</div>
        </div>
        <div class="stat-card {'warning' if total_errors > 0 else 'success'}">
            <div class="icon">{'❌' if total_errors > 0 else '✅'}</div>
            <div class="value">{total_errors}</div>
            <div class="label">Errors</div>
        </div>
        <div class="stat-card">
            <div class="icon">💻</div>
            <div class="value">{avg_cpu:.1f}%</div>
            <div class="label">Avg CPU</div>
        </div>
    </div>
    
    <!-- Frame Performance Section -->
    <h2 class="section-title">📊 Frame Performance</h2>
    <div class="charts-grid">
        <div class="chart-container">
            <h3>📈 FPS Over Time</h3>
            <div id="fps-chart"></div>
        </div>
        <div class="chart-container">
            <h3>⏱️ Processing Latency</h3>
            <div id="latency-chart"></div>
        </div>
        <div class="chart-container">
            <h3>🎯 Detections Per Frame</h3>
            <div id="detections-chart"></div>
        </div>
        <div class="chart-container">
            <h3>📊 Confidence Distribution</h3>
            <div id="confidence-chart"></div>
        </div>
    </div>
    
    <div class="chart-container full-width" style="margin-bottom: 30px;">
        <h3>🏷️ Top Detected Objects</h3>
        <div id="labels-chart"></div>
    </div>
    
    <!-- Voice & AI Section -->
    <h2 class="section-title">🎤 Voice & AI Operations</h2>
    <div class="small-charts-grid">
        <div class="chart-container">
            <h3>🎤 Voice Response Time</h3>
            <div id="voice-chart"></div>
        </div>
        <div class="chart-container">
            <h3>🤖 AI Operation Latency</h3>
            <div id="ai-latency-chart"></div>
        </div>
        <div class="chart-container">
            <h3>📊 AI Operations by Type</h3>
            <div id="ai-ops-chart"></div>
        </div>
        <div class="chart-container">
            <h3>🔊 Speech by Source</h3>
            <div id="speech-chart"></div>
        </div>
    </div>
    
    <!-- Safety Section -->
    <h2 class="section-title">🚨 Safety & Guidance</h2>
    <div class="small-charts-grid">
        <div class="chart-container">
            <h3>⚠️ Safety Event Severity</h3>
            <div id="safety-severity-chart"></div>
        </div>
        <div class="chart-container">
            <h3>📋 Safety Event Types</h3>
            <div id="safety-types-chart"></div>
        </div>
    </div>
    
    <!-- System Health Section -->
    <h2 class="section-title">💻 System Health</h2>
    <div class="charts-grid">
        <div class="chart-container full-width">
            <h3>📈 CPU & Memory Usage</h3>
            <div id="system-chart"></div>
        </div>
    </div>
    
    <!-- Errors Section -->
    <h2 class="section-title">❌ Errors</h2>
    <div class="small-charts-grid">
        <div class="chart-container">
            <h3>🐛 Error Types</h3>
            <div id="errors-chart"></div>
        </div>
    </div>
    
    <div class="footer">
        Generated by Smart Glasses Telemetry System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    
    <script>
        const darkLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#e2e8f0' }},
            xaxis: {{
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                zerolinecolor: 'rgba(148, 163, 184, 0.2)'
            }},
            yaxis: {{
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                zerolinecolor: 'rgba(148, 163, 184, 0.2)'
            }},
            margin: {{ t: 30, r: 30, b: 50, l: 60 }}
        }};
        
        const config = {{ responsive: true, displayModeBar: false }};
        
        // FPS Chart
        const fpsData = {{
            x: {frames_json},
            y: {fps_json},
            type: 'scatter',
            mode: 'lines',
            name: 'FPS',
            line: {{ color: '#3b82f6', width: 1.5 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.1)'
        }};
        
        const fpsAvgLine = {{
            x: {frames_json},
            y: Array({len(data.frame_indices)}).fill({avg_fps}),
            type: 'scatter',
            mode: 'lines',
            name: 'Average',
            line: {{ color: '#ef4444', width: 2, dash: 'dash' }}
        }};
        
        Plotly.newPlot('fps-chart', [fpsData, fpsAvgLine], {{
            ...darkLayout,
            xaxis: {{ ...darkLayout.xaxis, title: 'Frame' }},
            yaxis: {{ ...darkLayout.yaxis, title: 'FPS', rangemode: 'tozero' }},
            showlegend: true,
            legend: {{ x: 1, xanchor: 'right', y: 1 }}
        }}, config);
        
        // Latency Chart
        const latencyData = {{
            x: {frames_json},
            y: {latency_json},
            type: 'scatter',
            mode: 'lines',
            name: 'Loop Time',
            line: {{ color: '#8b5cf6', width: 1.5 }}
        }};
        
        const targetLine = {{
            x: {frames_json},
            y: Array({len(data.frame_indices)}).fill(33.3),
            type: 'scatter',
            mode: 'lines',
            name: '30 FPS Target',
            line: {{ color: '#f59e0b', width: 2, dash: 'dot' }}
        }};
        
        Plotly.newPlot('latency-chart', [latencyData, targetLine], {{
            ...darkLayout,
            xaxis: {{ ...darkLayout.xaxis, title: 'Frame' }},
            yaxis: {{ ...darkLayout.yaxis, title: 'Time (ms)' }},
            showlegend: true,
            legend: {{ x: 1, xanchor: 'right', y: 1 }}
        }}, config);
        
        // Detections Chart
        const detectionsData = {{
            x: {frames_json},
            y: {detections_json},
            type: 'bar',
            name: 'Detections',
            marker: {{ color: '#22c55e', opacity: 0.7 }}
        }};
        
        Plotly.newPlot('detections-chart', [detectionsData], {{
            ...darkLayout,
            xaxis: {{ ...darkLayout.xaxis, title: 'Frame' }},
            yaxis: {{ ...darkLayout.yaxis, title: 'Count', rangemode: 'tozero' }},
            bargap: 0
        }}, config);
        
        // Confidence Distribution
        const confidenceData = {{
            x: {confidence_json},
            type: 'histogram',
            nbinsx: 20,
            marker: {{ color: '#3b82f6', opacity: 0.7 }}
        }};
        
        Plotly.newPlot('confidence-chart', [confidenceData], {{
            ...darkLayout,
            xaxis: {{ ...darkLayout.xaxis, title: 'Confidence', range: [0, 1] }},
            yaxis: {{ ...darkLayout.yaxis, title: 'Frequency' }}
        }}, config);
        
        // Top Labels Chart
        const labelsData = {{
            x: {top_labels_counts},
            y: {top_labels_names},
            type: 'bar',
            orientation: 'h',
            marker: {{ 
                color: {top_labels_counts},
                colorscale: 'Blues',
                reversescale: true
            }},
            text: {top_labels_counts},
            textposition: 'outside'
        }};
        
        Plotly.newPlot('labels-chart', [labelsData], {{
            ...darkLayout,
            xaxis: {{ ...darkLayout.xaxis, title: 'Count' }},
            yaxis: {{ ...darkLayout.yaxis, automargin: true }},
            height: 300
        }}, config);
        
        // Voice Response Time Chart
        const voiceData = {voice_total_json};
        if (voiceData.length > 0) {{
            Plotly.newPlot('voice-chart', [{{
                x: voiceData,
                type: 'histogram',
                nbinsx: 15,
                marker: {{ color: '#06b6d4', opacity: 0.7 }}
            }}], {{
                ...darkLayout,
                xaxis: {{ ...darkLayout.xaxis, title: 'Response Time (ms)' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Frequency' }}
            }}, config);
        }} else {{
            document.getElementById('voice-chart').innerHTML = '<div class="no-data"><div class="icon">🎤</div><p>No voice interactions recorded.<br>Press "v" during a run to use voice.</p></div>';
        }}
        
        // AI Latency Distribution Chart
        const aiLatencyData = {ai_latencies_json};
        if (aiLatencyData.length > 0) {{
            Plotly.newPlot('ai-latency-chart', [{{
                x: aiLatencyData,
                type: 'histogram',
                nbinsx: 15,
                marker: {{ color: '#a855f7', opacity: 0.7 }}
            }}], {{
                ...darkLayout,
                xaxis: {{ ...darkLayout.xaxis, title: 'Latency (ms)' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Frequency' }}
            }}, config);
        }} else {{
            document.getElementById('ai-latency-chart').innerHTML = '<div class="no-data"><div class="icon">🤖</div><p>No AI operations recorded.<br>Press "d" to describe or "r" to read.</p></div>';
        }}
        
        // AI Operations by Type Chart
        const aiOpsNames = {ai_ops_names};
        const aiOpsAvg = {ai_ops_avg};
        if (aiOpsNames.length > 0 && aiOpsAvg.some(v => v > 0)) {{
            Plotly.newPlot('ai-ops-chart', [{{
                x: aiOpsNames,
                y: aiOpsAvg,
                type: 'bar',
                marker: {{ 
                    color: ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6'].slice(0, aiOpsNames.length)
                }},
                text: aiOpsAvg.map(v => v.toFixed(0) + 'ms'),
                textposition: 'outside'
            }}], {{
                ...darkLayout,
                xaxis: {{ ...darkLayout.xaxis, title: 'Operation' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Avg Latency (ms)' }}
            }}, config);
        }} else {{
            document.getElementById('ai-ops-chart').innerHTML = '<div class="no-data"><div class="icon">📊</div><p>No AI operation data available.</p></div>';
        }}
        
        // Speech by Source Chart
        const speechLabels = {speech_source_labels};
        const speechValues = {speech_source_values};
        if (speechValues.some(v => v > 0)) {{
            Plotly.newPlot('speech-chart', [{{
                labels: speechLabels,
                values: speechValues,
                type: 'pie',
                marker: {{ colors: ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'] }},
                textinfo: 'label+value',
                hole: 0.4
            }}], {{
                ...darkLayout,
                showlegend: false
            }}, config);
        }} else {{
            document.getElementById('speech-chart').innerHTML = '<div class="no-data"><div class="icon">🔊</div><p>No speech output recorded.</p></div>';
        }}
        
        // Safety Severity Chart
        const safetySeverityValues = {safety_severity_values};
        if (safetySeverityValues.some(v => v > 0)) {{
            Plotly.newPlot('safety-severity-chart', [{{
                x: {safety_severity_labels},
                y: safetySeverityValues,
                type: 'bar',
                marker: {{ 
                    color: ['#6b7280', '#f59e0b', '#f97316', '#ef4444']
                }},
                text: safetySeverityValues,
                textposition: 'outside'
            }}], {{
                ...darkLayout,
                xaxis: {{ ...darkLayout.xaxis, title: 'Severity Level' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Count' }}
            }}, config);
        }} else {{
            document.getElementById('safety-severity-chart').innerHTML = '<div class="no-data"><div class="icon">✅</div><p>No safety events recorded.<br>Enable OBSTACLE_ENABLED or GUIDANCE_ENABLED in config.</p></div>';
        }}
        
        // Safety Types Chart
        const safetyTypeValues = {safety_type_values};
        if (safetyTypeValues.some(v => v > 0)) {{
            Plotly.newPlot('safety-types-chart', [{{
                labels: {safety_type_labels},
                values: safetyTypeValues,
                type: 'pie',
                marker: {{ colors: ['#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'] }},
                textinfo: 'label+percent'
            }}], {{
                ...darkLayout,
                showlegend: true,
                legend: {{ x: 1, xanchor: 'right', y: 0.5 }}
            }}, config);
        }} else {{
            document.getElementById('safety-types-chart').innerHTML = '<div class="no-data"><div class="icon">🛡️</div><p>No safety events by type.</p></div>';
        }}
        
        // System Health Chart
        const cpuData = {cpu_json};
        const memoryData = {memory_json};
        if (cpuData.length > 0 || memoryData.length > 0) {{
            const traces = [];
            if (cpuData.length > 0) {{
                traces.push({{
                    x: {system_indices},
                    y: cpuData,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'CPU %',
                    line: {{ color: '#3b82f6', width: 2 }},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(59, 130, 246, 0.1)'
                }});
            }}
            if (memoryData.length > 0) {{
                traces.push({{
                    x: {system_indices},
                    y: memoryData,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Memory %',
                    line: {{ color: '#22c55e', width: 2 }},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(34, 197, 94, 0.1)'
                }});
            }}
            Plotly.newPlot('system-chart', traces, {{
                ...darkLayout,
                xaxis: {{ ...darkLayout.xaxis, title: 'Sample' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Usage %', range: [0, 100] }},
                showlegend: true,
                legend: {{ x: 1, xanchor: 'right', y: 1 }}
            }}, config);
        }} else {{
            document.getElementById('system-chart').innerHTML = '<div class="no-data"><div class="icon">💻</div><p>No system health data recorded.<br>Install psutil: pip install psutil</p></div>';
        }}
        
        // Errors Chart
        const errorValues = {error_type_values};
        if (errorValues.some(v => v > 0)) {{
            Plotly.newPlot('errors-chart', [{{
                x: {error_type_labels},
                y: errorValues,
                type: 'bar',
                marker: {{ color: '#ef4444', opacity: 0.8 }},
                text: errorValues,
                textposition: 'outside'
            }}], {{
                ...darkLayout,
                xaxis: {{ ...darkLayout.xaxis, title: 'Error Type' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Count' }}
            }}, config);
        }} else {{
            document.getElementById('errors-chart').innerHTML = '<div class="no-data"><div class="icon">✅</div><p>No errors recorded. Great job!</p></div>';
        }}
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate Interactive HTML Dashboard')
    parser.add_argument('file', nargs='?', help='Path to JSONL telemetry file')
    parser.add_argument('--latest', action='store_true', help='Use the latest telemetry file')
    parser.add_argument('--open', action='store_true', help='Open in browser after generation')
    
    args = parser.parse_args()
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {args.file}")
            return
    else:
        file_path = get_latest_telemetry_file()
        if file_path is None:
            print("❌ No telemetry files found in telemetry/runs/")
            return
    
    print(f"📄 Loading: {file_path}")
    data = load_telemetry(file_path)
    
    run_id = data.meta.get('run_id', file_path.stem)
    output_path = Path('telemetry/dashboards') / f'{run_id}_dashboard.html'
    
    result = generate_html_dashboard(data, output_path)
    print(f"✅ Generated: {result}")
    
    if args.open:
        import webbrowser
        url = f"file://{result.absolute()}"
        print(f"🌐 Opening in browser: {url}")
        webbrowser.open(url)


if __name__ == '__main__':
    main()