#!/usr/bin/env python3
"""
Generate Telemetry Graphs and Dashboard

Quick script to visualize your telemetry data.

Usage:
    python generate_graphs.py              # Process latest run
    python generate_graphs.py --all        # Process all runs
    python generate_graphs.py --open       # Process and open in browser
    python generate_graphs.py path/to.jsonl  # Process specific file
"""

import sys
import webbrowser
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.telemetry_viz import (
    load_telemetry,
    get_latest_telemetry_file,
    get_all_telemetry_files,
    TelemetryVisualizer,
    generate_summary_stats,
    print_summary,
)
from src.utils.telemetry_dashboard import generate_html_dashboard


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Telemetry Visualizations')
    parser.add_argument('file', nargs='?', help='Path to JSONL telemetry file')
    parser.add_argument('--latest', action='store_true', help='Use the latest telemetry file')
    parser.add_argument('--all', action='store_true', help='Process all telemetry files')
    parser.add_argument('--no-html', action='store_true', help='Skip HTML dashboard generation')
    parser.add_argument('--no-png', action='store_true', help='Skip PNG graph generation')
    parser.add_argument('--open', action='store_true', help='Open dashboard in browser after generation')
    
    args = parser.parse_args()
    
    # Determine which files to process
    files_to_process = []
    
    if args.all:
        files_to_process = get_all_telemetry_files()
        if not files_to_process:
            print("❌ No telemetry files found in telemetry/runs/")
            return
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"❌ File not found: {args.file}")
            return
        files_to_process = [path]
    else:
        # Default to latest
        latest = get_latest_telemetry_file()
        if latest is None:
            print("❌ No telemetry files found in telemetry/runs/")
            print("   Run your smart glasses first to generate telemetry data.")
            return
        files_to_process = [latest]
    
    print(f"📁 Processing {len(files_to_process)} telemetry file(s)...\n")
    print("=" * 60)
    
    last_html_path = None
    
    for file_path in files_to_process:
        print(f"\n📄 Loading: {file_path}")
        data = load_telemetry(file_path)
        
        # Print summary
        stats = generate_summary_stats(data)
        print_summary(stats)
        
        run_id = data.meta.get('run_id', file_path.stem)
        
        # Generate PNG graphs
        if not args.no_png:
            try:
                output_dir = Path('telemetry/graphs') / run_id
                viz = TelemetryVisualizer(data, output_dir)
                results = viz.generate_all()
                
                successful = sum(1 for v in results.values() if v is not None)
                print(f"\n📊 Generated {successful} PNG graphs in {output_dir}/")
            except Exception as e:
                print(f"⚠️  PNG generation failed: {e}")
                print("   Make sure matplotlib is installed: pip install matplotlib")
        
        # Generate HTML dashboard
        if not args.no_html:
            try:
                html_path = Path('telemetry/dashboards') / f'{run_id}_dashboard.html'
                result = generate_html_dashboard(data, html_path)
                last_html_path = result
                print(f"🌐 Generated HTML dashboard: {result}")
            except Exception as e:
                print(f"⚠️  HTML generation failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Done! Check the telemetry/ folder for your visualizations.")
    print("=" * 60)
    
    # Open in browser if requested
    if args.open and last_html_path and last_html_path.exists():
        url = f"file://{last_html_path.absolute()}"
        print(f"\n🌐 Opening dashboard in browser...")
        print(f"   {url}")
        webbrowser.open(url)


if __name__ == '__main__':
    main()