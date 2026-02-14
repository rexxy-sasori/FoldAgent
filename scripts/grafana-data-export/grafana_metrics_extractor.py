#!/usr/bin/env python3
"""
Grafana Metrics Extractor

This script extracts metrics from all Grafana dashboards and dumps them into CSV files.
It:
1. Reads dashboard JSON files
2. Extracts all unique Prometheus queries
3. Queries Prometheus API for the metrics
4. Exports results to CSV

Usage:
    python grafana_metrics_extractor.py --dashboards-dir <path> --prometheus-url <url> --output-dir <path>
"""

import json
import argparse
import csv
import time
import requests
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Set

class GrafanaMetricsExtractor:
    def __init__(self, prometheus_url: str):
        """Initialize the extractor with Prometheus URL"""
        self.prometheus_url = prometheus_url
        self.metrics_cache: Dict[str, List[Dict]] = {}
    
    def load_dashboard(self, dashboard_path: str) -> Dict:
        """Load dashboard JSON file"""
        with open(dashboard_path, 'r') as f:
            return json.load(f)
    
    def extract_queries(self, dashboard: Dict) -> Set[str]:
        """Extract all unique queries from dashboard panels"""
        queries = set()
        
        if 'panels' not in dashboard:
            return queries
        
        def process_panels(panels):
            for panel in panels:
                if panel.get('type') == 'row' and 'panels' in panel:
                    process_panels(panel['panels'])
                elif 'targets' in panel:
                    for target in panel['targets']:
                        if 'expr' in target and target['expr']:
                            queries.add(target['expr'])
        
        process_panels(dashboard['panels'])
        return queries
    
    def query_prometheus(self, query: str, start_time: datetime, end_time: datetime, step: str = '1m') -> List[Dict]:
        """Query Prometheus API for metric data"""
        # Check cache first
        cache_key = f"{query}_{start_time}_{end_time}_{step}"
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
        
        params = {
            'query': query,
            'start': int(start_time.timestamp()),
            'end': int(end_time.timestamp()),
            'step': step
        }
        
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query_range", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'success' and 'data' in data:
                result = data['data']['result']
                self.metrics_cache[cache_key] = result
                return result
            else:
                print(f"Error querying Prometheus: {data.get('error', 'Unknown error')}")
                return []
        except Exception as e:
            print(f"Exception querying Prometheus: {e}")
            return []
    
    def export_to_csv(self, results: Dict[str, List[Dict]], output_dir: str):
        """Export metrics results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for metric_name, data in results.items():
            csv_file = os.path.join(output_dir, f"{metric_name.replace(':', '_').replace('{', '').replace('}', '').replace(' ', '_')}.csv")
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'value', 'labels'])
                
                for series in data:
                    labels = series.get('metric', {})
                    label_str = json.dumps(labels)
                    
                    for value in series.get('values', []):
                        timestamp = datetime.fromtimestamp(value[0]).isoformat()
                        writer.writerow([timestamp, value[1], label_str])
            
            print(f"Exported {metric_name} to {csv_file}")
    
    def process_dashboard(self, dashboard_path: str, output_dir: str, time_range_hours: int = 24):
        """Process a single dashboard"""
        print(f"Processing dashboard: {dashboard_path}")
        
        # Load dashboard
        dashboard = self.load_dashboard(dashboard_path)
        
        # Extract queries
        queries = self.extract_queries(dashboard)
        print(f"Found {len(queries)} unique queries")
        
        # Set time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_range_hours)
        
        # Query all metrics
        results = {}
        for query in queries:
            print(f"Querying: {query}")
            data = self.query_prometheus(query, start_time, end_time)
            if data:
                # Use the first part of the query as metric name
                metric_name = query.split('{')[0].strip()
                results[metric_name] = data
                time.sleep(0.1)  # Throttle requests
        
        # Export to CSV
        if results:
            self.export_to_csv(results, output_dir)
            print(f"Exported {len(results)} metrics to CSV")
        else:
            print("No data found for any metrics")
    
    def process_dashboards_dir(self, dashboards_dir: str, output_dir: str, time_range_hours: int = 24):
        """Process all dashboards in a directory"""
        for filename in os.listdir(dashboards_dir):
            if filename.endswith('.json'):
                dashboard_path = os.path.join(dashboards_dir, filename)
                dashboard_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
                self.process_dashboard(dashboard_path, dashboard_output_dir, time_range_hours)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract metrics from Grafana dashboards and export to CSV')
    parser.add_argument('--dashboards-dir', type=str, default='/Users/rexsasori/FoldAgent/deployment/observability/grafana-dashboards',
                        help='Directory containing Grafana dashboard JSON files')
    parser.add_argument('--prometheus-url', type=str, default='http://localhost:9090',
                        help='Prometheus API URL')
    parser.add_argument('--output-dir', type=str, default='/Users/rexsasori/FoldAgent/metrics_output',
                        help='Directory to save CSV output files')
    parser.add_argument('--time-range', type=int, default=24,
                        help='Time range in hours to query')
    
    args = parser.parse_args()
    
    extractor = GrafanaMetricsExtractor(args.prometheus_url)
    extractor.process_dashboards_dir(args.dashboards_dir, args.output_dir, args.time_range)
