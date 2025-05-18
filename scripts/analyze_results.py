import sys
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import glob

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_test_results(scenario_name: Optional[str] = None):
    """
    Load test results from the data directory.
    
    Args:
        scenario_name: Optional name of the scenario to load
        
    Returns:
        List of test result dictionaries
    """
    data_dir = "data/test_scenarios"
    
    # Get all JSON files in the directory
    json_files = glob.glob(f"{data_dir}/*.json")
    
    # Filter out summary files
    result_files = [f for f in json_files if "summary" not in f]
    
    if scenario_name:
        # Filter by scenario name
        scenario_name_lower = scenario_name.lower().replace(' ', '_')
        result_files = [f for f in result_files if scenario_name_lower in f]
    
    results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def analyze_service_availability(results):
    """
    Analyze service availability across test scenarios.
    
    Args:
        results: List of test result dictionaries
        
    Returns:
        Analysis results
    """
    analysis = {}
    
    for result in results:
        if not result.get("success"):
            continue
        
        scenario_name = result.get("name", "Unknown")
        snapshots = result.get("execution", {}).get("result", {}).get("snapshots", [])
        
        if not snapshots:
            continue
        
        # Extract service availability over time
        timeline = []
        service_availability = {}
        
        for snapshot in snapshots:
            relative_time = snapshot.get("relative_time", 0)
            timeline.append(relative_time)
            
            for service_id, status in snapshot.get("services_status", {}).items():
                if service_id not in service_availability:
                    service_availability[service_id] = []
                
                availability = status.get("availability", 0)
                service_availability[service_id].append(availability)
        
        analysis[scenario_name] = {
            "timeline": timeline,
            "service_availability": service_availability
        }
    
    return analysis

def plot_availability_chart(analysis, output_file: Optional[str] = None):
    """
    Plot service availability chart.
    
    Args:
        analysis: Analysis results
        output_file: Optional output file path
    """
    num_scenarios = len(analysis)
    if num_scenarios == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(num_scenarios, 1, figsize=(12, 4 * num_scenarios))
    if num_scenarios == 1:
        axes = [axes]
    
    for i, (scenario_name, data) in enumerate(analysis.items()):
        ax = axes[i]
        timeline = data["timeline"]
        
        for service_id, availability in data["service_availability"].items():
            if len(timeline) != len(availability):
                # Pad with zeros if necessary
                availability = availability + [0] * (len(timeline) - len(availability))
            
            ax.plot(timeline, availability, label=service_id)
        
        ax.set_title(f"Service Availability - {scenario_name}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Availability (%)")
        ax.set_ylim(0, 105)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Chart saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze test results')
    parser.add_argument('--scenario', type=str, help='Name of the scenario to analyze')
    parser.add_argument('--output', type=str, help='Output file path for charts')
    args = parser.parse_args()
    
    # Load test results
    results = load_test_results(args.scenario)
    
    if not results:
        print("No test results found")
        return
    
    print(f"Loaded {len(results)} test results")
    
    # Analyze service availability
    availability_analysis = analyze_service_availability(results)
    
    # Plot charts
    output_file = args.output or "data/test_scenarios/availability_chart.png"
    plot_availability_chart(availability_analysis, output_file)

if __name__ == "__main__":
    main()