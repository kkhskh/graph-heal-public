#!/usr/bin/env python3

import os
import sys
import time
import json
import logging
import argparse
from collections import defaultdict
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

from graph_heal.anomaly_detection import StatisticalAnomalyDetector, GraphAnomalyDetector
from graph_heal.fault_localization import GraphBasedFaultLocalizer
from graph_heal.evaluation import Evaluator
from graph_heal.service_monitor import ServiceMonitor
from graph_heal.fault_injection import FaultInjector
from graph_heal.graph_updater import GraphUpdater

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluation')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run evaluation tests')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'suite', 'plot'],
                       help='Evaluation mode')
    
    parser.add_argument('--duration', type=int, default=120,
                       help='Test duration in seconds')
    
    parser.add_argument('--fault-duration', type=int, default=30,
                       help='Duration of each fault in seconds')
    
    parser.add_argument('--target', type=str, default='service_b',
                       help='Target service for single test')
    
    return parser.parse_args()

def setup_components() -> Dict[str, Any]:
    """
    Set up test components.
    
    Returns:
        Dictionary of components
    """
    # Create sample graph
    graph = GraphUpdater()
    
    # Set up service monitor
    monitor = ServiceMonitor()
    
    # Set up anomaly detectors
    statistical_detector = StatisticalAnomalyDetector(
        window_size=10,
        z_score_threshold=2.5
    )
    
    graph_detector = GraphAnomalyDetector(
        correlation_threshold=0.7
    )
    
    # Set up fault localizer
    localizer = GraphBasedFaultLocalizer()
    
    # Set up fault injector
    fault_injector = FaultInjector()
    
    # Set up evaluator
    evaluator = Evaluator()
    
    return {
        "graph": graph,
        "monitor": monitor,
        "statistical_detector": statistical_detector,
        "graph_detector": graph_detector,
        "localizer": localizer,
        "fault_injector": fault_injector,
        "evaluator": evaluator
    }

def run_single_test(components: Dict[str, Any], args) -> None:
    """
    Run a single evaluation test.
    
    Args:
        components: Dictionary of test components
        args: Command line arguments
    """
    # Extract components
    graph = components["graph"]
    monitor = components["monitor"]
    statistical_detector = components["statistical_detector"]
    graph_detector = components["graph_detector"]
    localizer = components["localizer"]
    fault_injector = components["fault_injector"]
    evaluator = components["evaluator"]
    
    # Start test
    test_name = f"single_test_{args.target}"
    evaluator.start_test(test_name, vars(args))
    
    # Start monitoring
    monitor.start_monitoring()
    graph.start_updating()
    
    try:
        # Initialize test state
        injected_fault = None
        service_statuses = []
        detected_anomalies = []
        localized_faults = []
        
        # Inject fault
        fault_id = fault_injector.inject_fault(
            fault_type="latency",
            target=args.target,
            params={"duration": args.fault_duration}
        )
        
        if fault_id:
            injected_fault = {
                "id": fault_id,
                "type": "latency",
                "target": args.target,
                "params": {"duration": args.fault_duration},
                "timestamp": time.time()
            }
            logger.info(f"Injected latency fault in {args.target}")
            evaluator.log_event("fault_injection", injected_fault)
        
        # Main monitoring loop
        start_time = time.time()
        end_time = start_time + args.duration
        
        while time.time() < end_time:
            # Get current system state
            current_graph = graph.get_current_graph()
            current_statuses = monitor.get_all_services_status()
            
            # Store status snapshot
            service_statuses.append(current_statuses)
            
            # Detect anomalies
            statistical_anomalies = statistical_detector.detect_anomalies(current_statuses)
            graph_anomalies = graph_detector.detect_anomalies(current_statuses)
            
            # Combine anomalies
            new_anomalies = statistical_anomalies + graph_anomalies
            detected_anomalies.extend(new_anomalies)
            
            if new_anomalies:
                evaluator.log_event("anomalies_detected", {
                    "count": len(new_anomalies),
                    "anomalies": new_anomalies
                })
            
            # Localize faults
            new_faults = localizer.localize_faults(current_statuses, new_anomalies)
            localized_faults.extend(new_faults)
            
            if new_faults:
                evaluator.log_event("faults_localized", {
                    "count": len(new_faults),
                    "faults": new_faults
                })
            
            # Sleep briefly
            time.sleep(1)
        
        # Evaluate results
        if injected_fault:
            evaluator.evaluate_detection([injected_fault], detected_anomalies)
            evaluator.evaluate_localization([injected_fault], localized_faults)
        
        evaluator.evaluate_recovery(service_statuses)
        evaluator.evaluate_availability([s.get("metrics", {}) for s in service_statuses])
        
        # Generate plots
        evaluator.plot_metrics()
    
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        graph.stop_updating()
        
        # End test
        evaluator.end_test()

def run_test_suite(components: Dict[str, Any], args) -> None:
    """
    Run a suite of evaluation tests.
    
    Args:
        components: Dictionary of test components
        args: Command line arguments
    """
    # Define test scenarios
    scenarios = [
            {
            "name": "latency_test",
            "fault_type": "latency",
                        "target": "service_b",
            "duration": args.fault_duration
        },
        {
            "name": "crash_test",
            "fault_type": "crash",
            "target": "service_a",
            "duration": args.fault_duration
            },
            {
            "name": "cpu_stress_test",
            "fault_type": "cpu_stress",
                        "target": "service_c",
            "duration": args.fault_duration
        }
    ]
    
    # Run each scenario
    for scenario in scenarios:
        logger.info(f"Running scenario: {scenario['name']}")
        
        # Update args for this scenario
        args.target = scenario["target"]
        args.fault_duration = scenario["duration"]
        
        # Run the test
        run_single_test(components, args)
        
        # Wait between tests
        time.sleep(10)

def generate_plots(evaluator: Evaluator) -> None:
    """
    Generate summary plots from evaluation results.
    
    Args:
        evaluator: Evaluator instance
    """
    # Load all test results
    results_dir = evaluator.data_dir
    test_files = [f for f in os.listdir(results_dir) if f.startswith("test_") and f.endswith(".json")]
    
    all_metrics = defaultdict(list)
    
    for filename in test_files:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r') as f:
            result = json.load(f)
            metrics = result.get("metrics", {})
            
            for metric_name, metric_data in metrics.items():
                all_metrics[metric_name].append({
                    "test": result["name"],
                    "value": metric_data.get("mean", 0)
                })
    
    # Generate comparison plots
    for metric_name, values in all_metrics.items():
        plt.figure(figsize=(10, 6))
        
        tests = [v["test"] for v in values]
        metric_values = [v["value"] for v in values]
        
        plt.bar(tests, metric_values)
        plt.title(f"{metric_name} Comparison")
        plt.xlabel("Test")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(results_dir, f"{metric_name}_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Generated comparison plot for {metric_name}: {plot_path}")

def main():
    """Main function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Set up components
        components = setup_components()
        
        # Run tests based on mode
        if args.mode == "single":
            run_single_test(components, args)
        elif args.mode == "suite":
            run_test_suite(components, args)
        elif args.mode == "plot":
            generate_plots(components["evaluator"])
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()