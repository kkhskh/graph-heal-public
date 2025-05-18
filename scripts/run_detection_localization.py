#!/usr/bin/env python3

import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, Any, Optional

from graph_heal.anomaly_detection import AnomalyDetector, StatisticalAnomalyDetector, GraphAnomalyDetector
from graph_heal.fault_localization import FaultLocalizer, GraphBasedFaultLocalizer
from graph_heal.evaluation import Evaluator
from graph_heal.service_monitor import ServiceMonitor
from graph_heal.fault_injection import FaultInjector
from graph_heal.graph_updater import GraphUpdater

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('detection_localization')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run detection and localization test')
    
    parser.add_argument('--duration', type=int, default=120,
                       help='Test duration in seconds')
    
    parser.add_argument('--inject-fault', action='store_true',
                       help='Whether to inject a fault')
    
    parser.add_argument('--fault-type', type=str, default='latency',
                       choices=['latency', 'crash', 'cpu_stress', 'memory_leak'],
                       help='Type of fault to inject')
    
    parser.add_argument('--target', type=str, default='service_b',
                        help='Target service for fault injection')
    
    parser.add_argument('--fault-duration', type=int, default=30,
                       help='Duration of the fault in seconds')
    
    parser.add_argument('--fault-params', type=str, default='{}',
                       help='Additional fault parameters as JSON string')
    
    return parser.parse_args()

def setup_components(args) -> Dict[str, Any]:
    """
    Set up test components.
    
    Args:
        args: Command line arguments
    
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
    
    # Set up fault injector if needed
    fault_injector = None
    if args.inject_fault:
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

def run_test(components: Dict[str, Any], args) -> None:
    """
    Run the detection and localization test.
    
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
    test_name = f"detection_localization_{args.fault_type}_{args.target}"
    evaluator.start_test(test_name, vars(args))
    
    # Start monitoring
    monitor.start_monitoring()
    graph.start_updating()
    
    try:
        # Inject fault if requested
        injected_fault = None
        if fault_injector and args.inject_fault:
            # Parse fault parameters
            try:
                fault_params = json.loads(args.fault_params)
            except json.JSONDecodeError:
                fault_params = {}
            
            # Add duration if not specified
            if "duration" not in fault_params:
                fault_params["duration"] = args.fault_duration
        
        # Inject the fault
            fault_id = fault_injector.inject_fault(
                fault_type=args.fault_type,
                target=args.target,
                params=fault_params
            )
        
        if fault_id:
                injected_fault = {
                    "id": fault_id,
                    "type": args.fault_type,
                    "target": args.target,
                    "params": fault_params,
                    "timestamp": time.time()
                }
                logger.info(f"Injected fault: {args.fault_type} in {args.target}")
                evaluator.log_event("fault_injection", injected_fault)
    
        # Main monitoring loop
    start_time = time.time()
    end_time = start_time + args.duration
    
        service_statuses = []
        detected_anomalies = []
        localized_faults = []
    
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

def main():
    """Main function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Set up components
        components = setup_components(args)
        
        # Run test
        run_test(components, args)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()