#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, Any
import docker
import threading
from flask import Flask, jsonify, request

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_heal.graph_model import create_sample_graph
from graph_heal.monitoring import ServiceMonitor, GraphUpdater
from graph_heal.fault_injection import FaultInjector, FaultInjectionAPI
from graph_heal.anomaly_detection import StatisticalAnomalyDetector, GraphAnomalyDetector, AnomalyManager
from graph_heal.fault_localization import GraphBasedFaultLocalizer, FaultManager
from graph_heal.recovery import RecoveryActionFactory, RecoveryDecisionEngine
from graph_heal.orchestration import RecoveryOrchestrator, SystemController

# Global controller instance
controller = None

def create_api_server(port=8000):
    """
    Create an API server for the system.
    
    Args:
        port: Port to listen on
    
    Returns:
        Flask app
    """
    app = Flask(__name__)
    
    @app.route('/status', methods=['GET'])
    def get_status():
        """Get system status."""
        if controller is None:
            return jsonify({"status": "not_running"}), 503
        
        return jsonify(controller.get_system_status())
    
    @app.route('/faults', methods=['GET'])
    def get_faults():
        """Get active faults."""
        if controller is None:
            return jsonify({"error": "System not running"}), 503
        
        return jsonify(controller.fault_manager.get_active_faults())
    
    @app.route('/tasks', methods=['GET'])
    def get_tasks():
        """Get active recovery tasks."""
        if controller is None:
            return jsonify({"error": "System not running"}), 503
        
        return jsonify(controller.orchestrator.get_active_tasks())
    
    @app.route('/services', methods=['GET'])
    def get_services():
        """Get service statuses."""
        if controller is None:
            return jsonify({"error": "System not running"}), 503
        
        return jsonify(controller.monitor.get_all_services_status())
    
    @app.route('/intervention', methods=['POST'])
    def manual_intervention():
        """Perform a manual intervention."""
        if controller is None:
            return jsonify({"error": "System not running"}), 503
        
        data = request.json
        action_type = data.get('action_type')
        target_id = data.get('target_id')
        params = data.get('params', {})
        
        if not action_type or not target_id:
            return jsonify({"error": "Missing action_type or target_id"}), 400
        
        result = controller.manual_intervention(action_type, target_id, params)
        return jsonify(result)
    
    def run_server():
        app.run(host='0.0.0.0', port=port)
    
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    print(f"API server running on http://localhost:{port}")
    
    return app

def main():
    global controller
    
    parser = argparse.ArgumentParser(description='Run the complete system')
    parser.add_argument('--duration', type=int, default=3600,
                        help='Duration to run (in seconds)')
    parser.add_argument('--api-port', type=int, default=8000,
                        help='Port for the API server')
    parser.add_argument('--inject-fault', action='store_true',
                        help='Inject a fault during the run')
    parser.add_argument('--fault-type', type=str, choices=['cpu_stress', 'memory_leak', 'latency', 'crash'],
                        default='latency', help='Type of fault to inject')
    parser.add_argument('--target', type=str, default='service_b',
                        help='Target service for fault injection')
    args = parser.parse_args()
    
    # Create Docker client
    docker_client = docker.from_env()
    
    # Create a sample graph
    graph = create_sample_graph()
    
    # Load service configuration
    with open('config/system_config.json', 'r') as f:
        config = json.load(f)
    
    # Create components
    
    # 1. Monitoring
    monitor = ServiceMonitor(services_config=config['services'], poll_interval=2)
    graph_updater = GraphUpdater(graph, monitor, update_interval=5)
    
    # 2. Anomaly Detection
    stat_cfg = config.get("anomaly_detection", {}).get("statistical", {})
    window_size = stat_cfg.get("window_size", 10)
    z_score_threshold = stat_cfg.get("z_score_threshold", 2.5)
    #statistical_detector = StatisticalAnomalyDetector(window_size=window_size, z_score_threshold=z_score_threshold)
    #graph_detector = GraphAnomalyDetector()
    anomaly_manager = AnomalyManager([])
    
    # 3. Fault Localization
    #fault_localizer = GraphBasedFaultLocalizer()
    fault_manager = FaultManager([])
    
    # 4. Recovery
    action_factory = RecoveryActionFactory(docker_client)
    decision_engine = RecoveryDecisionEngine(action_factory)
    
    # 5. Orchestration
    orchestrator = RecoveryOrchestrator(decision_engine, fault_manager)
    
    # 6. System Controller
    controller = SystemController(monitor, graph_updater, fault_manager, orchestrator)
    
    # 7. Fault Injection (for testing)
    injector = FaultInjector()
    fault_api = FaultInjectionAPI(injector)
    
    # Start API server
    create_api_server(port=args.api_port)
    
    # Start the system
    print("Starting the system...")
    controller.start()
    
    # Inject a fault if requested
    fault_id = None
    if args.inject_fault:
        # Wait for the system to initialize
        time.sleep(20)
        
        print(f"Injecting {args.fault_type} fault in {args.target}...")
        
        # Prepare parameters based on fault type
        params = {"duration": 30}  # Default 30s duration
        target = args.target
        
        if args.fault_type == 'cpu_stress':
            params["load"] = 80
        elif args.fault_type == 'memory_leak':
            params["memory_mb"] = 50
        elif args.fault_type == 'latency':
            params["latency_ms"] = 500
            target = f"http://localhost:{5001 if args.target == 'service_a' else 5002 if args.target == 'service_b' else 5003 if args.target == 'service_c' else 5004}"
        elif args.fault_type == 'crash':
            params["restart_after"] = 30
        
        # Inject the fault
        fault_id = fault_api.inject_fault(args.fault_type, target, params)
        
        if fault_id:
            print(f"Successfully injected fault with ID: {fault_id}")
        else:
            print("Failed to inject fault")
    
    # Run for the specified duration
    try:
        print(f"System running for {args.duration} seconds...")
        print(f"API available at http://localhost:{args.api_port}")
        
        end_time = time.time() + args.duration
        while time.time() < end_time:
            # Print periodic status updates
            if int(time.time()) % 30 == 0:  # Every 30 seconds
                status = controller.get_system_status()
                print(f"\nSystem Status: {status['status']}")
                print(f"Services: {status['services']['healthy']} healthy, {status['services']['unhealthy']} unhealthy")
                print(f"Active faults: {status['faults']['active']}")
                print(f"Active recovery tasks: {status['recovery']['active_tasks']}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping the system...")
    
    finally:
        # Stop the system
        controller.stop()
        
        # Clean up any injected faults
        if fault_id:
            fault_api.stop_fault(fault_id)
        
        print("System stopped.")

if __name__ == "__main__":
    main()