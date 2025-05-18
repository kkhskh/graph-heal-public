import sys
import os
import time
import argparse
import docker
import subprocess

print("Starting test_fault_injection.py...")

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(f"Python path: {sys.path}")

from graph_heal.fault_injection import FaultInjector, FaultInjectionAPI, ScheduledFaultInjector
from graph_heal.graph_model import create_sample_graph
from graph_heal.monitoring import ServiceMonitor, GraphUpdater

def main():
    print("Entering main()...")
    parser = argparse.ArgumentParser(description='Test fault injection')
    parser.add_argument('--fault-type', type=str, choices=['cpu_stress', 'memory_leak', 'latency', 'crash'],
                        default='latency', help='Type of fault to inject')
    parser.add_argument('--target', type=str, default='service_a',
                        help='Target service or container')
    parser.add_argument('--duration', type=int, default=30,
                        help='Duration of the fault (in seconds)')
    parser.add_argument('--monitor', action='store_true',
                        help='Enable system monitoring during fault injection')
    args = parser.parse_args()
    print(f"Parsed arguments: {args}")
    
    # Create Docker client
    print("Creating Docker client...")
    docker_client = docker.from_env()
    
    # Create fault injector and API
    print("Creating fault injector and API...")
    # injector = FaultInjector(docker_client)
    injector = FaultInjector()
    fault_api = FaultInjectionAPI(injector)
    
    # Create scheduler (optional)
    scheduler = ScheduledFaultInjector(fault_api)
    
    # Setup monitoring if requested
    if args.monitor:
        print("Setting up monitoring...")
        # Create a sample graph
        graph = create_sample_graph()
        
        # Define service configurations for the monitor
        services_config = [
            {
                "id": "service_a",
                "name": "User Service",
                "url": "http://localhost:5001",
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            {
                "id": "service_b",
                "name": "Order Service",
                "url": "http://localhost:5002",
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            {
                "id": "service_c",
                "name": "Inventory Service",
                "url": "http://localhost:5003",
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            {
                "id": "service_d",
                "name": "Notification Service",
                "url": "http://localhost:5004",
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            }
        ]
        
        # Create a service monitor
        print("Creating service monitor...")
        monitor = ServiceMonitor(services_config, poll_interval=2)
        
        # Create a graph updater
        print("Creating graph updater...")
        updater = GraphUpdater(graph, monitor, update_interval=5)
        
        # Start monitoring
        print("Starting monitoring...")
        monitor.start_monitoring()
        updater.start_updating()
        
        print("Started monitoring system")
    
    # Prepare parameters based on fault type
    params = {"duration": args.duration}
    target = args.target
    
    if args.fault_type == 'cpu_stress':
        params["load"] = 80
    elif args.fault_type == 'memory_leak':
        params["memory_mb"] = 50
    elif args.fault_type == 'latency':
        params["latency_ms"] = 500
        target = f"http://localhost:{5001 if args.target == 'service_a' else 5002 if args.target == 'service_b' else 5003 if args.target == 'service_c' else 5004}"
    elif args.fault_type == 'crash':
        params["restart_after"] = args.duration
    
    print(f"Injecting {args.fault_type} fault into {target} with params: {params}")
    
    # Inject the fault
    fault_id = fault_api.inject_fault(args.fault_type, target, params)
    
    if fault_id:
        print(f"Successfully injected fault with ID: {fault_id}")
        
        # Wait for fault to complete
        for i in range(args.duration + 5):  # Add a small buffer
            time.sleep(1)
            # Get fault info
            fault_info = fault_api.get_fault_info(fault_id)
            if fault_info and fault_info["status"] != "active":
                print(f"Fault {fault_id} completed with status: {fault_info['status']}")
                break
            
            if i % 5 == 0:
                print(f"Fault running for {i} seconds...")
        
        print("\nFault injection test completed.")
        
        # If monitoring, show final status
        if args.monitor:
            print("\nFinal system status:")
            statuses = monitor.get_all_services_status()
            for service_id, status in statuses.items():
                print(f"{status['name']} ({service_id}): {status['health']} "
                      f"(Availability: {status['availability']:.1f}%)")
            
            # Stop monitoring
            print("Stopping monitoring...")
            monitor.stop_monitoring()
            updater.stop_updating()
            
            # Save final graph
            print("Saving final graph...")
            graph.visualize("data/graphs/post_fault_graph.png")
            print("\nFinal graph saved to data/graphs/post_fault_graph.png")
    else:
        print(f"Failed to inject {args.fault_type} fault")
        if args.monitor:
            monitor.stop_monitoring()
            updater.stop_updating()

if __name__ == "__main__":
    print("Script starting...")
    main()
    print("Script finished.")