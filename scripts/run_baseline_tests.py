import sys
import os
import time
import argparse
import docker

print("Starting run_baseline_tests.py...")

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(f"Python path: {sys.path}")

from graph_heal.graph_model import create_sample_graph
from graph_heal.monitoring import ServiceMonitor, GraphUpdater
from graph_heal.fault_injection import FaultInjector, FaultInjectionAPI
from graph_heal.test_scenarios import (
    TestScenario, NormalOperationScenario, SinglePointFailureScenario,
    CascadingFailureScenario, ScenarioRunner
)

def main():
    print("Entering main()...")
    parser = argparse.ArgumentParser(description='Run baseline tests')
    parser.add_argument('--scenarios', type=str, nargs='+',
                        choices=['normal', 'single', 'cascading', 'all'],
                        default=['all'],
                        help='Test scenarios to run')
    parser.add_argument('--duration', type=int, default=60,
                        help='Base duration for tests (in seconds)')
    args = parser.parse_args()
    print(f"Parsed arguments: {args}")
    
    # Create Docker client
    print("Creating Docker client...")
    docker_client = docker.from_env()
    
    # Create a sample graph
    print("Creating sample graph...")
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
    
    # Create fault injector
    print("Creating fault injector...")
    injector = FaultInjector(docker_client)
    fault_api = FaultInjectionAPI(injector)
    
    # Create test scenarios
    print("Creating test scenarios...")
    scenarios = []
    
    if 'all' in args.scenarios or 'normal' in args.scenarios:
        print("Adding normal operation scenario...")
        scenarios.append(
            NormalOperationScenario(
                fault_api=fault_api,
                service_monitor=monitor,
                graph_updater=updater,
                duration=args.duration
            )
        )
    
    if 'all' in args.scenarios or 'single' in args.scenarios:
        print("Adding single point failure scenarios...")
        # Add several single-point failure scenarios
        scenarios.extend([
            SinglePointFailureScenario(
                fault_api=fault_api,
                service_monitor=monitor,
                graph_updater=updater,
                target="service_a",
                fault_type="crash",
                fault_duration=30,
                total_duration=args.duration
            ),
            SinglePointFailureScenario(
                fault_api=fault_api,
                service_monitor=monitor,
                graph_updater=updater,
                target="service_c",
                fault_type="cpu_stress",
                fault_duration=40,
                total_duration=args.duration
            ),
            SinglePointFailureScenario(
                fault_api=fault_api,
                service_monitor=monitor,
                graph_updater=updater,
                target="service_b",
                fault_type="latency",
                fault_duration=40,
                total_duration=args.duration
            )
        ])
    
    if 'all' in args.scenarios or 'cascading' in args.scenarios:
        print("Adding cascading failure scenario...")
        scenarios.append(
            CascadingFailureScenario(
                fault_api=fault_api,
                service_monitor=monitor,
                graph_updater=updater,
                total_duration=args.duration * 2  # Cascading failures need more time
            )
        )
    
    # Create scenario runner
    print("Creating scenario runner...")
    runner = ScenarioRunner(scenarios)
    
    # Run all scenarios
    print(f"Running {len(scenarios)} test scenarios...")
    summary = runner.run_all()
    
    print("\nTest Summary:")
    print(f"Total scenarios: {summary['total_scenarios']}")
    print(f"Successful scenarios: {summary['successful_scenarios']}")
    print(f"Failed scenarios: {summary['failed_scenarios']}")
    print(f"Total duration: {summary['total_duration']:.1f} seconds")
    
    print("\nScenario Details:")
    for scenario in summary['scenarios']:
        status = "✓" if scenario['success'] else "✗"
        print(f"{status} {scenario['name']} - Duration: {scenario['duration']:.1f}s")

if __name__ == "__main__":
    print("Script starting...")
    main()
    print("Script finished.")