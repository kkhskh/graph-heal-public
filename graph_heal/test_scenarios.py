import time
import logging
import random
import threading
import os
from typing import Dict, List, Any, Optional, Callable
import json
import datetime

# Set fixed random seed for reproducibility
random.seed(42)
try:
    import numpy as np
    np.random.seed(42)
except ImportError:
    pass

from graph_heal.fault_injection import FaultInjectionAPI
from graph_heal.monitoring import ServiceMonitor, GraphUpdater

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_scenarios')

class TestScenario:
    """
    Base class for test scenarios.
    """
    def __init__(self, name: str, description: str, 
                fault_api: FaultInjectionAPI, 
                service_monitor: ServiceMonitor, 
                graph_updater: GraphUpdater):
        """
        Initialize the test scenario.
        
        Args:
            name: Name of the scenario
            description: Description of the scenario
            fault_api: FaultInjectionAPI instance
            service_monitor: ServiceMonitor instance
            graph_updater: GraphUpdater instance
        """
        self.name = name
        self.description = description
        self.fault_api = fault_api
        self.monitor = service_monitor
        self.graph_updater = graph_updater
        self.data_dir = "data/test_scenarios"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def setup(self):
        """Set up the test scenario."""
        pass
    
    def run(self):
        """Run the test scenario."""
        pass
    
    def cleanup(self):
        """Clean up the test scenario."""
        pass
    
    def execute(self):
        """Execute the full test scenario."""
        results = {
            "name": self.name,
            "description": self.description,
            "started": datetime.datetime.now().isoformat(),
            "setup": {},
            "execution": {},
            "cleanup": {},
            "finished": None,
            "success": False
        }
        
        try:
            logger.info(f"Starting test scenario: {self.name}")
            logger.info(f"Description: {self.description}")
            
            # Set up
            logger.info("Setting up test scenario...")
            setup_start = time.time()
            setup_result = self.setup()
            setup_end = time.time()
            
            results["setup"] = {
                "duration": setup_end - setup_start,
                "result": setup_result
            }
            
            # Run
            logger.info("Running test scenario...")
            run_start = time.time()
            run_result = self.run()
            run_end = time.time()
            
            results["execution"] = {
                "duration": run_end - run_start,
                "result": run_result
            }
            
            # Clean up
            logger.info("Cleaning up test scenario...")
            cleanup_start = time.time()
            cleanup_result = self.cleanup()
            cleanup_end = time.time()
            
            results["cleanup"] = {
                "duration": cleanup_end - cleanup_start,
                "result": cleanup_result
            }
            
            results["finished"] = datetime.datetime.now().isoformat()
            results["success"] = True
            
            logger.info(f"Test scenario {self.name} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in test scenario {self.name}: {e}")
            results["error"] = str(e)
            results["finished"] = datetime.datetime.now().isoformat()
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.data_dir}/{self.name.lower().replace(' ', '_')}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {filename}")
        
        return results


class NormalOperationScenario(TestScenario):
    """
    Normal operation scenario without any faults.
    """
    def __init__(self, fault_api, service_monitor, graph_updater, duration: int = 60):
        super().__init__(
            name="Normal Operation",
            description="Test scenario with no faults to establish baseline",
            fault_api=fault_api,
            service_monitor=service_monitor,
            graph_updater=graph_updater
        )
        self.duration = duration
    
    def setup(self):
        # Start monitoring
        self.monitor.start_monitoring()
        self.graph_updater.start_updating()
        return {"status": "monitoring_started"}
    
    def run(self):
        # Just wait for the specified duration
        logger.info(f"Running normal operation for {self.duration} seconds...")
        
        # Take snapshots at regular intervals
        snapshot_interval = min(10, self.duration // 6)  # Take at least 6 snapshots
        start_time = time.time()
        snapshots = []
        
        while time.time() - start_time < self.duration:
            # Sleep for the snapshot interval
            time.sleep(snapshot_interval)
            
            # Take a snapshot
            services_status = self.monitor.get_all_services_status()
            snapshot = {
                "timestamp": time.time(),
                "relative_time": time.time() - start_time,
                "services_status": services_status
            }
            snapshots.append(snapshot)
            
            logger.info(f"Snapshot at {snapshot['relative_time']:.1f}s:")
            for service_id, status in services_status.items():
                logger.info(f"  {status['name']}: {status['health']} (Avail: {status['availability']:.1f}%)")
        
        # Get final status
        final_status = self.monitor.get_all_services_status()
        
        return {
            "snapshots": snapshots,
            "final_status": final_status
        }
    
    def cleanup(self):
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.graph_updater.stop_updating()
        
        # Save final graph
        filename = f"{self.data_dir}/normal_operation_final_graph.png"
        self.graph_updater.get_current_graph().visualize(filename)
        
        return {"status": "monitoring_stopped", "final_graph": filename}


class SinglePointFailureScenario(TestScenario):
    """
    Single point failure scenario with recovery.
    """
    def __init__(self, fault_api, service_monitor, graph_updater, 
                target: str = "service_a", fault_type: str = "crash", 
                fault_duration: int = 30, total_duration: int = 90):
        super().__init__(
            name=f"Single Point Failure - {target} - {fault_type}",
            description=f"Test scenario with a {fault_type} fault injected into {target} for {fault_duration}s",
            fault_api=fault_api,
            service_monitor=service_monitor,
            graph_updater=graph_updater
        )
        self.target = target
        self.fault_type = fault_type
        self.fault_duration = fault_duration
        self.total_duration = total_duration
        self.fault_id = None
    
    def setup(self):
        # Start monitoring
        self.monitor.start_monitoring()
        self.graph_updater.start_updating()
        
        # Wait a bit for initial monitoring data
        time.sleep(5)
        
        return {"status": "monitoring_started"}
    
    def run(self):
        # Prepare fault parameters
        params = {"duration": self.fault_duration}
        target = self.target
        
        # Adjust parameters based on fault type
        if self.fault_type == 'cpu_stress':
            params["load"] = 80
        elif self.fault_type == 'memory_leak':
            params["memory_mb"] = 50
        elif self.fault_type == 'latency':
            params["latency_ms"] = 500
            # Convert service name to URL
            service_port = {"service_a": 5001, "service_b": 5002, "service_c": 5003, "service_d": 5004}.get(target, 5001)
            target = f"http://localhost:{service_port}"
        elif self.fault_type == 'crash':
            params["restart_after"] = self.fault_duration
        
        logger.info(f"Injecting {self.fault_type} fault into {target} for {self.fault_duration} seconds...")
        
        # Inject the fault
        self.fault_id = self.fault_api.inject_fault(self.fault_type, target, params)
        if not self.fault_id:
            logger.error(f"Failed to inject {self.fault_type} fault into {target}")
            return {"status": "fault_injection_failed"}
        
        logger.info(f"Fault injected with ID: {self.fault_id}")
        
        # Take snapshots at regular intervals
        snapshot_interval = min(10, self.total_duration // 10)  # Take at least 10 snapshots
        start_time = time.time()
        snapshots = []
        
        while time.time() - start_time < self.total_duration:
            # Sleep for the snapshot interval
            time.sleep(snapshot_interval)
            
            # Take a snapshot
            services_status = self.monitor.get_all_services_status()
            snapshot = {
                "timestamp": time.time(),
                "relative_time": time.time() - start_time,
                "services_status": services_status,
                "fault_info": self.fault_api.get_fault_info(self.fault_id)
            }
            snapshots.append(snapshot)
            
            logger.info(f"Snapshot at {snapshot['relative_time']:.1f}s:")
            for service_id, status in services_status.items():
                logger.info(f"  {status['name']}: {status['health']} (Avail: {status['availability']:.1f}%)")
        
        # Get final status
        final_status = self.monitor.get_all_services_status()
        
        return {
            "snapshots": snapshots,
            "final_status": final_status,
            "fault_id": self.fault_id,
            "fault_info": self.fault_api.get_fault_info(self.fault_id)
        }
    
    def cleanup(self):
        # Make sure the fault is stopped
        if self.fault_id:
            self.fault_api.stop_fault(self.fault_id)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.graph_updater.stop_updating()
        
        # Save final graph
        filename = f"{self.data_dir}/single_point_failure_final_graph.png"
        self.graph_updater.get_current_graph().visualize(filename)
        
        return {"status": "monitoring_stopped", "final_graph": filename}


class CascadingFailureScenario(TestScenario):
    """
    Cascading failure scenario with multiple faults.
    """
    def __init__(self, fault_api, service_monitor, graph_updater, 
                total_duration: int = 180):
        super().__init__(
            name="Cascading Failure",
            description="Test scenario with multiple cascading failures",
            fault_api=fault_api,
            service_monitor=service_monitor,
            graph_updater=graph_updater
        )
        self.total_duration = total_duration
        self.fault_ids = []
    
    def setup(self):
        # Start monitoring
        self.monitor.start_monitoring()
        self.graph_updater.start_updating()
        
        # Wait a bit for initial monitoring data
        time.sleep(5)
        
        return {"status": "monitoring_started"}
    
    def run(self):
        # Plan the cascading failures
        # First phase: Introduce latency in one service
        # Second phase: CPU stress in a dependent service
        # Third phase: Memory leak in another service
        # Fourth phase: Crash a critical service
        
        phase_duration = self.total_duration // 4
        snapshots = []
        start_time = time.time()
        
        # Phase 1: Latency in service_a
        logger.info("Phase 1: Introducing latency in service_a...")
        fault_id = self.fault_api.inject_fault(
            "latency",
            "http://localhost:5001",
            {"duration": phase_duration * 2, "latency_ms": 500}
        )
        if fault_id:
            self.fault_ids.append(fault_id)
        
        # Take snapshots during phase 1
        phase_end = start_time + phase_duration
        while time.time() < phase_end:
            time.sleep(5)
            services_status = self.monitor.get_all_services_status()
            snapshot = {
                "timestamp": time.time(),
                "relative_time": time.time() - start_time,
                "services_status": services_status,
                "phase": 1,
                "fault_ids": self.fault_ids.copy()
            }
            snapshots.append(snapshot)
            
            logger.info(f"Phase 1 - Snapshot at {snapshot['relative_time']:.1f}s:")
            for service_id, status in services_status.items():
                logger.info(f"  {status['name']}: {status['health']} (Avail: {status['availability']:.1f}%)")
                    
        # Phase 2: CPU stress in service_b
        logger.info("Phase 2: Adding CPU stress in service_b...")
        fault_id = self.fault_api.inject_fault(
            "cpu_stress",
            "service_b",
            {"duration": phase_duration * 2, "load": 80}
        )
        if fault_id:
            self.fault_ids.append(fault_id)
        
        # Take snapshots during phase 2
        phase_end = start_time + phase_duration * 2
        while time.time() < phase_end:
            time.sleep(5)
            services_status = self.monitor.get_all_services_status()
            snapshot = {
                "timestamp": time.time(),
                "relative_time": time.time() - start_time,
                "services_status": services_status,
                "phase": 2,
                "fault_ids": self.fault_ids.copy()
            }
            snapshots.append(snapshot)
            
            logger.info(f"Phase 2 - Snapshot at {snapshot['relative_time']:.1f}s:")
            for service_id, status in services_status.items():
                logger.info(f"  {status['name']}: {status['health']} (Avail: {status['availability']:.1f}%)")
        
        # Phase 3: Memory leak in service_c
        logger.info("Phase 3: Adding memory leak in service_c...")
        fault_id = self.fault_api.inject_fault(
            "memory_leak",
            "service_c",
            {"duration": phase_duration * 2, "memory_mb": 50}
        )
        if fault_id:
            self.fault_ids.append(fault_id)
            
        # Take snapshots during phase 3
        phase_end = start_time + phase_duration * 3
        while time.time() < phase_end:
            time.sleep(5)
            services_status = self.monitor.get_all_services_status()
            snapshot = {
                "timestamp": time.time(),
                "relative_time": time.time() - start_time,
                "services_status": services_status,
                "phase": 3,
                "fault_ids": self.fault_ids.copy()
            }
            snapshots.append(snapshot)
            
            logger.info(f"Phase 3 - Snapshot at {snapshot['relative_time']:.1f}s:")
            for service_id, status in services_status.items():
                logger.info(f"  {status['name']}: {status['health']} (Avail: {status['availability']:.1f}%)")
        
        # Phase 4: Crash service_d
        logger.info("Phase 4: Crashing service_d...")
        fault_id = self.fault_api.inject_fault(
            "crash",
            "service_d",
            {"restart_after": phase_duration}
        )
        if fault_id:
            self.fault_ids.append(fault_id)
        
        # Take snapshots during phase 4
        phase_end = start_time + phase_duration * 4
        while time.time() < phase_end:
            time.sleep(5)
            services_status = self.monitor.get_all_services_status()
            snapshot = {
                "timestamp": time.time(),
                "relative_time": time.time() - start_time,
                "services_status": services_status,
                "phase": 4,
                "fault_ids": self.fault_ids.copy()
            }
            snapshots.append(snapshot)
            
            logger.info(f"Phase 4 - Snapshot at {snapshot['relative_time']:.1f}s:")
            for service_id, status in services_status.items():
                logger.info(f"  {status['name']}: {status['health']} (Avail: {status['availability']:.1f}%)")
        
        # Get final status
        final_status = self.monitor.get_all_services_status()
        
        return {
            "snapshots": snapshots,
            "final_status": final_status,
            "fault_ids": self.fault_ids,
            "fault_info": {
                fault_id: self.fault_api.get_fault_info(fault_id)
                for fault_id in self.fault_ids
            }
        }
    
    def cleanup(self):
        # Make sure all faults are stopped
        for fault_id in self.fault_ids:
            self.fault_api.stop_fault(fault_id)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.graph_updater.stop_updating()
        
        # Save final graph
        filename = f"{self.data_dir}/cascading_failure_final_graph.png"
        self.graph_updater.get_current_graph().visualize(filename)
        
        return {"status": "monitoring_stopped", "final_graph": filename}


class ScenarioRunner:
    """
    Runner for executing multiple test scenarios.
    """
    def __init__(self, scenarios: List[TestScenario]):
        """
        Initialize the runner.
        
        Args:
            scenarios: List of test scenarios to run
        """
        self.scenarios = scenarios
    
    def run_all(self):
        """Run all test scenarios."""
        results = []
        start_time = time.time()
        
        for scenario in self.scenarios:
            logger.info(f"\nExecuting scenario: {scenario.name}")
            result = scenario.execute()
            results.append(result)
            
            # Wait a bit between scenarios
            time.sleep(5)
        
        end_time = time.time()
        
        # Prepare summary
        summary = {
            "total_scenarios": len(self.scenarios),
            "successful_scenarios": sum(1 for r in results if r.get("success", False)),
            "failed_scenarios": sum(1 for r in results if not r.get("success", False)),
            "total_duration": end_time - start_time,
            "scenarios": [
                {
                    "name": r["name"],
                    "success": r.get("success", False),
                    "duration": (
                        r.get("setup", {}).get("duration", 0) +
                        r.get("execution", {}).get("duration", 0) +
                        r.get("cleanup", {}).get("duration", 0)
                    )
                }
                for r in results
            ]
        }
        
        # Save summary
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"data/test_scenarios/summary_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Test summary saved to {filename}")
        
        return summary