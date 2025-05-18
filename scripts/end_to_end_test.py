#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import argparse
import unittest
from typing import Dict, Any, List
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_heal.orchestration import SystemOrchestrator
from graph_heal.fault_injection import FaultInjector
from graph_heal.monitoring import ServiceMonitor
from graph_heal.anomaly_detection import StatisticalAnomalyDetector, GraphAnomalyDetector
from graph_heal.fault_localization import CausalAnalyzer
from graph_heal.recovery import RecoveryActionFactory, RecoveryDecisionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('end_to_end_test')

class EndToEndTest(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.config = self._load_config()
        self.orchestrator = SystemOrchestrator(self.config)
        self.fault_injector = FaultInjector()
        self.monitor = ServiceMonitor(self.config['services'], poll_interval=2)
        self.statistical_detector = StatisticalAnomalyDetector()
        self.graph_detector = GraphAnomalyDetector()
        self.fault_localizer = CausalAnalyzer()
        self.recovery_factory = RecoveryActionFactory()
        self.recovery_engine = RecoveryDecisionEngine(self.recovery_factory)
    
    def tearDown(self):
        """Clean up test environment."""
        self.orchestrator.stop()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load test configuration."""
        try:
            with open('config/test_config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load test configuration: {e}")
            sys.exit(1)
    
    def test_system_initialization(self):
        """Test system initialization and basic functionality."""
        # Start the system
        self.orchestrator.start()
        time.sleep(5)  # Wait for initialization
        
        # Check if all services are healthy
        services_status = self.monitor.get_all_services_status()
        for service_id, status in services_status.items():
            self.assertEqual(status['health'], 'healthy',
                           f"Service {service_id} is not healthy")
    
    def test_fault_detection(self):
        """Test fault detection capabilities."""
        # Start the system
        self.orchestrator.start()
        time.sleep(5)
            
        # Inject a latency fault
        fault_id = self.fault_injector.inject_fault(
            'latency',
            'service_b',
            {'latency_ms': 500, 'duration': 30}
        )
        self.assertIsNotNone(fault_id, "Failed to inject fault")
        
        # Wait for detection
        time.sleep(10)
        
        # Check if fault was detected
        anomalies = self.statistical_detector.get_detected_anomalies()
        self.assertTrue(len(anomalies) > 0, "No anomalies detected")
    
    def test_fault_localization(self):
        """Test fault localization capabilities."""
        # Start the system
        self.orchestrator.start()
        time.sleep(5)
        
        # Inject a CPU stress fault
        fault_id = self.fault_injector.inject_fault(
            'cpu_stress',
            'service_c',
            {'load': 80, 'duration': 30}
        )
        self.assertIsNotNone(fault_id, "Failed to inject fault")
        
        # Wait for detection and localization
        time.sleep(15)
        
        # Check if fault was localized
        root_causes = self.fault_localizer.get_root_causes()
        self.assertTrue(len(root_causes) > 0, "No root causes identified")

    def test_recovery_actions(self):
        """Test recovery action execution."""
        # Start the system
        self.orchestrator.start()
        time.sleep(5)
        
        # Inject a memory leak fault
        fault_id = self.fault_injector.inject_fault(
            'memory_leak',
            'service_a',
            {'memory_mb': 50, 'duration': 30}
        )
        self.assertIsNotNone(fault_id, "Failed to inject fault")
        
        # Wait for detection and recovery
        time.sleep(20)
        
        # Check if recovery actions were executed
        recovery_actions = self.recovery_engine.get_executed_actions()
        self.assertTrue(len(recovery_actions) > 0, "No recovery actions executed")
    
    def test_system_resilience(self):
        """Test system resilience to multiple faults."""
        # Start the system
        self.orchestrator.start()
        time.sleep(5)
        
        # Inject multiple faults
        faults = []
        for service_id, fault_type in [
            ('service_a', 'latency'),
            ('service_b', 'cpu_stress'),
            ('service_c', 'memory_leak')
        ]:
            fault_id = self.fault_injector.inject_fault(
                fault_type,
                service_id,
                {'duration': 30}
            )
            faults.append(fault_id)
            time.sleep(5)
        
        # Wait for system to handle faults
        time.sleep(40)
    
        # Check if system recovered
        services_status = self.monitor.get_all_services_status()
        healthy_services = sum(1 for status in services_status.values()
                             if status['health'] == 'healthy')
        self.assertGreaterEqual(healthy_services, 2,
                              "System did not recover from multiple faults")

def run_tests():
    """Run all end-to-end tests."""
    # Create test results directory
    results_dir = 'data/test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Configure test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(EndToEndTest)
    result = runner.run(suite)
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'test_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success': result.wasSuccessful()
        }, f, indent=2)
    
    return result.wasSuccessful()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run end-to-end tests')
    parser.add_argument('--config', type=str, default='config/test_config.json',
                      help='Path to the test configuration file')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Run tests
    success = run_tests()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()