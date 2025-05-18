#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, Any, List

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_heal.orchestration import SystemOrchestrator
from graph_heal.fault_injection import FaultInjector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('demo')

class SystemDemo:
    def __init__(self, config_file: str):
        """
        Initialize the demo system.
        
        Args:
            config_file: Path to the system configuration file
        """
        self.config = self._load_config(config_file)
        self.orchestrator = SystemOrchestrator(self.config)
        self.fault_injector = FaultInjector()
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def run_demo_sequence(self, sequence: List[Dict[str, Any]]):
        """
        Run a sequence of demo actions.
        
        Args:
            sequence: List of demo actions to execute
        """
        try:
            # Start the system
            self.orchestrator.start()
            logger.info("System started successfully")
            
            # Execute each action in the sequence
            for action in sequence:
                logger.info(f"Executing action: {action['description']}")
                
                # Wait for the specified duration
                time.sleep(action.get('wait_before', 0))
    
                # Execute the action
                if action['type'] == 'fault_injection':
                    self._inject_fault(action['fault_type'], action['target'], action['params'])
                elif action['type'] == 'wait':
                    time.sleep(action['duration'])
                
                # Wait after the action
                time.sleep(action.get('wait_after', 0))
                
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Demo error: {e}")
        finally:
            # Stop the system
            self.orchestrator.stop()
            logger.info("System stopped")
    
    def _inject_fault(self, fault_type: str, target: str, params: Dict[str, Any]):
        """
        Inject a fault into the system.
        
        Args:
            fault_type: Type of fault to inject
            target: Target service or component
            params: Fault parameters
        """
        try:
            fault_id = self.fault_injector.inject_fault(fault_type, target, params)
            logger.info(f"Injected {fault_type} fault into {target} (ID: {fault_id})")
        except Exception as e:
            logger.error(f"Failed to inject fault: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Graph-Heal system demo')
    parser.add_argument('--config', type=str, default='config/system_config.json',
                      help='Path to the system configuration file')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Define the demo sequence
    demo_sequence = [
        {
            'type': 'wait',
            'description': 'Initial system warm-up',
            'duration': 30
        },
        {
            'type': 'fault_injection',
            'description': 'Inject latency fault into service_b',
            'fault_type': 'latency',
            'target': 'service_b',
            'params': {'latency_ms': 500, 'duration': 30},
            'wait_before': 10,
            'wait_after': 40
        },
        {
            'type': 'fault_injection',
            'description': 'Inject CPU stress into service_c',
            'fault_type': 'cpu_stress',
            'target': 'service_c',
            'params': {'load': 80, 'duration': 30},
            'wait_before': 20,
            'wait_after': 40
        },
        {
            'type': 'fault_injection',
            'description': 'Inject memory leak into service_a',
            'fault_type': 'memory_leak',
            'target': 'service_a',
            'params': {'memory_mb': 50, 'duration': 30},
            'wait_before': 20,
            'wait_after': 40
        },
        {
            'type': 'wait',
            'description': 'Final observation period',
            'duration': 60
        }
    ]
    
    # Create and run the demo
    demo = SystemDemo(args.config)
    demo.run_demo_sequence(demo_sequence)

if __name__ == '__main__':
    main()