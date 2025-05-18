#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('service_starter')

def start_service(service_id: str, port: int) -> subprocess.Popen:
    """
    Start a service.
    
    Args:
        service_id: ID of the service
        port: Port to run the service on
        
    Returns:
        Process object for the service
    """
    env = os.environ.copy()
    env["FLASK_APP"] = f"services/{service_id}/app.py"
    env["FLASK_ENV"] = "development"
    env["PORT"] = str(port)
    
    process = subprocess.Popen(
        ["flask", "run", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    logger.info(f"Started {service_id} on port {port}")
    return process

def main():
    """Main function."""
    try:
        # Define services and their ports
        services = {
            "service_a": 5001,
            "service_b": 5002,
            "service_c": 5003,
            "service_d": 5004
        }
        
        # Start each service
        processes = {}
        for service_id, port in services.items():
            processes[service_id] = start_service(service_id, port)
        
        # Wait for services to start
        time.sleep(5)
        
        # Check if services are running
        for service_id, process in processes.items():
            if process.poll() is not None:
                logger.error(f"{service_id} failed to start")
                sys.exit(1)
        
        logger.info("All services started successfully")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping services...")
        for process in processes.values():
            process.terminate()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting services: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 