import os
import time
import json
import logging
import requests
from typing import Dict, Any, List, Optional
from threading import Thread, Event

logger = logging.getLogger(__name__)

class ServiceMonitor:
    """Monitor service health and metrics."""
    
    def __init__(self, services: List[Dict[str, str]], poll_interval: float = 1.0, metrics_interval: float = 5.0):
        """
        Initialize the service monitor.
        
        Args:
            services: List of service configurations
            poll_interval: Time between service status checks in seconds
            metrics_interval: Time between metrics collection in seconds
        """
        self.poll_interval = poll_interval
        self.metrics_interval = metrics_interval
        self.services = {
            service["id"]: f"{service['url']}{service['health_endpoint']}"
            for service in services
        }
        self.stop_event = Event()
        self.monitor_thread = None
        self.service_statuses = {}
        self.service_metrics = {}
        self.min_poll_interval = 0.5
        self.max_poll_interval = 10.0
        self.anomaly_history = []  # List of bools: True if anomaly detected in interval
        self.anomaly_window = 10   # Number of intervals to consider
        
        # Create data directory if it doesn't exist
        self.data_dir = "data/monitoring"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return
        
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Started service monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        if not self.monitor_thread:
            return
        
        self.stop_event.set()
        self.monitor_thread.join()
        self.monitor_thread = None
        logger.info("Stopped service monitoring")
    
    def get_service_status(self, service_id: str) -> Dict[str, Any]:
        """
        Get the current status of a specific service.
        
        Args:
            service_id: ID of the service to check
            
        Returns:
            Dictionary containing service status and metrics
        """
        if service_id not in self.services:
            raise ValueError(f"Unknown service: {service_id}")
        
        return self.service_statuses.get(service_id, {
            "status": "unknown",
            "metrics": {}
        })
    
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current status of all services.
        
        Returns:
            Dictionary mapping service IDs to their status and metrics
        """
        return self.service_statuses.copy()
    
    def get_latest_metrics(self, service_id: str) -> Dict[str, Any]:
        """
        Get the latest metrics for a specific service.
        
        Args:
            service_id: ID of the service to get metrics for
            
        Returns:
            Dictionary containing service metrics
        """
        return self.service_metrics.get(service_id, {})
    
    def get_all_services_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the latest metrics for all services.
        
        Returns:
            Dictionary mapping service IDs to their metrics
        """
        return self.service_metrics.copy()
    
    def _check_service_health(self, service_id: str, health_url: str) -> Dict[str, Any]:
        """
        Check the health of a service.
        
        Args:
            service_id: ID of the service to check
            health_url: URL of the service's health endpoint
            
        Returns:
            Dictionary containing service health status and metrics
        """
        try:
            start_time = time.time()
            response = requests.get(health_url, timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                metrics = response.json()
                metrics["response_time"] = response_time
                self.service_metrics[service_id] = metrics
                return {
                    "status": "healthy",
                    "metrics": metrics
                }
            else:
                return {
                    "status": "unhealthy",
                    "metrics": {
                        "response_time": response_time,
                        "status_code": response.status_code
                    }
                }
                
        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "metrics": {
                    "response_time": 5.0
                }
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "unreachable",
                "metrics": {}
            }
        except Exception as e:
            logger.error(f"Error checking {service_id} health: {e}")
            return {
                "status": "error",
                "metrics": {
                    "error": str(e)
                }
            }
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop with adaptive polling and batch processing."""
        while not self.stop_event.is_set():
            timestamp = time.time()
            batch_metrics = {}
            batch_statuses = {}
            anomalies_this_interval = False
            # Batch: collect all service metrics first
            for service_id, health_url in self.services.items():
                status = self._check_service_health(service_id, health_url)
                status["timestamp"] = timestamp
                batch_statuses[service_id] = status
                batch_metrics[service_id] = status.get("metrics", {})
                # Detect anomaly: unhealthy, timeout, or error
                if status["status"] not in ["healthy", "unreachable"]:
                    anomalies_this_interval = True
            # Update all at once (batch)
            self.service_statuses = batch_statuses
            self.service_metrics = batch_metrics
            # Save status snapshot
            snapshot = {
                "timestamp": timestamp,
                "services": self.service_statuses
            }
            snapshot_file = os.path.join(
                self.data_dir,
                f"status_{int(timestamp)}.json"
            )
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f, indent=2)
            # Adaptive polling: adjust poll_interval based on anomaly rate
            self.anomaly_history.append(anomalies_this_interval)
            if len(self.anomaly_history) > self.anomaly_window:
                self.anomaly_history = self.anomaly_history[-self.anomaly_window:]
            anomaly_rate = sum(self.anomaly_history) / len(self.anomaly_history)
            if anomaly_rate > 0.3:
                # More anomalies: poll faster
                self.poll_interval = max(self.min_poll_interval, self.poll_interval * 0.7)
            elif anomaly_rate == 0:
                # No anomalies: poll slower
                self.poll_interval = min(self.max_poll_interval, self.poll_interval * 1.2)
            # Wait for next check
            time.sleep(self.poll_interval) 