import time
import threading
import requests
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import os
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
from .prometheus_metrics import format_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('monitoring')

# Create a custom registry
REGISTRY = CollectorRegistry()

# Define metrics
REQUEST_COUNT = Counter(
    'service_request_total',
    'Total number of requests',
    ['service', 'endpoint', 'method', 'status'],
    registry=REGISTRY
)

REQUEST_LATENCY = Histogram(
    'service_request_duration_seconds',
    'Request latency in seconds',
    ['service', 'endpoint'],
    registry=REGISTRY
)

SERVICE_HEALTH = Gauge(
    'service_health',
    'Service health status (1 for healthy, 0 for unhealthy)',
    ['service'],
    registry=REGISTRY
)

SERVICE_AVAILABILITY = Gauge(
    'service_availability_percentage',
    'Service availability percentage',
    ['service'],
    registry=REGISTRY
)

class ServiceMonitor:
    """
    Monitor for tracking service health and metrics.
    """
    def __init__(self, services_config: List[Dict[str, str]], poll_interval: int = 5):
        """
        Initialize the service monitor.
        
        Args:
            services_config: List of service configurations
            poll_interval: Polling interval in seconds
        """
        self.services = services_config
        self.poll_interval = poll_interval
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.service_status = {}
        self.service_metrics = {}
        self.availability_history = {s["id"]: [] for s in services_config}
        self.last_check = {s["id"]: None for s in services_config}
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return
        
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Started service monitoring")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join()
            logger.info("Stopped service monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            for service in self.services:
                try:
                    # Check health
                    health_url = f"{service['url']}{service['health_endpoint']}"
                    health_response = requests.get(health_url, timeout=1)
                    is_healthy = health_response.status_code == 200
                    
                    # Update health status
                    current_time = datetime.now()
                    self.service_status[service["id"]] = {
                        "name": service["name"],
                        "health": "healthy" if is_healthy else "unhealthy",
                        "last_check": current_time.isoformat()
                    }
                    
                    # Update availability history
                    self.availability_history[service["id"]].append(is_healthy)
                    # Keep last hour of history (assuming 5s interval)
                    max_history = 720  # 3600s / 5s = 720 samples
                    if len(self.availability_history[service["id"]]) > max_history:
                        self.availability_history[service["id"]] = self.availability_history[service["id"]][-max_history:]
                    
                    # Calculate availability percentage
                    history = self.availability_history[service["id"]]
                    availability = (sum(history) / len(history)) * 100 if history else 0
                    self.service_status[service["id"]]["availability"] = availability
                    
                    # Get metrics if healthy
                    if is_healthy:
                        metrics_url = f"{service['url']}{service['metrics_endpoint']}"
                        metrics_response = requests.get(metrics_url, timeout=1)
                        if metrics_response.status_code == 200:
                            metrics = metrics_response.json()
                            # Add timestamp to metrics
                            metrics["timestamp"] = current_time.isoformat()
                            self.service_metrics[service["id"]] = metrics
                    
                    self.last_check[service["id"]] = current_time.isoformat()
                    
                except requests.RequestException as e:
                    logger.warning(f"Failed to check {service['name']}: {e}")
                    current_time = datetime.now()
                    self.service_status[service["id"]] = {
                        "name": service["name"],
                        "health": "unhealthy",
                        "last_check": current_time.isoformat(),
                        "error": str(e)
                    }
                    
                    # Update availability history
                    self.availability_history[service["id"]].append(False)
                    if len(self.availability_history[service["id"]]) > 720:
                        self.availability_history[service["id"]] = self.availability_history[service["id"]][-720:]
                    
                    # Calculate availability percentage
                    history = self.availability_history[service["id"]]
                    availability = (sum(history) / len(history)) * 100 if history else 0
                    self.service_status[service["id"]]["availability"] = availability
            
            # Sleep until next check
            time.sleep(self.poll_interval)
    
    def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a service."""
        return self.service_status.get(service_id)
    
    def get_service_metrics(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get the current metrics of a service."""
        return self.service_metrics.get(service_id)
    
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the current status of all services."""
        return self.service_status.copy()
    
    def get_all_services_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get the current metrics of all services."""
        return self.service_metrics.copy()

class GraphUpdater:
    """
    Updates the system graph based on monitoring data.
    """
    def __init__(self, graph, monitor: ServiceMonitor, update_interval: int = 5):
        """
        Initialize the graph updater.
        
        Args:
            graph: SystemGraph instance
            monitor: ServiceMonitor instance
            update_interval: Update interval in seconds
        """
        self.graph = graph
        self.monitor = monitor
        self.update_interval = update_interval
        self.stop_event = threading.Event()
        self.updater_thread = None
    
    def start_updating(self):
        """Start the graph update thread."""
        if self.updater_thread and self.updater_thread.is_alive():
            logger.warning("Graph updater thread already running")
            return
        
        self.stop_event.clear()
        self.updater_thread = threading.Thread(target=self._update_loop)
        self.updater_thread.daemon = True
        self.updater_thread.start()
        logger.info("Started graph updating")
    
    def stop_updating(self):
        """Stop the graph update thread."""
        if self.updater_thread and self.updater_thread.is_alive():
            self.stop_event.set()
            self.updater_thread.join()
            logger.info("Stopped graph updating")
    
    def _update_loop(self):
        """Main graph update loop."""
        while not self.stop_event.is_set():
            try:
                # Get current status and metrics
                services_status = self.monitor.get_all_services_status()
                services_metrics = self.monitor.get_all_services_metrics()
                
                # Update node status in graph
                for service_id, status in services_status.items():
                    self.graph.update_node_status(service_id, status["health"])
                    
                    # Update node metrics if available
                    if service_id in services_metrics:
                        node = self.graph.get_node_by_id(service_id)
                        if node:
                            node.metrics = services_metrics[service_id]
                
                # Save current state
                self.graph.save_to_file("data/graphs/current_graph.json")
                
            except Exception as e:
                logger.error(f"Error updating graph: {e}")
            
            # Sleep until next update
            time.sleep(self.update_interval)
    
    def get_current_graph(self):
        """Get the current graph."""
        return self.graph