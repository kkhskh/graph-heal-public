import numpy as np
import pandas as pd
import time
import logging
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import deque
import json
import os
import datetime
import threading
import docker
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('anomaly_detection')

class AnomalyDetector:
    """
    Base class for anomaly detection.
    """
    def __init__(self, data_dir: str = "data/anomalies"):
        """
        Initialize the anomaly detector.
        
        Args:
            data_dir: Directory to store anomaly data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def detect_anomalies(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the provided data.
        
        Returns:
            List of anomalies, each as a dictionary
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def log_anomaly(self, anomaly: Dict[str, Any]):
        """
        Log an anomaly to the data directory.
        
        Args:
            anomaly: Anomaly information
        """
        try:
            # Create a unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            anomaly_id = anomaly.get("id", "unknown")
            filename = f"{self.data_dir}/anomaly_{timestamp}_{anomaly_id}.json"
            
            with open(filename, 'w') as f:
                json.dump(anomaly, f, indent=2)
                
            logger.debug(f"Logged anomaly to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to log anomaly: {e}")


class StatisticalAnomalyDetector(AnomalyDetector):
    """
    Detects anomalies using statistical methods.
    """
    def __init__(self, window_size: int = 10, z_score_threshold: float = 3.0, data_dir: str = "data/anomalies"):
        """
        Initialize the statistical anomaly detector.
        
        Args:
            window_size: Size of the moving window for statistics
            z_score_threshold: Threshold for z-score based anomaly detection
            data_dir: Directory to store anomaly data
        """
        super().__init__(data_dir)
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.metrics_history: Dict[str, Dict[str, List[float]]] = {}
        self.moving_averages: Dict[str, Dict[str, float]] = {}
        self.moving_stds: Dict[str, Dict[str, float]] = {}
    
    def update_statistics(self, service_id: str, metrics: Dict[str, Any]):
        """
        Update the statistical model with new metrics.
        
        Args:
            service_id: ID of the service
            metrics: Current metrics for the service
        """
        # Initialize history for the service if not exists
        if service_id not in self.metrics_history:
            self.metrics_history[service_id] = {}
            self.moving_averages[service_id] = {}
            self.moving_stds[service_id] = {}
        
        # Update history for each metric
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                if metric_name not in self.metrics_history[service_id]:
                    self.metrics_history[service_id][metric_name] = []
                
                # Append new value
                self.metrics_history[service_id][metric_name].append(metric_value)
                
                # Keep only the last window_size values
                if len(self.metrics_history[service_id][metric_name]) > self.window_size:
                    self.metrics_history[service_id][metric_name] = self.metrics_history[service_id][metric_name][-self.window_size:]
                
                # Update moving average and std
                if len(self.metrics_history[service_id][metric_name]) > 1:
                    self.moving_averages[service_id][metric_name] = np.mean(self.metrics_history[service_id][metric_name])
                    self.moving_stds[service_id][metric_name] = np.std(self.metrics_history[service_id][metric_name])
                else:
                    self.moving_averages[service_id][metric_name] = metric_value
                    self.moving_stds[service_id][metric_name] = 0.0
    
    def detect_anomalies(self, service_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the provided metrics.
        
        Args:
            service_metrics: Dictionary of service IDs to their metrics
        
        Returns:
            List of anomalies
        """
        anomalies = []
        
        for service_id, metrics in service_metrics.items():
            # Update statistics
            self.update_statistics(service_id, metrics)
            
            # Check for anomalies
            service_anomalies = self._detect_service_anomalies(service_id, metrics)
            anomalies.extend(service_anomalies)
        
        return anomalies
    
    def _detect_service_anomalies(self, service_id: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies for a specific service.
        
        Args:
            service_id: ID of the service
            metrics: Current metrics for the service
        
        Returns:
            List of anomalies for this service
        """
        anomalies = []
        
        # Skip if we don't have enough history
        if service_id not in self.metrics_history:
            return anomalies
        
        for metric_name, metric_value in metrics.items():
            if not isinstance(metric_value, (int, float)):
                continue
            
            if metric_name not in self.metrics_history[service_id]:
                continue
            
            # Skip if we don't have enough history
            if len(self.metrics_history[service_id][metric_name]) < 2:
                continue
            
            # Calculate z-score
            avg = self.moving_averages[service_id][metric_name]
            std = self.moving_stds[service_id][metric_name]
            
            # Avoid division by zero
            if std == 0:
                continue
            
            z_score = abs(metric_value - avg) / std
            
            # Check if this is an anomaly
            if z_score > self.z_score_threshold:
                anomaly = {
                    "id": f"stat_anomaly_{service_id}_{metric_name}_{int(time.time())}",
                    "type": "statistical",
                    "service_id": service_id,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "expected_value": avg,
                    "z_score": z_score,
                    "threshold": self.z_score_threshold,
                    "timestamp": time.time(),
                    "severity": self._calculate_severity(z_score)
                }
                
                anomalies.append(anomaly)
                self.log_anomaly(anomaly)
                
                logger.info(f"Detected statistical anomaly: {service_id}.{metric_name} = {metric_value} "
                           f"(z-score: {z_score:.2f}, expected: {avg:.2f}±{std:.2f})")
        
        return anomalies
    
    def _calculate_severity(self, z_score: float) -> str:
        """
        Calculate the severity of an anomaly based on its z-score.
        
        Args:
            z_score: Z-score of the anomaly
        
        Returns:
            Severity level: low, medium, or high
        """
        if z_score <= self.z_score_threshold * 1.5:
            return "low"
        elif z_score <= self.z_score_threshold * 3:
            return "medium"
        else:
            return "high"


class GraphAnomalyDetector(AnomalyDetector):
    """
    Detects anomalies using graph-based patterns.
    """
    def __init__(self, data_dir: str = "data/anomalies"):
        """
        Initialize the graph anomaly detector.
        
        Args:
            data_dir: Directory to store anomaly data
        """
        super().__init__(data_dir)
        self.previous_graph = None
    
    def detect_anomalies(self, current_graph, service_statuses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the graph.
        
        Args:
            current_graph: Current system graph
            service_statuses: Dictionary of service statuses
        
        Returns:
            List of anomalies
        """
        anomalies = []
        
        # Skip if this is the first graph we're seeing
        if self.previous_graph is None:
            self.previous_graph = current_graph.copy()
            return anomalies
        
        # Detect connectivity issues
        connectivity_anomalies = self._detect_connectivity_anomalies(current_graph, service_statuses)
        anomalies.extend(connectivity_anomalies)
        
        # Detect path disruptions
        path_anomalies = self._detect_path_disruptions(current_graph, self.previous_graph)
        anomalies.extend(path_anomalies)
        
        # Detect known patterns
        pattern_anomalies = self._detect_known_patterns(current_graph, service_statuses)
        anomalies.extend(pattern_anomalies)
        
        # Update previous graph
        self.previous_graph = current_graph.copy()
        
        return anomalies
    
    def _detect_connectivity_anomalies(self, graph, service_statuses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect connectivity anomalies in the graph.
        
        Args:
            graph: Current system graph
            service_statuses: Dictionary of service statuses
        
        Returns:
            List of connectivity anomalies
        """
        anomalies = []
        
        # Check if any unhealthy service has healthy neighbors
        for node_id in graph.nodes():
            if node_id not in service_statuses:
                continue
            
            if service_statuses[node_id].get("health") != "unhealthy":
                continue
            
            # Check if this node has healthy neighbors
            has_healthy_neighbor = False
            for neighbor in graph.neighbors(node_id):
                if neighbor in service_statuses and service_statuses[neighbor].get("health") == "healthy":
                    has_healthy_neighbor = True
                    break
            
            if has_healthy_neighbor:
                anomaly = {
                    "id": f"connectivity_anomaly_{node_id}_{int(time.time())}",
                    "type": "connectivity",
                    "service_id": node_id,
                    "timestamp": time.time(),
                    "description": f"Service {node_id} is unhealthy but has healthy neighbors",
                    "severity": "medium"
                }
                
                anomalies.append(anomaly)
                self.log_anomaly(anomaly)
                
                logger.info(f"Detected connectivity anomaly: {node_id} is unhealthy but has healthy neighbors")
        
        return anomalies
    
    def _detect_path_disruptions(self, current_graph, previous_graph) -> List[Dict[str, Any]]:
        """
        Detect path disruptions in the graph.
        
        Args:
            current_graph: Current system graph
            previous_graph: Previous system graph
        
        Returns:
            List of path disruption anomalies
        """
        anomalies = []
        
        # Check if any paths that existed before are now broken
        for source in previous_graph.nodes():
            for target in previous_graph.nodes():
                if source == target:
                    continue
                
                # Check if there was a path before
                had_path = nx.has_path(previous_graph, source, target)
                
                # Check if there's a path now
                has_path = nx.has_path(current_graph, source, target)
                
                if had_path and not has_path:
                    anomaly = {
                        "id": f"path_disruption_{source}_{target}_{int(time.time())}",
                        "type": "path_disruption",
                        "source": source,
                        "target": target,
                        "timestamp": time.time(),
                        "description": f"Path from {source} to {target} is disrupted",
                        "severity": "high"
                    }
                    
                    anomalies.append(anomaly)
                    self.log_anomaly(anomaly)
                    
                    logger.info(f"Detected path disruption: Path from {source} to {target} is disrupted")
        
        return anomalies
    
    def _detect_known_patterns(self, graph, service_statuses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect known problematic patterns in the graph.
        
        Args:
            graph: Current system graph
            service_statuses: Dictionary of service statuses
        
        Returns:
            List of pattern anomalies
        """
        anomalies = []
        
        # Check for cascading failures pattern
        # This is a pattern where multiple connected services are unhealthy
        unhealthy_nodes = [
            node_id for node_id in graph.nodes()
            if node_id in service_statuses and service_statuses[node_id].get("health") == "unhealthy"
        ]
        
        if len(unhealthy_nodes) >= 2:
            # Check if the unhealthy nodes form a connected subgraph
            subgraph = graph.subgraph(unhealthy_nodes)
            
            if nx.is_connected(subgraph.to_undirected()):
                anomaly = {
                    "id": f"cascading_failure_{int(time.time())}",
                    "type": "cascading_failure",
                    "affected_services": unhealthy_nodes,
                    "timestamp": time.time(),
                    "description": f"Detected cascading failure pattern across {len(unhealthy_nodes)} services",
                    "severity": "high"
                }
                
                anomalies.append(anomaly)
                self.log_anomaly(anomaly)
                
                logger.info(f"Detected cascading failure pattern across {len(unhealthy_nodes)} services")
        
        # Check for bottleneck pattern
        # This is a pattern where a service is a critical point of connection between other services
        for node_id in graph.nodes():
            if nx.is_isolate(graph, node_id):
                continue
            
            # Create a copy of the graph without this node
            temp_graph = graph.copy()
            temp_graph.remove_node(node_id)
            
            # Check if removing this node disconnects the graph
            if not nx.is_connected(temp_graph.to_undirected()):
                # This node is a cut vertex, i.e., a bottleneck
                anomaly = {
                    "id": f"bottleneck_{node_id}_{int(time.time())}",
                    "type": "bottleneck",
                    "service_id": node_id,
                    "timestamp": time.time(),
                    "description": f"Service {node_id} is a bottleneck in the system",
                    "severity": "medium"
                }
                
                anomalies.append(anomaly)
                self.log_anomaly(anomaly)
                
                logger.info(f"Detected bottleneck pattern: {node_id} is a critical connection point")
        
        return anomalies


class AnomalyManager:
    """
    Manages multiple anomaly detectors and coordinates detection.
    """
    def __init__(self, detectors: List[AnomalyDetector]):
        """
        Initialize the anomaly manager.
        
        Args:
            detectors: List of anomaly detectors
        """
        self.detectors = detectors
        self.anomalies: List[Dict[str, Any]] = []
        self.data_dir = "data/anomalies"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def detect_anomalies(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies using all detectors.
        
        Args:
            system_state: Current state of the system
        
        Returns:
            List of detected anomalies
        """
        new_anomalies = []
        
        # Extract relevant data from system state
        graph = system_state.get("graph")
        service_metrics = system_state.get("service_metrics", {})
        service_statuses = system_state.get("service_statuses", {})
        
        # Run each detector
        for detector in self.detectors:
            try:
                if isinstance(detector, StatisticalAnomalyDetector):
                    anomalies = detector.detect_anomalies(service_metrics)
                elif isinstance(detector, GraphAnomalyDetector):
                    anomalies = detector.detect_anomalies(graph, service_statuses)
                else:
                    # Generic case
                    anomalies = detector.detect_anomalies(system_state)
                
                new_anomalies.extend(anomalies)
            except Exception as e:
                logger.error(f"Error in anomaly detector {detector.__class__.__name__}: {e}")
        
        # Add new anomalies to the list
        self.anomalies.extend(new_anomalies)
        
        # Save all anomalies
        self._save_anomalies()
        
        return new_anomalies
    
    def get_active_anomalies(self, max_age_seconds: int = 300) -> List[Dict[str, Any]]:
        """
        Get active anomalies (detected recently).
        
        Args:
            max_age_seconds: Maximum age of anomalies to consider active
        
        Returns:
            List of active anomalies
        """
        current_time = time.time()
        
        active_anomalies = [
            anomaly for anomaly in self.anomalies
            if current_time - anomaly.get("timestamp", 0) <= max_age_seconds
        ]
        
        return active_anomalies
    
    def get_all_anomalies(self) -> List[Dict[str, Any]]:
        """
        Get all detected anomalies.
        
        Returns:
            List of all anomalies
        """
        return self.anomalies.copy()
    
    def _save_anomalies(self):
        """Save all anomalies to a file."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{self.data_dir}/all_anomalies_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.anomalies, f, indent=2)
            
            logger.debug(f"Saved all anomalies to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save anomalies: {e}")


class RecoveryAction:
    """
    Base class for recovery actions.
    """
    def __init__(self, target_id: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the recovery action.
        
        Args:
            target_id: ID of the target service/component
            params: Additional parameters for the action
        """
        self.target_id = target_id
        self.params = params or {}
        self.success = None
        self.start_time = None
        self.end_time = None
        self.error = None
    
    def execute(self) -> bool:
        """
        Execute the recovery action.
        
        Returns:
            True if the action was successful, False otherwise
        """
        self.start_time = time.time()
        try:
            result = self._execute_impl()
            self.success = result
        except Exception as e:
            logger.error(f"Error executing recovery action: {e}")
            self.success = False
            self.error = str(e)
        
        self.end_time = time.time()
        return self.success
    
    def _execute_impl(self) -> bool:
        """
        Implementation of the execution logic.
        
        Returns:
            True if the action was successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "type": self.__class__.__name__,
            "target_id": self.target_id,
            "params": self.params,
            "success": self.success,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time and self.start_time else None,
            "error": self.error
        }


class ContainerRestartAction(RecoveryAction):
    """
    Action to restart a container.
    """
    def __init__(self, target_id: str, docker_client=None, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the container restart action.
        
        Args:
            target_id: ID of the target container
            docker_client: Docker client
            params: Additional parameters
        """
        super().__init__(target_id, params)
        self.docker_client = docker_client or docker.from_env()
    
    def _execute_impl(self) -> bool:
        """
        Restart the container.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the container
            container = self.docker_client.containers.get(self.target_id)
            
            # Check if it's running
            if container.status != "running":
                logger.info(f"Container {self.target_id} is not running, starting it...")
                container.start()
            else:
                # Stop and restart
                timeout = self.params.get("timeout", 10)
                logger.info(f"Restarting container {self.target_id} with timeout {timeout}s...")
                container.restart(timeout=timeout)
            
            # Wait for container to be healthy
            wait_time = self.params.get("wait_time", 5)
            for _ in range(wait_time):
                container.reload()  # Refresh container status
                if container.status == "running":
                    # Check if the container has health status
                    if hasattr(container, "health") and container.health:
                        if container.health.get("status") == "healthy":
                            break
                time.sleep(1)
            
            logger.info(f"Container {self.target_id} restarted successfully")
            return True
            
        except docker.errors.NotFound:
            logger.error(f"Container {self.target_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error restarting container {self.target_id}: {e}")
            return False


class ServiceReconfigurationAction(RecoveryAction):
    """
    Action to reconfigure a service.
    """
    def __init__(self, target_id: str, config_changes: Dict[str, Any], 
                service_url: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the service reconfiguration action.
        
        Args:
            target_id: ID of the target service
            config_changes: Configuration changes to apply
            service_url: URL of the service
            params: Additional parameters
        """
        super().__init__(target_id, params)
        self.config_changes = config_changes
        self.service_url = service_url
    
    def _execute_impl(self) -> bool:
        """
        Reconfigure the service.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # In a real system, we would have a configuration API endpoint
            # Here we'll simulate by logging the changes
            
            logger.info(f"Reconfiguring service {self.target_id} at {self.service_url}")
            logger.info(f"Configuration changes: {self.config_changes}")
            
            # Simulate API call to update configuration
            # In a real system, this would be:
            # response = requests.post(f"{self.service_url}/config", json=self.config_changes)
            # return response.status_code == 200
            
            # For our demonstration, we'll just pretend it worked
            time.sleep(1)  # Simulate API call delay
            
            logger.info(f"Service {self.target_id} reconfigured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error reconfiguring service {self.target_id}: {e}")
            return False


class TrafficRoutingAction(RecoveryAction):
    """
    Action to reroute traffic.
    """
    def __init__(self, target_id: str, routing_changes: Dict[str, Any], 
                load_balancer_url: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the traffic routing action.
        
        Args:
            target_id: ID of the target service
            routing_changes: Routing changes to apply
            load_balancer_url: URL of the load balancer
            params: Additional parameters
        """
        super().__init__(target_id, params)
        self.routing_changes = routing_changes
        self.load_balancer_url = load_balancer_url
    
    def _execute_impl(self) -> bool:
        """
        Reroute traffic.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # In a real system, we would have a load balancer API
            # Here we'll simulate by logging the changes
            
            logger.info(f"Rerouting traffic for service {self.target_id}")
            logger.info(f"Routing changes: {self.routing_changes}")
            
            # Simulate API call to update routing
            # In a real system, this would be:
            # if self.load_balancer_url:
            #     response = requests.post(f"{self.load_balancer_url}/routes", json=self.routing_changes)
            #     return response.status_code == 200
            
            # For our demonstration, we'll just pretend it worked
            time.sleep(1)  # Simulate API call delay
            
            logger.info(f"Traffic for service {self.target_id} rerouted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error rerouting traffic for service {self.target_id}: {e}")
            return False


class ResourceAdjustmentAction(RecoveryAction):
    """
    Action to adjust resources.
    """
    def __init__(self, target_id: str, resource_changes: Dict[str, Any], 
                docker_client=None, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the resource adjustment action.
        
        Args:
            target_id: ID of the target container
            resource_changes: Resource changes to apply
            docker_client: Docker client
            params: Additional parameters
        """
        super().__init__(target_id, params)
        self.resource_changes = resource_changes
        self.docker_client = docker_client or docker.from_env()
    
    def _execute_impl(self) -> bool:
        """
        Adjust resources.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the container
            container = self.docker_client.containers.get(self.target_id)
            
            # Log the intended resource changes
            logger.info(f"Adjusting resources for container {self.target_id}")
            logger.info(f"Resource changes: {self.resource_changes}")
            
            # In a real production system, we would update container resources
            # This might involve using the Docker API to update container settings
            # or using Kubernetes API to scale resources
            
            # For our demonstration, we'll just log the changes
            # In a real system, this would involve container.update() or similar
            
            # Simulate a delay for the resource adjustment
            time.sleep(1)
            
            logger.info(f"Resources for container {self.target_id} adjusted successfully")
            return True
            
        except docker.errors.NotFound:
            logger.error(f"Container {self.target_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error adjusting resources for container {self.target_id}: {e}")
            return False


class CompositeRecoveryAction(RecoveryAction):
    """
    A composite recovery action that executes multiple actions in sequence.
    """
    def __init__(self, target_id: str, actions: List[RecoveryAction], params: Optional[Dict[str, Any]] = None):
        """
        Initialize the composite recovery action.
        
        Args:
            target_id: ID of the target service/component
            actions: List of recovery actions to execute
            params: Additional parameters
        """
        super().__init__(target_id, params)
        self.actions = actions
        self.action_results = []
    
    def _execute_impl(self) -> bool:
        """
        Execute all actions in sequence.
        
        Returns:
            True if all actions were successful, False otherwise
        """
        all_successful = True
        
        for action in self.actions:
            success = action.execute()
            self.action_results.append(action.to_dict())
            
            if not success:
                all_successful = False
                if self.params.get("fail_fast", True):
                    logger.warning(f"Action {action.__class__.__name__} failed, stopping composite action")
                    break
        
        return all_successful
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result["actions"] = self.action_results
        return result


class RecoveryActionFactory:
    """
    Factory for creating recovery actions.
    """
    def __init__(self, docker_client=None, load_balancer_url: Optional[str] = None):
        """
        Initialize the recovery action factory.
        
        Args:
            docker_client: Docker client
            load_balancer_url: URL of the load balancer
        """
        self.docker_client = docker_client or docker.from_env()
        self.load_balancer_url = load_balancer_url
        self.service_urls = {}
    
    def set_service_url(self, service_id: str, url: str):
        """
        Set the URL for a service.
        
        Args:
            service_id: ID of the service
            url: URL of the service
        """
        self.service_urls[service_id] = url
    
    def create_action(self, action_type: str, target_id: str, 
                     params: Optional[Dict[str, Any]] = None) -> RecoveryAction:
        """
        Create a recovery action.
        
        Args:
            action_type: Type of action
            target_id: ID of the target service/component
            params: Additional parameters
        
        Returns:
            RecoveryAction instance
        """
        params = params or {}
        
        if action_type == "container_restart":
            return ContainerRestartAction(target_id, self.docker_client, params)
        
        elif action_type == "service_reconfiguration":
            service_url = params.get("service_url") or self.service_urls.get(target_id)
            config_changes = params.get("config_changes", {})
            return ServiceReconfigurationAction(target_id, config_changes, service_url, params)
        
        elif action_type == "traffic_routing":
            routing_changes = params.get("routing_changes", {})
            load_balancer_url = params.get("load_balancer_url") or self.load_balancer_url
            return TrafficRoutingAction(target_id, routing_changes, load_balancer_url, params)
        
        elif action_type == "resource_adjustment":
            resource_changes = params.get("resource_changes", {})
            return ResourceAdjustmentAction(target_id, resource_changes, self.docker_client, params)
        
        elif action_type == "composite":
            sub_actions = []
            for sub_action_config in params.get("actions", []):
                sub_type = sub_action_config.get("type")
                sub_target = sub_action_config.get("target_id") or target_id
                sub_params = sub_action_config.get("params", {})
                sub_action = self.create_action(sub_type, sub_target, sub_params)
                sub_actions.append(sub_action)
            
            return CompositeRecoveryAction(target_id, sub_actions, params)
        
        else:
            raise ValueError(f"Unknown action type: {action_type}")


class RecoveryDecisionEngine:
    """
    Engine for deciding which recovery actions to take.
    """
    def __init__(self, action_factory: RecoveryActionFactory, data_dir: str = "data/recovery"):
        """
        Initialize the recovery decision engine.
        
        Args:
            action_factory: RecoveryActionFactory instance
            data_dir: Directory to store recovery data
        """
        self.action_factory = action_factory
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Rules for recovery decision making
        self.rules = [
            self._rule_container_restart,
            self._rule_service_reconfiguration,
            self._rule_traffic_routing,
            self._rule_resource_adjustment,
            self._rule_composite_action
        ]
    
    def decide_recovery_actions(self, fault_reports: List[Dict[str, Any]], 
                               system_state: Dict[str, Any]) -> List[RecoveryAction]:
        """
        Decide which recovery actions to take.
        
        Args:
            fault_reports: List of fault reports
            system_state: Current state of the system
        
        Returns:
            List of recovery actions
        """
        actions = []
        
        for fault in fault_reports:
            # Apply each rule to the fault
            fault_actions = []
            
            for rule in self.rules:
                rule_actions = rule(fault, system_state)
                fault_actions.extend(rule_actions)
            
            # If multiple actions are suggested, choose the one with lowest impact
            if fault_actions:
                best_action = self._choose_lowest_impact_action(fault_actions, system_state)
                actions.append(best_action)
        
        return actions
    
    def _rule_container_restart(self, fault: Dict[str, Any], 
                             system_state: Dict[str, Any]) -> List[RecoveryAction]:
        """
        Rule for container restart.
        
        Args:
            fault: Fault report
            system_state: Current state of the system
        
        Returns:
            List of recovery actions
        """
        actions = []
        
        service_id = fault.get("service_id")
        if not service_id:
            return actions
        
        # Check if the service is unhealthy
        service_statuses = system_state.get("service_statuses", {})
        if service_id in service_statuses and service_statuses[service_id].get("health") == "unhealthy":
            # Container restart is a good option for many types of failures
            actions.append(self.action_factory.create_action(
                "container_restart",
                service_id,
                {"impact": 0.7}  # Medium-high impact
            ))
        
        return actions
    
    def _rule_service_reconfiguration(self, fault: Dict[str, Any], 
                                  system_state: Dict[str, Any]) -> List[RecoveryAction]:
        """
        Rule for service reconfiguration.
        
        Args:
            fault: Fault report
            system_state: Current state of the system
        
        Returns:
            List of recovery actions
        """
        actions = []
        
        service_id = fault.get("service_id")
        if not service_id:
            return actions
        
        # Check root causes for specific issues that could be addressed by reconfiguration
        root_causes = fault.get("root_causes", [])
        
        for cause in root_causes:
            cause_type = cause.get("type")
            
            # For certain types of root causes, reconfiguration might help
            if cause_type in ["upstream_latency", "bottleneck"]:
                # Determine appropriate configuration changes
                config_changes = {}
                
                if cause_type == "upstream_latency":
                    # Increase timeouts
                    config_changes["request_timeout"] = 10000  # 10 seconds
                
                elif cause_type == "bottleneck":
                    # Adjust connection limits
                    config_changes["max_connections"] = 200
                
                actions.append(self.action_factory.create_action(
                    "service_reconfiguration",
                    service_id,
                    {
                        "config_changes": config_changes,
                        "impact": 0.3  # Low impact
                    }
                ))
        
        return actions
    
    def _rule_traffic_routing(self, fault: Dict[str, Any], 
                          system_state: Dict[str, Any]) -> List[RecoveryAction]:
        """
        Rule for traffic routing.
        
        Args:
            fault: Fault report
            system_state: Current state of the system
        
        Returns:
            List of recovery actions
        """
        actions = []
        
        service_id = fault.get("service_id")
        if not service_id:
            return actions
        
        # Check if the service is part of a group
        graph = system_state.get("graph")
        if not graph:
            return actions
        
        # Check if there are alternative services that can handle the load
        # In a real system, we would check for services with the same functionality
        # Here we'll just check for services with outgoing edges to the same targets
        
        alternatives = []
        
        for node in graph.nodes():
            if node != service_id:
                # Check if this node has similar outgoing edges
                node_targets = set(target for _, target in graph.out_edges(node))
                service_targets = set(target for _, target in graph.out_edges(service_id))
                
                if node_targets.intersection(service_targets):
                    alternatives.append(node)
        
        if alternatives:
            # Create routing changes to divert traffic
            routing_changes = {
                "divert_from": service_id,
                "divert_to": alternatives,
                "percentage": 100  # Divert all traffic
            }
            
            actions.append(self.action_factory.create_action(
                "traffic_routing",
                service_id,
                {
                    "routing_changes": routing_changes,
                    "impact": 0.5  # Medium impact
                }
            ))
        
        return actions
    
    def _rule_resource_adjustment(self, fault: Dict[str, Any], 
                              system_state: Dict[str, Any]) -> List[RecoveryAction]:
        """
        Rule for resource adjustment.
        
        Args:
            fault: Fault report
            system_state: Current state of the system
        
        Returns:
            List of recovery actions
        """
        actions = []
        
        service_id = fault.get("service_id")
        if not service_id:
            return actions
        
        # Check for anomalies related to resource issues
        anomalies = fault.get("anomalies", [])
        
        resource_issues = False
        for anomaly in anomalies:
            metric_name = anomaly.get("metric_name", "")
            if "memory" in metric_name.lower() or "cpu" in metric_name.lower():
                resource_issues = True
                break
        
        if resource_issues:
            # Determine the resource changes needed
            resource_changes = {
                "memory": "1g",  # Increase memory limit
                "cpu_shares": 1024  # Increase CPU share
            }
            
            actions.append(self.action_factory.create_action(
                "resource_adjustment",
                service_id,
                {
                    "resource_changes": resource_changes,
                    "impact": 0.4  # Medium-low impact
                }
            ))
        
        return actions
    
    def _rule_composite_action(self, fault: Dict[str, Any], 
                           system_state: Dict[str, Any]) -> List[RecoveryAction]:
        """
        Rule for composite actions.
        
        Args:
            fault: Fault report
            system_state: Current state of the system
        
        Returns:
            List of recovery actions
        """
        actions = []
        
        service_id = fault.get("service_id")
        if not service_id:
            return actions
        
        # Check for severe faults that might need multiple actions
        if fault.get("severity") == "high":
            root_causes = fault.get("root_causes", [])
            
            # Check for cascading failures
            cascading = False
            for cause in root_causes:
                if cause.get("type") == "cascading_failure":
                    cascading = True
                    break
            
            if cascading:
                # Create a composite action for cascading failures
                composite_params = {
                    "actions": [
                        {
                            "type": "traffic_routing",
                            "params": {
                                "routing_changes": {
                                    "divert_from": service_id,
                                    "divert_to": [],  # Will be filled in dynamically
                                    "percentage": 100
                                }
                            }
                        },
                        {
                            "type": "container_restart",
                            "params": {}
                        },
                        {
                            "type": "resource_adjustment",
                            "params": {
                                "resource_changes": {
                                    "memory": "2g",
                                    "cpu_shares": 2048
                                }
                            }
                        }
                    ],
                    "impact": 0.9  # High impact
                }
                
                actions.append(self.action_factory.create_action(
                    "composite",
                    service_id,
                    composite_params
                ))
        
        return actions
    
    def _choose_lowest_impact_action(self, actions: List[RecoveryAction], 
                                  system_state: Dict[str, Any]) -> RecoveryAction:
        """
        Choose the action with the lowest impact.
        
        Args:
            actions: List of recovery actions
            system_state: Current state of the system
        
        Returns:
            The action with the lowest impact
        """
        if not actions:
            raise ValueError("No actions to choose from")
        
        # Sort actions by impact (lower is better)
        sorted_actions = sorted(actions, key=lambda a: a.params.get("impact", 1.0))
        
        # Return the action with the lowest impact
        return sorted_actions[0]
    
    def log_recovery_decision(self, fault: Dict[str, Any], action: RecoveryAction):
        """
        Log a recovery decision.
        
        Args:
            fault: Fault report
            action: Selected recovery action
        """
        try:
            # Create a unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            fault_id = fault.get("id", "unknown")
            action_type = action.__class__.__name__
            
            filename = f"{self.data_dir}/recovery_decision_{timestamp}_{fault_id}_{action_type}.json"
            
            decision_data = {
                "timestamp": timestamp,
                "fault": fault,
                "action": action.to_dict()
            }
            
            with open(filename, 'w') as f:
                json.dump(decision_data, f, indent=2)
            
            logger.debug(f"Logged recovery decision to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to log recovery decision: {e}")