import numpy as np
import pandas as pd
import time
import logging
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import deque, defaultdict
import json
import os
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fault_localization')

class FaultLocalizer:
    """
    Base class for fault localization.
    """
    def __init__(self, data_dir: str = "data/faults"):
        """
        Initialize the fault localizer.
        
        Args:
            data_dir: Directory to store fault data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.faults: List[Dict[str, Any]] = []
        self.active_faults: Dict[str, Dict[str, Any]] = {}
    
    def localize_faults(self, service_statuses: Dict[str, Dict[str, Any]], anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Localize faults based on service statuses and detected anomalies.
        
        Args:
            service_statuses: Current status of all services
            anomalies: List of detected anomalies
        
        Returns:
            List of localized faults
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def log_fault(self, fault: Dict[str, Any]):
        """
        Log a fault to the data directory.
        
        Args:
            fault: Fault information to log
        """
        try:
            timestamp = int(time.time())
            filename = f"fault_{fault['id']}_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(fault, f, indent=2)
                
            logger.info(f"Logged fault to {filepath}")
        except Exception as e:
            logger.error(f"Failed to log fault: {e}")

    def get_active_faults(self) -> List[Dict[str, Any]]:
        """
        Get currently active faults.
        
        Returns:
            List of active faults
        """
        return list(self.active_faults.values())
    
    def get_all_faults(self) -> List[Dict[str, Any]]:
        """
        Get all detected faults.
        
        Returns:
            List of all faults
        """
        return self.faults

class GraphBasedFaultLocalizer(FaultLocalizer):
    """
    Localizes faults using graph-based analysis.
    """
    def __init__(self, data_dir: str = "data/faults", confidence_threshold: float = 0.7):
        """
        Initialize the graph-based fault localizer.
        
        Args:
            data_dir: Directory to store fault data
            confidence_threshold: Confidence threshold for fault localization
        """
        super().__init__(data_dir)
        self.confidence_threshold = confidence_threshold
        self.fault_patterns: Dict[str, Dict[str, Any]] = {
            "single_point": {
                "description": "Single service failure affecting dependent services",
                "confidence_threshold": self.confidence_threshold + 0.1
            },
            "cascading": {
                "description": "Fault propagating through service dependencies",
                "confidence_threshold": self.confidence_threshold
            },
            "resource_contention": {
                "description": "Multiple services affected by resource constraints",
                "confidence_threshold": self.confidence_threshold - 0.1
            }
        }
    
    def localize_faults(self, service_statuses: Dict[str, Dict[str, Any]], anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Localize faults using graph-based analysis.
        
        Args:
            service_statuses: Current status of all services
            anomalies: List of detected anomalies
        
        Returns:
            List of localized faults
        """
        current_time = time.time()
        new_faults = []
        
        # Create service dependency graph
        graph = nx.DiGraph()
        for service_id, status in service_statuses.items():
            graph.add_node(service_id, **status)
            for dep_id in status.get("dependencies", []):
                if dep_id in service_statuses:
                    graph.add_edge(service_id, dep_id)
        
        # Group anomalies by service
        service_anomalies: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for anomaly in anomalies:
            service_id = anomaly.get("service_id")
            if service_id:
                service_anomalies[service_id].append(anomaly)
        
        # Analyze each service with anomalies
        for service_id, service_anomalies_list in service_anomalies.items():
            # Skip if service is not in the graph
            if service_id not in graph:
                continue
            
            # Get affected services (downstream dependencies)
            affected_services = set(nx.descendants(graph, service_id))
            affected_services.add(service_id)
            
            # Check for fault patterns
            for pattern_name, pattern_info in self.fault_patterns.items():
                if self._matches_pattern(pattern_name, service_id, affected_services, service_statuses, service_anomalies_list):
                    confidence = self._calculate_confidence(
                        pattern_name,
                        service_id,
                        affected_services,
                        service_statuses,
                        service_anomalies_list
                    )
                    
                    if confidence >= pattern_info["confidence_threshold"]:
                        fault = {
                            "id": f"fault_{pattern_name}_{service_id}_{int(current_time)}",
                            "type": pattern_name,
                            "service_id": service_id,
                            "affected_services": list(affected_services),
                            "confidence": confidence,
                            "timestamp": current_time,
                            "description": pattern_info["description"],
                            "related_anomalies": [a["id"] for a in service_anomalies_list],
                            "metrics": self._extract_relevant_metrics(service_id, affected_services, service_statuses)
                        }
                        
                        new_faults.append(fault)
                        self.faults.append(fault)
                        self.active_faults[fault["id"]] = fault
                        self.log_fault(fault)
                        
                        logger.info(f"Localized {pattern_name} fault in {service_id} "
                                  f"affecting {len(affected_services)} services "
                                  f"with confidence {confidence:.2f}")
        
        # Update active faults
        self._update_active_faults()
        
        return new_faults
    
    def _matches_pattern(self, pattern_name: str, service_id: str,
                        affected_services: Set[str],
                        service_statuses: Dict[str, Dict[str, Any]],
                        anomalies: List[Dict[str, Any]]) -> bool:
        """
        Check if the current situation matches a fault pattern.
        
        Args:
            pattern_name: Name of the pattern to check
            service_id: ID of the service being analyzed
            affected_services: Set of affected service IDs
            service_statuses: Current status of all services
            anomalies: List of anomalies for the service
        
        Returns:
            True if the pattern matches, False otherwise
        """
        if pattern_name == "single_point":
            # Check if only one service is unhealthy but affects multiple services
            unhealthy_services = {
                s_id for s_id in affected_services
                if service_statuses[s_id].get("health") == "unhealthy"
            }
            return len(unhealthy_services) == 1 and len(affected_services) > 1
        
        elif pattern_name == "cascading":
            # Check if there's a chain of affected services
            unhealthy_services = {
                s_id for s_id in affected_services
                if service_statuses[s_id].get("health") == "unhealthy"
            }
            return len(unhealthy_services) > 1 and any(
                a.get("type") == "graph" for a in anomalies
            )
        
        elif pattern_name == "resource_contention":
            # Check if multiple services show resource-related metrics anomalies
            resource_metrics = {"cpu_usage", "memory_usage", "latency"}
            return any(
                a.get("metric_name") in resource_metrics for a in anomalies
            )
        
        return False
    
    def _calculate_confidence(self, pattern_name: str, service_id: str,
                            affected_services: Set[str],
                            service_statuses: Dict[str, Dict[str, Any]],
                            anomalies: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for a fault pattern match.
        
        Args:
            pattern_name: Name of the pattern
            service_id: ID of the service being analyzed
            affected_services: Set of affected service IDs
            service_statuses: Current status of all services
            anomalies: List of anomalies for the service
        
        Returns:
            Confidence score between 0 and 1
        """
        if pattern_name == "single_point":
            # Higher confidence if:
            # 1. More services are affected
            # 2. More anomalies are detected
            # 3. Service health is clearly unhealthy
            affected_ratio = len(affected_services) / len(service_statuses)
            anomaly_score = min(len(anomalies) / 5, 1.0)  # Cap at 5 anomalies
            health_score = 1.0 if service_statuses[service_id].get("health") == "unhealthy" else 0.5
            
            return (affected_ratio * 0.4 + anomaly_score * 0.3 + health_score * 0.3)
        
        elif pattern_name == "cascading":
            # Higher confidence if:
            # 1. More services in the cascade
            # 2. Strong correlations between services
            # 3. Clear propagation pattern in timestamps
            cascade_size = len(affected_services)
            size_score = min(cascade_size / 5, 1.0)  # Cap at 5 services
            
            correlation_scores = [
                a.get("correlation", 0) for a in anomalies
                if a.get("type") == "graph"
            ]
            correlation_score = max(correlation_scores) if correlation_scores else 0
            
            return (size_score * 0.6 + correlation_score * 0.4)
        
        elif pattern_name == "resource_contention":
            # Higher confidence if:
            # 1. Multiple resource metrics are anomalous
            # 2. Resource metrics show high deviation
            resource_metrics = {"cpu_usage", "memory_usage", "latency"}
            anomalous_resources = {
                a.get("metric_name") for a in anomalies
                if a.get("metric_name") in resource_metrics
            }
            
            resource_score = len(anomalous_resources) / len(resource_metrics)
            severity_scores = [
                1.0 if a.get("severity") == "high" else 0.5
                for a in anomalies
                if a.get("metric_name") in resource_metrics
            ]
            severity_score = max(severity_scores) if severity_scores else 0
            
            return (resource_score * 0.7 + severity_score * 0.3)
        
        return 0.0
    
    def _extract_relevant_metrics(self, service_id: str,
                                affected_services: Set[str],
                                service_statuses: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Extract relevant metrics for the fault.
        
        Args:
            service_id: ID of the primary service
            affected_services: Set of affected service IDs
            service_statuses: Current status of all services
        
        Returns:
            Dictionary of relevant metrics by service
        """
        metrics = {}
        relevant_metric_names = {
            "health",
            "cpu_usage",
            "memory_usage",
            "latency",
            "error_rate",
            "request_count"
        }
        
        for s_id in affected_services:
            if s_id in service_statuses:
                status = service_statuses[s_id]
                metrics[s_id] = {
                    name: value for name, value in status.get("metrics", {}).items()
                    if name in relevant_metric_names
                }
        
        return metrics
    
    def _update_active_faults(self):
        """Update the list of active faults."""
        current_time = time.time()
        expired_faults = []
        
        for fault_id, fault in self.active_faults.items():
            # Remove faults older than 5 minutes
            if current_time - fault["timestamp"] > 300:
                expired_faults.append(fault_id)
        
        for fault_id in expired_faults:
            del self.active_faults[fault_id]

class FaultManager:
    """
    Manages fault localization and tracking.
    """
    def __init__(self, localizers: List[FaultLocalizer]):
        """
        Initialize the fault manager.
        
        Args:
            localizers: List of fault localizers to use
        """
        self.localizers = localizers
        self.active_faults: Dict[str, Dict[str, Any]] = {}
        self.fault_history: List[Dict[str, Any]] = []
    
    def process_system_state(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process the current system state to detect and localize faults.
        
        Args:
            system_state: Current system state including service statuses and metrics
        
        Returns:
            List of detected faults
        """
        # Get service statuses and anomalies from system state
        service_statuses = system_state.get("service_statuses", {})
        anomalies = system_state.get("anomalies", [])
        
        # Use all localizers to detect faults
        all_faults = []
        for localizer in self.localizers:
            try:
                faults = localizer.localize_faults(service_statuses, anomalies)
                all_faults.extend(faults)
            except Exception as e:
                logger.error(f"Error in fault localizer: {e}")
        
        # Update active faults
        current_time = time.time()
        for fault in all_faults:
            fault_id = fault["id"]
            if fault_id in self.active_faults:
                # Update existing fault
                self.active_faults[fault_id].update(fault)
                self.active_faults[fault_id]["last_seen"] = current_time
            else:
                # Add new fault
                fault["first_seen"] = current_time
                fault["last_seen"] = current_time
                self.active_faults[fault_id] = fault
        
        # Remove resolved faults
        resolved_faults = []
        for fault_id, fault in list(self.active_faults.items()):
            if current_time - fault["last_seen"] > 300:  # 5 minutes
                resolved_faults.append(fault)
                del self.active_faults[fault_id]
        
        # Update fault history
        self.fault_history.extend(resolved_faults)
        
        return all_faults
    
    def localize_faults(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Localize faults using all available localizers.
        
        Args:
            system_state: Current system state
        
        Returns:
            List of localized faults
        """
        all_faults = []
        
        for localizer in self.localizers:
            try:
                faults = localizer.localize_faults(
                    system_state.get("service_statuses", {}),
                    system_state.get("anomalies", [])
                )
                all_faults.extend(faults)
            except Exception as e:
                logger.error(f"Error in fault localizer: {e}")
        
        # Update active faults
        current_time = time.time()
        for fault in all_faults:
            fault_id = fault["id"]
            if fault_id in self.active_faults:
                # Update existing fault
                self.active_faults[fault_id].update(fault)
                self.active_faults[fault_id]["last_seen"] = current_time
            else:
                # Add new fault
                fault["first_seen"] = current_time
                fault["last_seen"] = current_time
                self.active_faults[fault_id] = fault
        
        # Remove resolved faults
        resolved_faults = []
        for fault_id, fault in list(self.active_faults.items()):
            if current_time - fault["last_seen"] > 300:  # 5 minutes
                resolved_faults.append(fault)
                del self.active_faults[fault_id]
        
        # Update fault history
        self.fault_history.extend(resolved_faults)
        
        return all_faults
    
    def get_active_faults(self, max_age_seconds: int = 300) -> List[Dict[str, Any]]:
        """
        Get currently active faults.
        
        Args:
            max_age_seconds: Maximum age of faults to consider active
        
        Returns:
            List of active faults
        """
        current_time = time.time()
        return [
            fault for fault in self.active_faults.values()
            if current_time - fault["last_seen"] <= max_age_seconds
        ]
    
    def get_all_faults(self) -> List[Dict[str, Any]]:
        """
        Get all faults (active and historical).
        
        Returns:
            List of all faults
        """
        return list(self.active_faults.values()) + self.fault_history