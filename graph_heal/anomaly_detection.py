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
import community as community_louvain

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
        self.anomalies: List[Dict[str, Any]] = []
        self.active_anomalies: Dict[str, Dict[str, Any]] = {}
    
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
    
    def get_active_anomalies(self) -> List[Dict[str, Any]]:
        """
        Get currently active anomalies.
        
        Returns:
            List of active anomalies
        """
        return list(self.active_anomalies.values())
    
    def get_all_anomalies(self) -> List[Dict[str, Any]]:
        """
        Get all detected anomalies.
        
        Returns:
            List of all anomalies
        """
        return self.anomalies


class StatisticalAnomalyDetector(AnomalyDetector):
    """
    Detects anomalies using statistical methods.
    """
    def __init__(self, window_size: int = 10, z_score_threshold: float = 2.5, data_dir: str = "data/anomalies"):
        """
        Initialize the statistical anomaly detector.
        
        Args:
            window_size: Size of the sliding window for statistics
            z_score_threshold: Threshold for z-score based detection
            data_dir: Directory to store anomaly data
        """
        super().__init__(data_dir)
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.metrics_history: Dict[str, Dict[str, List[float]]] = {}
    
    def detect_anomalies(self, service_statuses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies using statistical methods.
        
        Args:
            service_statuses: Current status of all services
        
        Returns:
            List of detected anomalies
        """
        print("[DEBUG] StatisticalAnomalyDetector.detect_anomalies CALLED")
        print("[DEBUG] service_statuses:", service_statuses)
        current_time = time.time()
        new_anomalies = []
        
        for service_id, status in service_statuses.items():
            metrics = status.get("metrics", {})
            
            # Update metrics history
        if service_id not in self.metrics_history:
            self.metrics_history[service_id] = {}
        
        for metric_name, metric_value in metrics.items():
                if not isinstance(metric_value, (int, float)):
                    continue
                
                if metric_name not in self.metrics_history[service_id]:
                    self.metrics_history[service_id][metric_name] = []
                
                self.metrics_history[service_id][metric_name].append(metric_value)
                
                # Keep only recent history
                if len(self.metrics_history[service_id][metric_name]) > self.window_size:
                    self.metrics_history[service_id][metric_name] = self.metrics_history[service_id][metric_name][-self.window_size:]
                
            # Check for anomalies in each metric
            for metric_name, values in self.metrics_history[service_id].items():
                if len(values) < 2:  # Need at least 2 points for statistics
                continue
            
            # Calculate z-score
                mean = np.mean(values[:-1])  # Exclude current value
                std = np.std(values[:-1])
            
                if std == 0:  # Avoid division by zero
                continue
            
                z_score = abs((values[-1] - mean) / std)
                
                print(f"[DEBUG] {service_id}.{metric_name}: value={values[-1]}, mean={mean:.4f}, std={std:.4f}, z_score={z_score:.4f}, threshold={self.z_score_threshold}")
                logger.debug(f"[DEBUG] {service_id}.{metric_name}: value={values[-1]}, mean={mean:.4f}, std={std:.4f}, z_score={z_score:.4f}, threshold={self.z_score_threshold}")
            
            if z_score > self.z_score_threshold:
                anomaly = {
                        "id": f"anomaly_{service_id}_{metric_name}_{int(current_time)}",
                    "type": "statistical",
                    "service_id": service_id,
                    "metric_name": metric_name,
                        "value": values[-1],
                        "mean": mean,
                        "std": std,
                    "z_score": z_score,
                        "timestamp": current_time,
                        "severity": "high" if z_score > self.z_score_threshold * 2 else "medium"
                }
                    new_anomalies.append(anomaly)
                    self.anomalies.append(anomaly)
                    self.active_anomalies[anomaly["id"]] = anomaly
                    logger.info(f"Detected statistical anomaly in {service_id}.{metric_name}: "
                                f"z-score = {z_score:.2f}")
        
        # Update active anomalies
        self._update_active_anomalies()
        
        if new_anomalies:
            print(f"[DEBUG] StatisticalAnomalyDetector detected {len(new_anomalies)} anomalies:", new_anomalies)
        
        return new_anomalies
    
    def _update_active_anomalies(self):
        """Update the list of active anomalies."""
        current_time = time.time()
        expired_anomalies = []
        
        for anomaly_id, anomaly in self.active_anomalies.items():
            # Remove anomalies older than 5 minutes
            if current_time - anomaly["timestamp"] > 300:
                expired_anomalies.append(anomaly_id)
        
        for anomaly_id in expired_anomalies:
            del self.active_anomalies[anomaly_id]


class GraphAnomalyDetector(AnomalyDetector):
    """
    Detects anomalies using graph-based analysis.
    """
    def __init__(self, correlation_threshold: float = 0.7, data_dir: str = "data/anomalies"):
        """
        Initialize the graph anomaly detector.
        
        Args:
            correlation_threshold: Threshold for correlation-based detection
            data_dir: Directory to store anomaly data
        """
        super().__init__(data_dir)
        self.correlation_threshold = correlation_threshold
        self.metrics_history: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.prev_communities = None
    
    def detect_anomalies(self, service_statuses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies using graph-based analysis.
        
        Args:
            service_statuses: Current status of all services
        
        Returns:
            List of detected anomalies
        """
        current_time = time.time()
        new_anomalies = []
        
        # Update metrics history
        for service_id, status in service_statuses.items():
            metrics = status.get("metrics", {})
            
            if service_id not in self.metrics_history:
                self.metrics_history[service_id] = {}
            
            for metric_name, metric_value in metrics.items():
                if not isinstance(metric_value, (int, float)):
                    continue
                
                if metric_name not in self.metrics_history[service_id]:
                    self.metrics_history[service_id][metric_name] = []
                
                self.metrics_history[service_id][metric_name].append({
                    "timestamp": current_time,
                    "value": metric_value
                })
        
                # Keep only recent history (last 100 points)
                if len(self.metrics_history[service_id][metric_name]) > 100:
                    self.metrics_history[service_id][metric_name] = self.metrics_history[service_id][metric_name][-100:]
        
        # Build the service dependency graph
        G = nx.Graph()
        for service_id, status in service_statuses.items():
            G.add_node(service_id)
            for dep_id in status.get("dependencies", []):
                if dep_id in service_statuses:
                    G.add_edge(service_id, dep_id)
        
        # Community detection using Louvain
        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G)
            if self.prev_communities is not None:
                # Compare with previous partition
                changed = sum(1 for n in partition if self.prev_communities.get(n) != partition[n])
                if changed > 0:
                anomaly = {
                        "id": f"anomaly_community_{int(current_time)}",
                        "type": "community",
                        "description": f"Community structure changed for {changed} nodes",
                        "changed_nodes": [n for n in partition if self.prev_communities.get(n) != partition[n]],
                        "timestamp": current_time,
                        "severity": "medium" if changed < len(partition) // 2 else "high"
                }
                    new_anomalies.append(anomaly)
                    self.anomalies.append(anomaly)
                    self.active_anomalies[anomaly["id"]] = anomaly
            self.prev_communities = partition
        
        # Check for anomalies in service dependencies
        for service_id, status in service_statuses.items():
            dependencies = status.get("dependencies", [])
            
            for dep_id in dependencies:
                if dep_id not in service_statuses:
                    continue
                
                # Check for correlation between service metrics
                service_metrics = self.metrics_history.get(service_id, {})
                dep_metrics = self.metrics_history.get(dep_id, {})
                
                for s_metric, s_values in service_metrics.items():
                    for d_metric, d_values in dep_metrics.items():
                        if len(s_values) < 2 or len(d_values) < 2:
                            continue
                        
                        # Calculate correlation
                        s_ts = [v["value"] for v in s_values]
                        d_ts = [v["value"] for v in d_values]
                        
                        # Ensure same length
                        min_length = min(len(s_ts), len(d_ts))
                        s_ts = s_ts[-min_length:]
                        d_ts = d_ts[-min_length:]
                        
                        correlation = np.corrcoef(s_ts, d_ts)[0, 1]
                        
                        if abs(correlation) > self.correlation_threshold:
                            # Check for sudden changes in correlation
                            if len(s_ts) >= 3 and len(d_ts) >= 3:
                                recent_corr = np.corrcoef(s_ts[-3:], d_ts[-3:])[0, 1]
        
                                if abs(correlation - recent_corr) > 0.5:  # Significant change
                    anomaly = {
                                        "id": f"anomaly_{service_id}_{dep_id}_{int(current_time)}",
                                        "type": "graph",
                                        "service_id": service_id,
                                        "dependency_id": dep_id,
                                        "metric_pair": f"{s_metric}-{d_metric}",
                                        "correlation": correlation,
                                        "recent_correlation": recent_corr,
                                        "timestamp": current_time,
                                        "severity": "high" if abs(correlation - recent_corr) > 0.8 else "medium"
                    }
                    
                                    new_anomalies.append(anomaly)
                                    self.anomalies.append(anomaly)
                                    self.active_anomalies[anomaly["id"]] = anomaly
                    
                                    logger.info(f"Detected graph anomaly between {service_id} and {dep_id}: "
                                              f"correlation change = {abs(correlation - recent_corr):.2f}")
        
        # Update active anomalies
        self._update_active_anomalies()
        
        return new_anomalies
    
    def _update_active_anomalies(self):
        """Update the list of active anomalies."""
        current_time = time.time()
        expired_anomalies = []
        
        for anomaly_id, anomaly in self.active_anomalies.items():
            # Remove anomalies older than 5 minutes
            if current_time - anomaly["timestamp"] > 300:
                expired_anomalies.append(anomaly_id)
        
        for anomaly_id in expired_anomalies:
            del self.active_anomalies[anomaly_id]


class BayesianFaultLocalizer:
    """
    Uses Bayesian reasoning to estimate the probability of each service being the root cause given observed anomalies.
    """
    def __init__(self, prior: Optional[Dict[str, float]] = None):
        self.prior = prior  # Prior probabilities for each service (can be uniform or based on history)

    def localize(self, anomalies: List[Dict[str, Any]], service_statuses: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        # If no prior, use uniform
        services = list(service_statuses.keys())
        if not self.prior:
            prior = {s: 1.0 / len(services) for s in services}
        else:
            prior = self.prior.copy()
        # Likelihood: for each service, product of likelihoods from anomalies
        likelihood = {s: 1.0 for s in services}
        for anomaly in anomalies:
            affected = anomaly.get("service_id") or anomaly.get("changed_nodes", [])
            if isinstance(affected, str):
                affected = [affected]
            for s in services:
                # Simple likelihood: if service is affected, higher likelihood
                if s in affected:
                    sev = anomaly.get("severity", "medium")
                    if sev == "high":
                        likelihood[s] *= 0.9
                    elif sev == "medium":
                        likelihood[s] *= 0.7
                    else:
                        likelihood[s] *= 0.5
                else:
                    likelihood[s] *= 0.2  # Less likely if not directly affected
        # Compute unnormalized posterior
        posterior = {s: prior[s] * likelihood[s] for s in services}
        # Normalize
        total = sum(posterior.values())
        if total > 0:
            posterior = {s: posterior[s] / total for s in services}
        else:
            posterior = {s: 1.0 / len(services) for s in services}
        # Return sorted list
        return sorted(posterior.items(), key=lambda x: x[1], reverse=True)


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
        self.bayesian_localizer = BayesianFaultLocalizer()
    
    def detect_anomalies(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies using all detectors.
        
        Args:
            system_state: Current state of the system
        
        Returns:
            List of detected anomalies
        """
        print("[DEBUG] AnomalyManager.detect_anomalies CALLED")
        print("[DEBUG] system_state:", system_state)
        new_anomalies = []
        
        # Extract relevant data from system state
        graph = system_state.get("graph")
        service_metrics = system_state.get("service_metrics", {})
        service_statuses = system_state.get("service_statuses", {})
        
        # Run each detector
        for detector in self.detectors:
            try:
                print(f"[DEBUG] Running detector: {detector.__class__.__name__}")
                if isinstance(detector, StatisticalAnomalyDetector):
                    anomalies = detector.detect_anomalies(service_statuses)
                elif isinstance(detector, GraphAnomalyDetector):
                    anomalies = detector.detect_anomalies(service_statuses)
                else:
                    anomalies = detector.detect_anomalies(system_state)
                print(f"[DEBUG] {detector.__class__.__name__} returned anomalies:", anomalies)
                new_anomalies.extend(anomalies)
            except Exception as e:
                logger.error(f"Error in anomaly detector {detector.__class__.__name__}: {e}")
        
        # After running all detectors, run Bayesian localization
        if service_statuses:
            bayes_ranking = self.bayesian_localizer.localize(new_anomalies, service_statuses)
            print("[DEBUG] BayesianFaultLocalizer ranking:", bayes_ranking)
            # Optionally, attach to anomalies or return as part of result
            if new_anomalies:
                for anomaly in new_anomalies:
                    anomaly["bayesian_ranking"] = bayes_ranking
        
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

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join()
            logger.info("Stopped service monitoring")