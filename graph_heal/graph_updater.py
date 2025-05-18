import os
import time
import json
import logging
import networkx as nx
from typing import Dict, Any, List, Optional
from threading import Thread, Event

logger = logging.getLogger(__name__)

class GraphUpdater:
    """Maintains and updates the service dependency graph."""
    
    def __init__(self, update_interval: float = 5.0, initial_graph: Dict[str, Any] = None):
        """
        Initialize the graph updater.
        
        Args:
            update_interval: Time between graph updates in seconds
            initial_graph: Initial graph structure to use
        """
        self.update_interval = update_interval
        self.graph = nx.DiGraph()
        self.stop_event = Event()
        self.update_thread = None
        
        # Initialize graph structure
        if initial_graph:
            self._initialize_graph_from_dict(initial_graph)
        else:
            self._initialize_graph()
        
        # Create data directory if it doesn't exist
        self.data_dir = "data/graphs"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _initialize_graph(self) -> None:
        """Initialize the base graph structure."""
        # Add service nodes
        services = ["service_a", "service_b", "service_c", "service_d"]
        for service in services:
            self.graph.add_node(service, type="service")
        
        # Add dependencies
        dependencies = [
            ("service_a", "service_b"),
            ("service_a", "service_c"),
            ("service_b", "service_d"),
            ("service_c", "service_d")
        ]
        
        for source, target in dependencies:
            self.graph.add_edge(source, target, weight=1.0)
    
    def _initialize_graph_from_dict(self, graph_dict: Dict[str, Any]) -> None:
        """Initialize the graph from a dictionary."""
        # Add nodes
        for node in graph_dict.get("nodes", []):
            self.graph.add_node(node["id"], **node.get("attributes", {}))
        
        # Add edges
        for edge in graph_dict.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                **edge.get("attributes", {})
            )
    
    def start_updating(self) -> None:
        """Start the graph update thread."""
        if self.update_thread and self.update_thread.is_alive():
            logger.warning("Graph update thread already running")
            return
        
        self.stop_event.clear()
        self.update_thread = Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        logger.info("Started graph updating")
    
    def stop_updating(self) -> None:
        """Stop the graph update thread."""
        if not self.update_thread:
            return
        
        self.stop_event.set()
        self.update_thread.join()
        self.update_thread = None
        logger.info("Stopped graph updating")
    
    def get_current_graph(self) -> nx.DiGraph:
        """
        Get the current state of the graph.
        
        Returns:
            Current graph state
        """
        return self.graph.copy()
    
    def get_service_dependencies(self, service_id: str) -> List[str]:
        """
        Get the dependencies of a service.
        
        Args:
            service_id: ID of the service
            
        Returns:
            List of service IDs that this service depends on
        """
        if service_id not in self.graph:
            raise ValueError(f"Unknown service: {service_id}")
        
        return list(self.graph.successors(service_id))
    
    def get_dependent_services(self, service_id: str) -> List[str]:
        """
        Get the services that depend on this service.
        
        Args:
            service_id: ID of the service
            
        Returns:
            List of service IDs that depend on this service
        """
        if service_id not in self.graph:
            raise ValueError(f"Unknown service: {service_id}")
        
        return list(self.graph.predecessors(service_id))
    
    def update_edge_weight(self, source: str, target: str, weight: float) -> None:
        """
        Update the weight of an edge in the graph.
        
        Args:
            source: Source service ID
            target: Target service ID
            weight: New edge weight
        """
        if not self.graph.has_edge(source, target):
            raise ValueError(f"No edge between {source} and {target}")
        
        self.graph[source][target]["weight"] = weight
    
    def _update_graph_weights(self) -> None:
        """Update graph edge weights based on service metrics."""
        # This would typically use service metrics to update weights
        # For now, we'll just keep the default weights
        pass
    
    def _save_graph_snapshot(self) -> None:
        """Save the current graph state to a file."""
        timestamp = int(time.time())
        snapshot = {
            "timestamp": timestamp,
            "nodes": list(self.graph.nodes()),
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "weight": data["weight"]
                }
                for source, target, data in self.graph.edges(data=True)
            ]
        }
        
        snapshot_file = os.path.join(
            self.data_dir,
            f"graph_{timestamp}.json"
        )
        
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
    
    def _update_loop(self) -> None:
        """Main graph update loop."""
        while not self.stop_event.is_set():
            # Update graph weights
            self._update_graph_weights()
            
            # Save graph snapshot
            self._save_graph_snapshot()
            
            # Wait for next update
            time.sleep(self.update_interval)
    
    def update_graph(self, service_statuses: dict) -> None:
        """
        Update the graph based on the latest service statuses.
        (Stub implementation: expand as needed.)
        """
        pass

    def get_graph(self):
        """Return the current graph (for compatibility)."""
        return self.get_current_graph() 