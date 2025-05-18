import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
import os
import datetime
import requests
from typing import Dict, List, Optional, Tuple, Any

class Node:
    """
    Represents a service node in the system graph.
    """
    def __init__(self, 
                 id: str, 
                 name: str, 
                 service_type: str,
                 url: str, 
                 health_endpoint: str = "/health",
                 metrics_endpoint: str = "/metrics"):
        self.id = id
        self.name = name
        self.service_type = service_type
        self.url = url
        self.health_endpoint = health_endpoint
        self.metrics_endpoint = metrics_endpoint
        self.status = "unknown"  # unknown, healthy, unhealthy
        self.metrics = {}
        self.last_updated = datetime.datetime.now()
    
    def check_health(self) -> str:
        """Check the health of the node by calling its health endpoint."""
        try:
            response = requests.get(f"{self.url}{self.health_endpoint}", timeout=1)
            if response.status_code == 200:
                self.status = "healthy"
            else:
                self.status = "unhealthy"
        except requests.RequestException:
            self.status = "unhealthy"
        
        self.last_updated = datetime.datetime.now()
        return self.status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from the node's metrics endpoint."""
        try:
            response = requests.get(f"{self.url}{self.metrics_endpoint}", timeout=1)
            if response.status_code == 200:
                self.metrics = response.json()
        except requests.RequestException:
            pass  # Keep existing metrics
        
        return self.metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "service_type": self.service_type,
            "url": self.url,
            "status": self.status,
            "metrics": self.metrics,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create node from dictionary."""
        node = cls(
            id=data["id"],
            name=data["name"],
            service_type=data["service_type"],
            url=data["url"]
        )
        node.status = data.get("status", "unknown")
        node.metrics = data.get("metrics", {})
        node.last_updated = datetime.datetime.fromisoformat(
            data.get("last_updated", datetime.datetime.now().isoformat())
        )
        return node

class Edge:
    """
    Represents a dependency edge between services in the system graph.
    """
    def __init__(self, 
                 source_id: str, 
                 target_id: str, 
                 edge_type: str = "api_call",
                 weight: float = 1.0):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.weight = weight
        self.metrics = {
            "call_count": 0,
            "error_count": 0,
            "avg_latency": 0
        }
        self.last_updated = datetime.datetime.now()
    
    def update_metrics(self, call_count: int, error_count: int, avg_latency: float):
        """Update the edge metrics."""
        self.metrics["call_count"] = call_count
        self.metrics["error_count"] = error_count
        self.metrics["avg_latency"] = avg_latency
        self.last_updated = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "metrics": self.metrics,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Edge':
        """Create edge from dictionary."""
        edge = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=data.get("edge_type", "api_call"),
            weight=data.get("weight", 1.0)
        )
        edge.metrics = data.get("metrics", {
            "call_count": 0,
            "error_count": 0,
            "avg_latency": 0
        })
        edge.last_updated = datetime.datetime.fromisoformat(
            data.get("last_updated", datetime.datetime.now().isoformat())
        )
        return edge

class SystemGraph:
    """
    Represents the overall system as a graph of nodes (services) and edges (dependencies).
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
    
    def add_node(self, node: Node):
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self.graph.add_node(
            node.id, 
            name=node.name,
            service_type=node.service_type,
            status=node.status
        )
    
    def add_edge(self, edge: Edge):
        """Add an edge to the graph."""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            type=edge.edge_type,
            weight=edge.weight
        )
    
    def update_node_status(self, node_id: str, status: str):
        """Update a node's status."""
        if node_id in self.nodes:
            self.nodes[node_id].status = status
            self.graph.nodes[node_id]["status"] = status
    
    def update_all_nodes_health(self):
        """Check and update the health status of all nodes."""
        for node_id, node in self.nodes.items():
            status = node.check_health()
            self.graph.nodes[node_id]["status"] = status
    
    def update_all_nodes_metrics(self):
        """Get and update metrics for all nodes."""
        for node_id, node in self.nodes.items():
            node.get_metrics()
    
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_edges_for_node(self, node_id: str) -> List[Edge]:
        """Get all edges connected to a node."""
        return [edge for edge in self.edges if edge.source_id == node_id or edge.target_id == node_id]
    
    def save_to_file(self, filename: str):
        """Save the graph to a JSON file."""
        data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'SystemGraph':
        """Load a graph from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        graph = cls()
        for node_data in data["nodes"]:
            node = Node.from_dict(node_data)
            graph.add_node(node)
        
        for edge_data in data["edges"]:
            edge = Edge.from_dict(edge_data)
            graph.add_edge(edge)
        
        return graph
    
    def visualize(self, output_file: str = None):
        """Visualize the graph and optionally save it to a file."""
        plt.figure(figsize=(12, 8))
        
        # Define node colors based on status
        node_colors = []
        for node_id in self.graph.nodes():
            status = self.graph.nodes[node_id]["status"]
            if status == "healthy":
                node_colors.append("green")
            elif status == "unhealthy":
                node_colors.append("red")
            else:
                node_colors.append("gray")
        
        # Create layout
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, 
            pos, 
            node_color=node_colors,
            node_size=700
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, 
            pos, 
            arrows=True,
            arrowsize=20,
            width=1.5
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, 
            pos,
            font_size=10,
            font_family="sans-serif"
        )
        
        plt.title("System Service Graph")
        plt.axis("off")
        
        if output_file:
            plt.savefig(output_file, format="png", dpi=300, bbox_inches="tight")
        
        plt.close()  # Close the figure to free memory

# Function to create a sample system graph
def create_sample_graph() -> SystemGraph:
    """Create a sample system graph using our microservices."""
    graph = SystemGraph()
    
    # Create nodes
    user_service = Node(
        id="service_a",
        name="User Service",
        service_type="user_management",
        url="http://localhost:5001"
    )
    
    order_service = Node(
        id="service_b",
        name="Order Service",
        service_type="order_management",
        url="http://localhost:5002"
    )
    
    inventory_service = Node(
        id="service_c",
        name="Inventory Service",
        service_type="inventory_management",
        url="http://localhost:5003"
    )
    
    notification_service = Node(
        id="service_d",
        name="Notification Service",
        service_type="notification",
        url="http://localhost:5004"
    )
    
    # Add nodes to graph
    graph.add_node(user_service)
    graph.add_node(order_service)
    graph.add_node(inventory_service)
    graph.add_node(notification_service)
    
    # Create edges
    graph.add_edge(Edge(
        source_id="service_b",
        target_id="service_a",
        edge_type="api_call",
        weight=1.0
    ))
    
    graph.add_edge(Edge(
        source_id="service_b",
        target_id="service_c",
        edge_type="api_call",
        weight=1.0
    ))
    
    graph.add_edge(Edge(
        source_id="service_a",
        target_id="service_d",
        edge_type="api_call",
        weight=1.0
    ))
    
    graph.add_edge(Edge(
        source_id="service_b",
        target_id="service_d",
        edge_type="api_call",
        weight=1.0
    ))
    
    return graph

if __name__ == "__main__":
    # Create a sample graph
    sample_graph = create_sample_graph()
    
    # Update nodes health
    sample_graph.update_all_nodes_health()
    
    # Visualize the graph
    sample_graph.visualize()
    
    # Save the graph to a file
    os.makedirs("data/graphs", exist_ok=True)
    sample_graph.save_to_file("data/graphs/initial_graph.json")
    
    # Save visualization to a file
    sample_graph.visualize("data/graphs/initial_graph.png")