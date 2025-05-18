import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Python path:", sys.path)

from graph_heal.graph_model import create_sample_graph
print("Successfully imported create_sample_graph")

def main():
    print("Starting main function...")
    # Create a sample graph
    graph = create_sample_graph()
    print("Created sample graph")
    
    # Update health status
    print("Updating health status...")
    graph.update_all_nodes_health()
    
    # Display node statuses
    print("\nNode Health Status:")
    for node_id, node in graph.nodes.items():
        print(f"{node.name} ({node_id}): {node.status}")
    
    # Visualize and save the graph
    print("\nSaving graph visualization...")
    os.makedirs("data/graphs", exist_ok=True)
    print("Created data/graphs directory")
    
    graph.visualize("data/graphs/test_graph.png")
    print("Generated graph visualization")
    
    graph.save_to_file("data/graphs/test_graph.json")
    print("Saved graph data to JSON")
    
    print(f"\nGraph visualization saved to data/graphs/test_graph.png")
    print(f"Graph data saved to data/graphs/test_graph.json")

if __name__ == "__main__":
    print("Script started")
    main()
    print("Script completed")