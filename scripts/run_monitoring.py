import argparse
import time
import logging
from flask import Flask, Response
from graph_heal.graph_model import create_sample_graph
from graph_heal.monitoring import ServiceMonitor, GraphUpdater, REGISTRY
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import threading
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('monitoring')

app = Flask(__name__)
service_monitor = None
graph_updater = None

def init_monitoring():
    """Initialize the service monitor and graph updater."""
    global service_monitor, graph_updater
    
    # Create sample graph
    graph = create_sample_graph()
    
    # Define service configurations
    services = [
        {
            "id": "user_service",
            "name": "User Service",
            "url": "http://service_a:5000",
            "health_endpoint": "/health",
            "metrics_endpoint": "/metrics"
        },
        {
            "id": "order_service",
            "name": "Order Service",
            "url": "http://service_b:5000",
            "health_endpoint": "/health",
            "metrics_endpoint": "/metrics"
        },
        {
            "id": "inventory_service",
            "name": "Inventory Service",
            "url": "http://service_c:5000",
            "health_endpoint": "/health",
            "metrics_endpoint": "/metrics"
        },
        {
            "id": "notification_service",
            "name": "Notification Service",
            "url": "http://service_d:5000",
            "health_endpoint": "/health",
            "metrics_endpoint": "/metrics"
        }
    ]
    
    # Initialize monitoring
    service_monitor = ServiceMonitor(services, poll_interval=5)
    graph_updater = GraphUpdater(graph, service_monitor)
    
    # Start monitoring and updating
    service_monitor.start_monitoring()
    graph_updater.start_updating()
        
    logger.info("Service monitor initialized and started")

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics for all services."""
    global service_monitor
    
    if not service_monitor:
        init_monitoring()
    
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)

def run_monitoring():
    """Run the monitoring loop in a separate thread."""
    try:
        while True:
            # Print system status every 30 seconds
            if service_monitor:
                status = service_monitor.get_all_services_status()
                logger.info("System Status:")
                for service_id, service_status in status.items():
                    logger.info(f"  {service_status['name']}: {service_status['health']} "
                              f"(Availability: {service_status['availability']:.1f}%)")
            time.sleep(30)
    except Exception as e:
        logger.error(f"Error in monitoring loop: {e}")

if __name__ == "__main__":
    # Initialize monitoring
    init_monitoring()
    
    # Start monitoring in a separate thread
    monitoring_thread = threading.Thread(target=run_monitoring)
    monitoring_thread.daemon = True
    monitoring_thread.start()
    
    # Run Flask application with development server
    logger.info("Starting Flask application on 0.0.0.0:5000")
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 5000, app, use_reloader=False, use_debugger=True)