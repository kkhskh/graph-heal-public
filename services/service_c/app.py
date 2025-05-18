# Compatibility layer for werkzeug
try:
    from werkzeug.urls import url_quote
except ImportError:
    from werkzeug.urls import quote as url_quote

from flask import Flask, jsonify, request
import time
import logging
import random
import psutil
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('service_c')

# Simulated inventory storage
inventory = []

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "inventory-service"})

# Get all inventory
@app.route('/inventory', methods=['GET'])
def get_inventory():
    logger.info("Retrieving all inventory")
    return jsonify(inventory)

# Update inventory
@app.route('/inventory', methods=['POST'])
def update_inventory():
    data = request.json
    item = {
        "id": len(inventory) + 1,
        "name": data.get("name"),
        "quantity": data.get("quantity", 0),
        "timestamp": time.time()
    }
    inventory.append(item)
    
    logger.info(f"Updated inventory for item {item['id']}: {data.get('name')}")
    
    # Simulate occasional latency
    if random.random() < 0.1:
        time.sleep(0.3)
    
    return jsonify(item), 201

# Metrics endpoint
@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)