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
logger = logging.getLogger('service_b')

# Simulated orders storage
orders = []

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "order-service"})

# Get all orders
@app.route('/orders', methods=['GET'])
def get_orders():
    logger.info("Retrieving all orders")
    return jsonify(orders)

# Create order
@app.route('/orders', methods=['POST'])
def create_order():
    data = request.json
    order = {
        "id": len(orders) + 1,
        "user_id": data.get("user_id"),
        "items": data.get("items", []),
        "total": data.get("total", 0),
        "timestamp": time.time()
    }
    orders.append(order)
    
    logger.info(f"Created order {order['id']} for user {data.get('user_id')}")
    
    # Simulate occasional latency
    if random.random() < 0.1:
        time.sleep(0.3)
    
    return jsonify(order), 201

# Metrics endpoint
@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    