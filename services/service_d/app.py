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
logger = logging.getLogger('service_d')

# Simulated notifications storage
notifications = []

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "notification-service"})

# Get all notifications
@app.route('/notifications', methods=['GET'])
def get_notifications():
    logger.info("Retrieving all notifications")
    return jsonify(notifications)

# Create notification
@app.route('/notifications', methods=['POST'])
def create_notification():
    data = request.json
    notification = {
        "id": len(notifications) + 1,
        "event": data.get("event", "unknown"),
        "user_id": data.get("user_id"),
        "order_id": data.get("order_id"),
        "timestamp": time.time()
    }
    notifications.append(notification)
    
    logger.info(f"Created notification for event: {data.get('event')}")
    
    # Simulate sending notification (just logging)
    logger.info(f"Sending notification to user {data.get('user_id')} about {data.get('event')}")
    
    # Simulate occasional latency
    if random.random() < 0.1:
        time.sleep(0.3)
    
    return jsonify(notification), 201

# Metrics endpoint
@app.route('/metrics')
def metrics():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem_info = psutil.virtual_memory()
    mem_mb = mem_info.used / 1024 / 1024
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)