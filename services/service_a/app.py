# Compatibility layer for werkzeug
try:
    from werkzeug.urls import url_quote
except ImportError:
    from werkzeug.urls import quote as url_quote

from flask import Flask, jsonify, request
import time
import os
import logging
import random
import requests
import psutil
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('service_a')

# Simulated users storage
users = []

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "user-service"})

# Get all users
@app.route('/users', methods=['GET'])
def get_users():
    logger.info("Retrieving all users")
    return jsonify(users)

# Create user
@app.route('/users', methods=['POST'])
def create_user():
    data = request.json
    user = {
        "id": len(users) + 1,
        "username": data.get("username"),
        "email": data.get("email"),
        "timestamp": time.time()
    }
    users.append(user)
    
    logger.info(f"Created user {user['id']}: {data.get('username')}")
    
    # Simulate occasional latency
    if random.random() < 0.1:
        time.sleep(0.3)
    
    return jsonify(user), 201

# Metrics endpoint
@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)