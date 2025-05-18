from prometheus_client import Counter, Gauge, Histogram, generate_latest
from typing import Dict, Any

# Define metrics
REQUEST_COUNT = Counter(
    'service_request_total',
    'Total number of requests',
    ['service', 'endpoint', 'method', 'status']
)

REQUEST_LATENCY = Histogram(
    'service_request_duration_seconds',
    'Request latency in seconds',
    ['service', 'endpoint']
)

SERVICE_HEALTH = Gauge(
    'service_health',
    'Service health status (1 for healthy, 0 for unhealthy)',
    ['service']
)

SERVICE_AVAILABILITY = Gauge(
    'service_availability_percentage',
    'Service availability percentage',
    ['service']
)

def format_metrics(metrics_data: Dict[str, Any], service_name: str) -> bytes:
    """
    Format metrics data in Prometheus format.
    
    Args:
        metrics_data: Dictionary containing metrics data
        service_name: Name of the service
    
    Returns:
        Prometheus-formatted metrics as bytes
    """
    # Update metrics with the latest data
    if 'requests' in metrics_data:
        for endpoint, data in metrics_data['requests'].items():
            REQUEST_COUNT.labels(
                service=service_name,
                endpoint=endpoint,
                method=data.get('method', 'GET'),
                status=data.get('status', '200')
            ).inc(data.get('count', 0))
            
            if 'latency' in data:
                REQUEST_LATENCY.labels(
                    service=service_name,
                    endpoint=endpoint
                ).observe(data['latency'])
    
    if 'health' in metrics_data:
        SERVICE_HEALTH.labels(service=service_name).set(
            1 if metrics_data['health'].get('status') == 'healthy' else 0
        )
    
    if 'availability' in metrics_data:
        SERVICE_AVAILABILITY.labels(service=service_name).set(
            metrics_data['availability']
        )
    
    return generate_latest() 