import unittest
from unittest.mock import patch, MagicMock
from graph_heal.service_monitor import ServiceMonitor

class TestServiceMonitor(unittest.TestCase):
    def setUp(self):
        self.services = [
            {"id": "service_a", "url": "http://localhost:5001", "health_endpoint": "/health"},
            {"id": "service_b", "url": "http://localhost:5002", "health_endpoint": "/health"}
        ]
        self.monitor = ServiceMonitor(self.services, poll_interval=0.1)

    @patch('graph_heal.service_monitor.requests.get')
    def test_check_service_health_healthy(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"cpu": 10, "memory": 100}
        mock_get.return_value = mock_resp
        result = self.monitor._check_service_health("service_a", "http://localhost:5001/health")
        self.assertEqual(result["status"], "healthy")
        self.assertIn("cpu", result["metrics"])

    @patch('graph_heal.service_monitor.requests.get')
    def test_check_service_health_unhealthy(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp
        result = self.monitor._check_service_health("service_a", "http://localhost:5001/health")
        self.assertEqual(result["status"], "unhealthy")

    @patch('graph_heal.service_monitor.requests.get', side_effect=Exception("fail"))
    def test_check_service_health_error(self, mock_get):
        result = self.monitor._check_service_health("service_a", "http://localhost:5001/health")
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result["metrics"])

    def test_get_service_status_unknown(self):
        status = self.monitor.get_service_status("service_x")
        self.assertEqual(status["status"], "unknown")

if __name__ == "__main__":
    unittest.main() 