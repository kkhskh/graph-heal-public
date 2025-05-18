import time
import logging
import threading
import os
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from collections import defaultdict, deque
import random

from graph_heal.recovery import RecoveryAction, RecoveryDecisionEngine
from graph_heal.fault_localization import FaultManager
from .service_monitor import ServiceMonitor
from .graph_updater import GraphUpdater
from .anomaly_detection import StatisticalAnomalyDetector, GraphAnomalyDetector, AnomalyManager
from .fault_localization import GraphBasedFaultLocalizer
from .recovery import RecoveryActionFactory
from graph_heal.fault_injection import FaultInjectionAPI, FaultInjector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('orchestration')

class RecoveryTask:
    """
    A task for recovering from a fault.
    """
    def __init__(self, fault_id: str, fault_report: Dict[str, Any], action: RecoveryAction):
        """
        Initialize the recovery task.
        
        Args:
            fault_id: ID of the fault
            fault_report: Fault report
            action: Recovery action
        """
        self.id = f"task_{fault_id}_{int(time.time())}"
        self.fault_id = fault_id
        self.fault_report = fault_report
        self.action = action
        self.status = "pending"  # pending, in_progress, completed, failed
        self.start_time = None
        self.end_time = None
        self.result = None
        self.verification_result = None
    
    def execute(self) -> bool:
        """
        Execute the recovery task.
        
        Returns:
            True if the action was successful, False otherwise
        """
        self.status = "in_progress"
        self.start_time = time.time()
        
        logger.info(f"Executing recovery task {self.id} for fault {self.fault_id}")
        
        success = self.action.execute()
        
        self.end_time = time.time()
        self.result = self.action.to_dict()
        
        if success:
            self.status = "completed"
        else:
            self.status = "failed"
        
        logger.info(f"Recovery task {self.id} {self.status}")
        
        return success
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "fault_id": self.fault_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time and self.start_time else None,
            "action": self.action.to_dict() if hasattr(self.action, "to_dict") else str(self.action),
            "result": self.result,
            "verification_result": self.verification_result
        }


class RecoveryOrchestrator:
    """
    Orchestrates recovery actions.
    """
    def __init__(self, decision_engine: RecoveryDecisionEngine, 
                fault_manager: FaultManager, data_dir: str = "data/orchestration"):
        """
        Initialize the recovery orchestrator.
        
        Args:
            decision_engine: RecoveryDecisionEngine instance
            fault_manager: FaultManager instance
            data_dir: Directory to store orchestration data
        """
        self.decision_engine = decision_engine
        self.fault_manager = fault_manager
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.tasks: Dict[str, RecoveryTask] = {}
        self.active_tasks: Dict[str, RecoveryTask] = {}
        self.completed_tasks: Dict[str, RecoveryTask] = {}
        self.task_history: List[Dict[str, Any]] = []
        
        self.dependency_graph = {}
        self.execution_thread = None
        self.stop_event = threading.Event()
    
    def start(self):
        """Start the orchestrator."""
        if self.execution_thread is None or not self.execution_thread.is_alive():
            self.stop_event.clear()
            self.execution_thread = threading.Thread(target=self._execution_loop)
            self.execution_thread.daemon = True
            self.execution_thread.start()
            logger.info("Recovery orchestrator started")
    
    def stop(self):
        """Stop the orchestrator."""
        if self.execution_thread is not None and self.execution_thread.is_alive():
            self.stop_event.set()
            self.execution_thread.join(timeout=5)
            logger.info("Recovery orchestrator stopped")
    
    def _execution_loop(self):
        """Main execution loop."""
        while not self.stop_event.is_set():
            try:
                # Handle active tasks
                for task_id, task in list(self.active_tasks.items()):
                    # Check for completed tasks
                    if task.status in ["completed", "failed"]:
                        # Move to completed tasks
                        self.completed_tasks[task_id] = task
                        del self.active_tasks[task_id]
                        
                        # Update task history
                        self.task_history.append(task.to_dict())
                        
                        # Log task completion
                        self._log_task(task)
                
                # Schedule new tasks
                self._schedule_tasks()
                
            except Exception as e:
                logger.error(f"Error in orchestrator execution loop: {e}")
            
            # Sleep for a short time
            time.sleep(1)
    
    def _schedule_tasks(self):
        """Schedule pending tasks for execution."""
        # Check for new faults
        active_faults = self.fault_manager.get_active_faults()
        
        for fault in active_faults:
            fault_id = fault["id"]
            # Skip faults that already have active tasks
            if any(task.fault_id == fault_id for task in self.active_tasks.values()):
                continue
            
# Skip faults that have been recently addressed
            recently_addressed = False
            for task in self.completed_tasks.values():
                if task.fault_id == fault_id and task.end_time and time.time() - task.end_time < 60:
                    # Skip if addressed in the last 60 seconds
                    recently_addressed = True
                    break
            
            if recently_addressed:
                continue
            
            # Decide on recovery actions
            logger.info(f"Deciding recovery actions for fault {fault_id}")
            actions = self.decision_engine.decide_recovery_actions([fault], {})
            
            if actions:
                # Create recovery tasks
                for action in actions:
                    task = RecoveryTask(fault_id, fault, action)
                    self.tasks[task.id] = task
                    
                    # Log the decision
                    self.decision_engine.log_recovery_decision(fault, action)
                    
                    # Determine dependencies
                    self._determine_task_dependencies(task)
                    
                    logger.info(f"Created recovery task {task.id} for fault {fault_id}")
        
        # Find tasks that are ready to execute
        for task_id, task in list(self.tasks.items()):
            if task.status == "pending" and self._can_execute_task(task):
                # Move to active tasks
                self.active_tasks[task_id] = task
                del self.tasks[task_id]
                
                # Execute the task in a separate thread
                thread = threading.Thread(target=self._execute_task, args=(task,))
                thread.daemon = True
                thread.start()
    
    def _determine_task_dependencies(self, task: RecoveryTask):
        """
        Determine dependencies for a task.
        
        Args:
            task: Recovery task
        """
        # Initialize dependencies
        self.dependency_graph[task.id] = set()
        
        # Check if this task depends on other active tasks
        for other_task_id, other_task in self.active_tasks.items():
            # If working on the same service, create a dependency
            if task.action.target_id == other_task.action.target_id:
                self.dependency_graph[task.id].add(other_task_id)
                logger.debug(f"Task {task.id} depends on {other_task_id}")
        
        # Check if this task depends on other pending tasks
        for other_task_id, other_task in self.tasks.items():
            # If working on the same service, create a dependency
            if task.action.target_id == other_task.action.target_id:
                self.dependency_graph[task.id].add(other_task_id)
                logger.debug(f"Task {task.id} depends on {other_task_id}")
    
    def _can_execute_task(self, task: RecoveryTask) -> bool:
        """
        Check if a task can be executed.
        
        Args:
            task: Recovery task
        
        Returns:
            True if the task can be executed, False otherwise
        """
        # Check dependencies
        if task.id in self.dependency_graph:
            for dependency_id in self.dependency_graph[task.id]:
                # Check if the dependency is still active
                if dependency_id in self.active_tasks:
                    return False
                
                # Check if the dependency is still pending
                if dependency_id in self.tasks:
                    return False
                
                # Check if the dependency failed
                if dependency_id in self.completed_tasks and self.completed_tasks[dependency_id].status == "failed":
                    return False
        
        return True
    
    def _execute_task(self, task: RecoveryTask):
        """
        Execute a task.
        
        Args:
            task: Recovery task
        """
        try:
            # Execute the task
            success = task.execute()
            
            # Verify the recovery
            verification_result = self._verify_recovery(task)
            task.verification_result = verification_result
            
            logger.info(f"Task {task.id} executed with result: {success}")
            logger.info(f"Verification result: {verification_result}")
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            task.status = "failed"
    
    def _verify_recovery(self, task: RecoveryTask) -> Dict[str, Any]:
        """
        Verify the recovery.
        
        Args:
            task: Recovery task
        
        Returns:
            Verification result
        """
        # In a real system, we would check if the service is healthy again
        # and if the fault has been resolved
        
        # For our demonstration, we'll just simulate a verification process
        time.sleep(2)  # Simulate verification delay
        
        # Check if the service is healthy
        service_id = task.action.target_id
        service_status = "unknown"
        
        # Get service status from fault manager
        service_statuses = self.fault_manager.monitor.get_all_services_status()
        if service_id in service_statuses:
            service_status = service_statuses[service_id].get("health", "unknown")
        
        # Simple verification result
        return {
            "timestamp": time.time(),
            "service_id": service_id,
            "service_status": service_status,
            "success": service_status == "healthy",
            "message": f"Service {service_id} is {service_status}"
        }
    
    def _log_task(self, task: RecoveryTask):
        """
        Log a task.
        
        Args:
            task: Recovery task
        """
        try:
            # Create a unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            task_id = task.id
            
            filename = f"{self.data_dir}/task_{timestamp}_{task_id}.json"
            
            with open(filename, 'w') as f:
                json.dump(task.to_dict(), f, indent=2)
            
            logger.debug(f"Logged task to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to log task: {e}")
    
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get active tasks.
        
        Returns:
            Dictionary of active tasks
        """
        return {task_id: task.to_dict() for task_id, task in self.active_tasks.items()}
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """
        Get task history.
        
        Returns:
            List of completed tasks
        """
        return self.task_history.copy()
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task
        
        Returns:
            Task dictionary if found, None otherwise
        """
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()
        
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()
        
        return None


class SystemController:
    """
    Controller for the entire system.
    """
    def __init__(self, service_monitor, graph_updater, fault_manager, recovery_orchestrator):
        """
        Initialize the system controller.
        
        Args:
            service_monitor: ServiceMonitor instance
            graph_updater: GraphUpdater instance
            fault_manager: FaultManager instance
            recovery_orchestrator: RecoveryOrchestrator instance
        """
        self.monitor = service_monitor
        self.graph_updater = graph_updater
        self.fault_manager = fault_manager
        self.orchestrator = recovery_orchestrator
        
        self.running = False
        self.control_thread = None
        self.stop_event = threading.Event()
        
        self.system_state = {}
    
    def start(self):
        """Start the system controller."""
        if self.running:
            logger.warning("System controller is already running")
            return
        
        logger.info("Starting system controller...")
        
        # Start all components
        self.monitor.start_monitoring()
        self.graph_updater.start_updating()
        self.orchestrator.start()
        
        # Start control loop
        self.stop_event.clear()
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        self.running = True
        logger.info("System controller started")
    
    def stop(self):
        """Stop the system controller."""
        if not self.running:
            logger.warning("System controller is not running")
            return
        
        logger.info("Stopping system controller...")
        
        # Stop control loop
        self.stop_event.set()
        if self.control_thread:
            self.control_thread.join(timeout=5)
        
        # Stop all components
        self.orchestrator.stop()
        self.graph_updater.stop_updating()
        self.monitor.stop_monitoring()
        
        self.running = False
        logger.info("System controller stopped")
    
    def _control_loop(self):
        """Main control loop."""
        while not self.stop_event.is_set():
            try:
                # Update system state
                self._update_system_state()
                
                # Process the system state
                self.fault_manager.process_system_state(self.system_state)
                
                # Let the orchestrator handle recovery
                # (it has its own execution thread)
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
            
            # Sleep for a short time
            time.sleep(1)
    
    def _update_system_state(self):
        """Update the system state."""
        try:
            # Get the current graph
            graph = self.graph_updater.get_current_graph()
            
            # Get service metrics
            service_metrics = {}
            for service in self.monitor.services:
                service_id = service["id"]
                service_metrics[service_id] = self.monitor.get_service_metrics(service_id)
            
            # Get service statuses
            service_statuses = self.monitor.get_all_services_status()
            
            # Update system state
            self.system_state = {
                "graph": graph,
                "service_metrics": service_metrics,
                "service_statuses": service_statuses,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error updating system state: {e}")
    
    def get_system_state(self) -> Dict[str, Any]:
        """
        Get the current system state.
        
        Returns:
            System state dictionary
        """
        return self.system_state.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current system status.
        
        Returns:
            System status dictionary
        """
        if not self.running:
            return {"status": "stopped"}
        
        try:
            # Get service statuses
            service_statuses = self.monitor.get_all_services_status()
            
            # Count healthy and unhealthy services
            healthy_count = 0
            unhealthy_count = 0
            unknown_count = 0
            
            for service_id, status in service_statuses.items():
                if status.get("health") == "healthy":
                    healthy_count += 1
                elif status.get("health") == "unhealthy":
                    unhealthy_count += 1
                else:
                    unknown_count += 1
            
            # Get active faults
            active_faults = self.fault_manager.get_active_faults()
            
            # Get active recovery tasks
            active_tasks = self.orchestrator.get_active_tasks()
            
            # Determine overall system status
            overall_status = "operational"
            if unhealthy_count > 0:
                if unhealthy_count == 1:
                    overall_status = "degraded"
                else:
                    overall_status = "impaired"
            
            if unhealthy_count >= len(service_statuses) / 2:
                overall_status = "critical"
            
            return {
                "status": overall_status,
                "timestamp": time.time(),
                "services": {
                    "total": len(service_statuses),
                    "healthy": healthy_count,
                    "unhealthy": unhealthy_count,
                    "unknown": unknown_count
                },
                "faults": {
                    "active": len(active_faults),
                    "details": active_faults
                },
                "recovery": {
                    "active_tasks": len(active_tasks),
                    "details": active_tasks
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"status": "error", "message": str(e)}
    
    def manual_intervention(self, action_type: str, target_id: str, 
                           params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform a manual intervention.
        
        Args:
            action_type: Type of action
            target_id: ID of the target service/component
            params: Additional parameters
        
        Returns:
            Result of the intervention
        """
        if not self.running:
            return {"success": False, "message": "System controller is not running"}
        
        try:
            # Create and execute the action
            action = self.orchestrator.decision_engine.action_factory.create_action(
                action_type, target_id, params
            )
            
            # Execute the action
            success = action.execute()
            
            return {
                "success": success,
                "action": action.to_dict(),
                "message": f"Manual intervention {action_type} on {target_id} {'succeeded' if success else 'failed'}"
            }
            
        except Exception as e:
            logger.error(f"Error in manual intervention: {e}")
            return {"success": False, "message": str(e)}


class SystemOrchestrator:
    """
    Orchestrates the entire system, coordinating monitoring, detection, localization, and recovery.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the system orchestrator.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.running = False
        self.monitoring_thread = None
        self.recovery_thread = None
        
        # Initialize components
        self.service_monitor = ServiceMonitor(
            services=config.get("services", []),
            metrics_interval=config.get("metrics_interval", 5)
        )
        
        self.graph_updater = GraphUpdater(
            initial_graph=config.get("initial_graph", {}),
            update_interval=config.get("graph_update_interval", 10)
        )
        
        # Initialize anomaly detectors
        self.statistical_detector = StatisticalAnomalyDetector(
            window_size=config.get("statistical_window_size", 10),
            z_score_threshold=config.get("z_score_threshold", 3.0)
        )
        
        self.graph_detector = GraphAnomalyDetector()
        
        self.anomaly_manager = AnomalyManager([
            self.statistical_detector,
            self.graph_detector
        ])
        
        # Initialize fault localizer
        self.fault_localizer = GraphBasedFaultLocalizer(
            confidence_threshold=config.get("localization_confidence", 0.7)
        )
        
        # Initialize recovery components
        self.recovery_factory = RecoveryActionFactory(
            load_balancer_url=config.get("load_balancer_url")
        )
        
        self.recovery_engine = RecoveryDecisionEngine(
            action_factory=self.recovery_factory,
            data_dir=config.get("recovery_data_dir", "data/recovery")
        )
        
        # Set service URLs for recovery
        for service in config.get("services", []):
            service_id = service["id"]
            service_url = service["url"]
            self.recovery_factory.set_service_url(service_id, service_url)
    
    def start(self):
        """Start the system orchestration."""
        if self.running:
            logger.warning("System is already running")
            return
        
        self.running = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start recovery thread
        self.recovery_thread = threading.Thread(target=self._recovery_loop)
        self.recovery_thread.daemon = True
        self.recovery_thread.start()
        
        logger.info("System orchestration started")
    
    def stop(self):
        """Stop the system orchestration."""
        if not self.running:
            logger.warning("System is not running")
            return
        
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.recovery_thread:
            self.recovery_thread.join(timeout=5)
        
        logger.info("System orchestration stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        print("Orchestration monitoring loop running")
        while self.running:
            try:
                # Get current system state
                service_statuses = self.service_monitor.get_all_services_status()
                service_metrics = self.service_monitor.get_all_services_metrics()
                print("[DEBUG] service_statuses:", service_statuses)
                print("[DEBUG] service_metrics:", service_metrics)
                # Update graph
                self.graph_updater.update_graph(service_statuses)
                current_graph = self.graph_updater.get_graph()
                # Prepare system state
                system_state = {
                    "graph": current_graph,
                    "service_statuses": service_statuses,
                    "service_metrics": service_metrics
                }
                print("Calling anomaly_manager.detect_anomalies")
                anomalies = self.anomaly_manager.detect_anomalies(system_state)
                print("[DEBUG] anomalies returned:", anomalies)
                if anomalies:
                    logger.info(f"Detected {len(anomalies)} anomalies")
                    
                    # Localize faults
                    fault_reports = self.fault_localizer.localize_faults(
                        anomalies,
                        system_state
                    )
                    
                    if fault_reports:
                        logger.info(f"Localized {len(fault_reports)} faults")
                        
                        # Decide on recovery actions
                        recovery_actions = self.recovery_engine.decide_recovery_actions(
                            fault_reports,
                            system_state
                        )
                        
                        if recovery_actions:
                            logger.info(f"Decided on {len(recovery_actions)} recovery actions")
                            
                            # Execute recovery actions
                            for action in recovery_actions:
                                success = action.execute()
                                if success:
                                    logger.info(f"Successfully executed recovery action: {action.__class__.__name__}")
                                else:
                                    logger.error(f"Failed to execute recovery action: {action.__class__.__name__}")
                
                # Sleep until next iteration
                time.sleep(self.config.get("monitoring_interval", 5))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Sleep before retrying
    
    def _recovery_loop(self):
        """Recovery monitoring loop."""
        while self.running:
            try:
                # Get active anomalies
                active_anomalies = self.anomaly_manager.get_active_anomalies()
                
                if active_anomalies:
                    # Get current system state
                    service_statuses = self.service_monitor.get_all_services_status()
                    service_metrics = self.service_monitor.get_all_services_metrics()
                    current_graph = self.graph_updater.get_graph()
                    
                    system_state = {
                        "graph": current_graph,
                        "service_statuses": service_statuses,
                        "service_metrics": service_metrics
                    }
                    
                    # Localize faults
                    fault_reports = self.fault_localizer.localize_faults(
                        active_anomalies,
                        system_state
                    )
                    
                    if fault_reports:
                        # Decide on recovery actions
                        recovery_actions = self.recovery_engine.decide_recovery_actions(
                            fault_reports,
                            system_state
                        )
                        
                        if recovery_actions:
                            # Execute recovery actions
                            for action in recovery_actions:
                                success = action.execute()
                                if success:
                                    logger.info(f"Successfully executed recovery action: {action.__class__.__name__}")
                                else:
                                    logger.error(f"Failed to execute recovery action: {action.__class__.__name__}")
                
                # Sleep until next iteration
                time.sleep(self.config.get("recovery_interval", 30))
                
            except Exception as e:
                logger.error(f"Error in recovery loop: {e}")
                time.sleep(5)  # Sleep before retrying
    
    def get_system_state(self) -> Dict[str, Any]:
        """
        Get the current system state.
        
        Returns:
            Current system state
        """
        return {
            "graph": self.graph_updater.get_graph(),
            "service_statuses": self.service_monitor.get_all_services_status(),
            "service_metrics": self.service_monitor.get_all_services_metrics(),
            "active_anomalies": self.anomaly_manager.get_active_anomalies()
        }

# --- REST API for Orchestration Layer ---
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import networkx as nx
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Global controller instance (for demo; in production, use better lifecycle management)
controller = None

# Fault injection API instance
fault_injector = FaultInjector()
fault_api = FaultInjectionAPI(fault_injector)

@app.route('/status', methods=['GET'])
def api_status():
    if controller is None:
        return jsonify({"error": "SystemController not initialized"}), 500
    return jsonify(controller.get_system_status())

@app.route('/state', methods=['GET'])
def api_state():
    if controller is None:
        return jsonify({"error": "SystemController not initialized"}), 500
    return jsonify(controller.get_system_state())

@app.route('/manual_intervention', methods=['POST'])
def api_manual_intervention():
    if controller is None:
        return jsonify({"error": "SystemController not initialized"}), 500
    data = request.get_json(force=True)
    action_type = data.get('action_type')
    target_id = data.get('target_id')
    params = data.get('params', {})
    result = controller.manual_intervention(action_type, target_id, params)
    return jsonify(result)

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return send_from_directory(app.static_folder, 'dashboard.html')

@app.route('/graph', methods=['GET'])
def api_graph():
    if controller is None:
        return jsonify({"error": "SystemController not initialized"}), 500
    # Get the current graph and convert to Cytoscape.js format
    graph = controller.get_system_state().get('graph')
    if graph is None:
        return jsonify({"error": "No graph available"}), 500
    # Convert to Cytoscape.js format
    elements = []
    for node in graph.nodes(data=True):
        elements.append({
            "data": {"id": node[0], **node[1]}
        })
    for edge in graph.edges(data=True):
        elements.append({
            "data": {"source": edge[0], "target": edge[1], **edge[2]}
        })
    return jsonify({"elements": elements})

@app.route('/inject_fault', methods=['POST'])
def api_inject_fault():
    data = request.get_json(force=True)
    fault_type = data.get('fault_type')
    target = data.get('target')
    params = data.get('params', {})
    try:
        fault_id = fault_api.inject_fault(fault_type, target, params)
        if fault_id:
            return jsonify({"success": True, "fault_id": fault_id})
        else:
            return jsonify({"success": False, "message": "Fault injection failed"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/stop_fault', methods=['POST'])
def api_stop_fault():
    data = request.get_json(force=True)
    fault_id = data.get('fault_id')
    try:
        result = fault_api.stop_fault(fault_id)
        return jsonify({"success": result})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/active_faults', methods=['GET'])
def api_active_faults():
    faults = fault_api.get_active_faults()
    return jsonify({"active_faults": list(faults.values())})

if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Run Graph-Heal Orchestration REST API')
    parser.add_argument('--config', type=str, default='config/system_config.json', help='Path to system config')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()
    # Load config and create controller
    with open(args.config, 'r') as f:
        config = json.load(f)
    from graph_heal.graph_updater import GraphUpdater
    from graph_heal.monitoring import ServiceMonitor
    from graph_heal.fault_localization import GraphBasedFaultLocalizer
    from graph_heal.recovery import RecoveryActionFactory, RecoveryDecisionEngine
    from graph_heal.anomaly_detection import StatisticalAnomalyDetector, GraphAnomalyDetector, AnomalyManager
    # Compose controller as in main system
    service_monitor = ServiceMonitor(services=config.get("services", []), metrics_interval=config.get("metrics_interval", 5))
    graph_updater = GraphUpdater(initial_graph=config.get("initial_graph", {}), update_interval=config.get("graph_update_interval", 10))
    statistical_detector = StatisticalAnomalyDetector(window_size=config.get("statistical_window_size", 10), z_score_threshold=config.get("z_score_threshold", 3.0))
    graph_detector = GraphAnomalyDetector()
    anomaly_manager = AnomalyManager([statistical_detector, graph_detector])
    fault_localizer = GraphBasedFaultLocalizer(confidence_threshold=config.get("localization_confidence", 0.7))
    recovery_factory = RecoveryActionFactory(load_balancer_url=config.get("load_balancer_url"))
    recovery_engine = RecoveryDecisionEngine(action_factory=recovery_factory, data_dir=config.get("recovery_data_dir", "data/recovery"))
    from graph_heal.orchestration import SystemController
    controller = SystemController(service_monitor, graph_updater, fault_localizer, recovery_engine)
    controller.start()
    app.run(host=args.host, port=args.port)