import threading
import time
import random
import logging
import docker
import requests
from typing import Dict, List, Any, Optional, Callable
import json
import os
import uuid
import subprocess
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fault_injection')

class FaultInjector:
    """Injects faults into services for testing."""
    
    def __init__(self):
        """Initialize the fault injector."""
        self.active_faults = {}
        self.is_macos = platform.system() == "Darwin"
        
        # Create data directory if it doesn't exist
        self.data_dir = "data/faults"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create pfctl rules directory on macOS
        if self.is_macos:
            self.pfctl_dir = "data/pfctl"
            os.makedirs(self.pfctl_dir, exist_ok=True)
    
    def inject_fault(self, fault_type: str, target: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Inject a fault into a service.
        
        Args:
            fault_type: Type of fault to inject (latency, crash, cpu_stress, memory_leak)
            target: Target service ID
            params: Additional parameters for the fault
        
        Returns:
            ID of the injected fault
        """
        fault_id = str(uuid.uuid4())
        params = params or {}
        
        # Validate fault type
        if fault_type not in ["latency", "crash", "cpu_stress", "memory_leak"]:
            raise ValueError(f"Invalid fault type: {fault_type}")
        
        # Validate target service
        if not target.startswith("service_"):
            raise ValueError(f"Invalid target service: {target}")
        
        # Get service port
        service_ports = {
            "service_a": 8001,
            "service_b": 8002,
            "service_c": 8003,
            "service_d": 8004
        }
        
        port = service_ports.get(target)
        if not port:
            raise ValueError(f"Unknown service: {target}")
            
        # Inject the fault
        try:
            if fault_type == "latency":
                duration = params.get("duration", 30)
                latency = params.get("latency", 1000)  # 1 second default
                self._inject_latency(port, latency, duration)
            
            elif fault_type == "crash":
                duration = params.get("duration", 30)
                self._inject_crash(port, duration)
            
            elif fault_type == "cpu_stress":
                duration = params.get("duration", 30)
                cpu_load = params.get("cpu_load", 80)  # 80% CPU load default
                self._inject_cpu_stress(port, cpu_load, duration)
            
            elif fault_type == "memory_leak":
                duration = params.get("duration", 30)
                mem_mb = params.get("memory_mb", 100)
                self._inject_memory_leak(port, mem_mb, duration)
            
            # Record the fault
            fault = {
                "id": fault_id,
                "type": fault_type,
                "target": target,
                "params": params,
                "timestamp": time.time(),
                "status": "active"
            }
            
            self.active_faults[fault_id] = fault
            self._save_fault(fault)
            
            logger.info(f"Injected {fault_type} fault into {target}")
            return fault_id
            
        except Exception as e:
            logger.error(f"Failed to inject {fault_type} fault into {target}: {e}")
            return None
    
    def remove_fault(self, fault_id: str) -> bool:
        """
        Remove an active fault.
        
        Args:
            fault_id: ID of the fault to remove
        
        Returns:
            True if fault was removed, False otherwise
        """
        if fault_id not in self.active_faults:
            return False
        
        fault = self.active_faults[fault_id]
        target = fault["target"]
        port = {
            "service_a": 8001,
            "service_b": 8002,
            "service_c": 8003,
            "service_d": 8004
        }[target]
        
        try:
            if fault["type"] == "latency":
                self._remove_latency(port)
            elif fault["type"] == "crash":
                self._remove_crash(port)
            elif fault["type"] == "cpu_stress":
                self._remove_cpu_stress(port)
            elif fault["type"] == "memory_leak":
                self._remove_memory_leak(port)
            
            del self.active_faults[fault_id]
            logger.info(f"Removed {fault['type']} fault from {target}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove fault {fault_id}: {e}")
            return False
    
    def get_active_faults(self) -> List[Dict[str, Any]]:
        """
        Get list of active faults.
        
        Returns:
            List of active fault dictionaries
        """
        return list(self.active_faults.values())
    
    def _save_fault(self, fault: Dict[str, Any]) -> None:
        """Save fault information to a file."""
        fault_file = os.path.join(
            self.data_dir,
            f"fault_{fault['id']}.json"
        )
        
        with open(fault_file, 'w') as f:
            json.dump(fault, f, indent=2)
    
    def _inject_latency(self, port: int, latency: int, duration: int) -> None:
        """
        Inject latency into a service.
        
        Args:
            port: Service port
            latency: Latency in milliseconds
            duration: Duration in seconds
        """
        if self.is_macos:
            try:
                # Use stress-ng to simulate CPU load which will indirectly cause latency
                cmd = [
                    "stress-ng",
                    "--cpu", "1",
                    "--cpu-load", "80",
                    "--timeout", str(duration)
                ]
                subprocess.Popen(cmd)
                logger.info(f"Injected latency using stress-ng for port {port}")
            except Exception as e:
                logger.error(f"Failed to inject latency using stress-ng: {e}")
                raise
        else:
            # Use tc on Linux
            cmd = [
                "tc", "qdisc", "add", "dev", "lo", "root", "handle", "1:",
                "prio", "priomap", "0", "0", "0", "0", "0", "0", "0", "0",
                "0", "0", "0", "0", "0", "0", "0", "0"
            ]
            subprocess.run(cmd, check=True)
            cmd = [
                "tc", "qdisc", "add", "dev", "lo", "parent", "1:1",
                "handle", "10:", "netem", "delay", f"{latency}ms"
            ]
            subprocess.run(cmd, check=True)
            # Schedule fault removal
            if duration > 0:
                cmd = f"sleep {duration} && tc qdisc del dev lo root"
                subprocess.Popen(cmd, shell=True)
    
    def _remove_latency(self, port: int) -> None:
        """
        Remove latency from a service.
        
        Args:
            port: Service port
        """
        if self.is_macos:
            # Kill stress-ng processes
            subprocess.run(["pkill", "stress-ng"])
        else:
            # Remove tc rules on Linux
            subprocess.run(["tc", "qdisc", "del", "dev", "lo", "root"], check=True)
    
    def _inject_crash(self, port: int, duration: int) -> None:
        """
        Crash a service.
        
        Args:
            port: Service port
            duration: Duration in seconds
        """
        cmd = f"kill -9 $(lsof -t -i:{port})"
        subprocess.run(cmd, shell=True)
            
        # Schedule service restart
        if duration > 0:
            cmd = f"sleep {duration} && systemctl restart service_{port}"
            subprocess.Popen(cmd, shell=True)
    
    def _remove_crash(self, port: int) -> None:
        """
        Restart a crashed service.
        
        Args:
            port: Service port
        """
        subprocess.run(f"systemctl restart service_{port}", shell=True)
    
    def _inject_cpu_stress(self, port: int, cpu_load: int, duration: int) -> None:
        """
        Inject CPU stress into a service.
        
        Args:
            port: Service port
            cpu_load: CPU load percentage
            duration: Duration in seconds
        """
        try:
            # Use stress-ng to simulate CPU load
            cmd = [
                "stress-ng",
                "--cpu", "1",
                "--cpu-load", str(cpu_load),
                "--timeout", str(duration)
            ]
            subprocess.Popen(cmd)
            logger.info(f"Injected CPU stress on port {port} with load {cpu_load}% for {duration}s")
        except Exception as e:
            logger.error(f"Failed to inject CPU stress: {e}")
            raise
    
    def _remove_cpu_stress(self, port: int) -> None:
        """
        Remove CPU stress from a service.
        
        Args:
            port: Service port
        """
        subprocess.run(["pkill", "stress-ng"])

    def _inject_memory_leak(self, port: int, mem_mb: int, duration: int) -> None:
        """
        Inject a memory leak into a service.
        """
        # Use a Python subprocess to allocate memory in the target container
        # For demo, just run a stress-ng command if available
        try:
            cmd = [
                "stress-ng", "--vm", "1", "--vm-bytes", f"{mem_mb}M", "--timeout", str(duration)
            ]
            subprocess.Popen(cmd)
            logger.info(f"Injected memory leak ({mem_mb}MB) using stress-ng for port {port}")
        except Exception as e:
            logger.error(f"Failed to inject memory leak using stress-ng: {e}")
            raise
    
    def _remove_memory_leak(self, port: int) -> None:
        """
        Remove memory leak from a service (best effort).
        """
        # In practice, memory leaks are hard to remove; for demo, kill stress-ng
        try:
            subprocess.run(["pkill", "-f", "stress-ng"], check=False)
            logger.info(f"Removed memory leak (stress-ng) for port {port}")
        except Exception as e:
            logger.error(f"Failed to remove memory leak: {e}")
    
    def get_fault_info(self, fault_id: str):
        """
        Get information about a specific fault.
        Args:
            fault_id: The ID of the fault.
        Returns:
            The fault dictionary if found, else None.
        """
        return self.active_faults.get(fault_id)

class FaultInjectionAPI:
    """API for fault injection operations."""
    
    def __init__(self, fault_injector: FaultInjector):
        """
        Initialize the API.
        
        Args:
            fault_injector: FaultInjector instance
        """
        self.injector = fault_injector
    
    def inject_fault(self, fault_type: str, target: str, params: Dict[str, Any] = None) -> Optional[str]:
        """
        Inject a fault.
        
        Args:
            fault_type: Type of fault to inject (cpu_stress, memory_leak, latency, crash)
            target: Target container or service URL
            params: Additional parameters for the fault
        
        Returns:
            Fault ID if successful, None otherwise
        """
        if fault_type not in ["latency", "crash", "cpu_stress", "memory_leak"]:
            raise ValueError(f"Invalid fault type: {fault_type}")
        try:
            return self.injector.inject_fault(fault_type, target, params)
        except Exception as e:
            logger.error(f"Failed to inject fault: {e}")
            return None
    
    def stop_fault(self, fault_id: str) -> bool:
        """
        Stop a running fault.
        
        Args:
            fault_id: ID of the fault to stop
        
        Returns:
            True if fault was stopped, False otherwise
        """
        return self.injector.stop_fault(fault_id)
    
    def get_active_faults(self) -> Dict[str, Dict[str, Any]]:
        """Get all active faults."""
        return self.injector.get_active_faults()
    
    def get_fault_info(self, fault_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific fault."""
        return self.injector.get_fault_info(fault_id)
    
    def get_all_faults(self) -> Dict[str, Dict[str, Any]]:
        """Get all faults, including completed and stopped ones."""
        return self.injector.get_all_faults()

class ScheduledFaultInjector:
    """Scheduler for fault injection operations."""
    
    def __init__(self, fault_api: FaultInjectionAPI):
        """
        Initialize the scheduler.
        
        Args:
            fault_api: FaultInjectionAPI instance
        """
        self.fault_api = fault_api
        self.scheduled_faults: Dict[str, Dict[str, Any]] = {}
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
    
    def schedule_fault(self, fault_type: str, target: str, params: Dict[str, Any] = None, 
                      schedule_time: Optional[float] = None, delay_seconds: Optional[int] = None) -> str:
        """
        Schedule a fault for future execution.
        
        Args:
            fault_type: Type of fault to inject
            target: Target container or service URL
            params: Additional parameters for the fault
            schedule_time: Optional Unix timestamp for execution
            delay_seconds: Optional delay in seconds from now
        
        Returns:
            Schedule ID
        """
        if schedule_time is None and delay_seconds is None:
            raise ValueError("Must specify either schedule_time or delay_seconds")
        
        if schedule_time is None:
            schedule_time = time.time() + delay_seconds
        
        schedule_id = f"schedule_{int(time.time())}_{random.randint(1000, 9999)}"
        
        self.scheduled_faults[schedule_id] = {
            "fault_type": fault_type,
            "target": target,
            "params": params or {},
            "schedule_time": schedule_time,
            "status": "scheduled"
        }
        
        # Ensure scheduler is running
        self._ensure_scheduler_running()
        
        logger.info(f"Scheduled fault {fault_type} for {target} at {schedule_time} with ID {schedule_id}")
        return schedule_id
    
    def _ensure_scheduler_running(self):
        """Ensure the scheduler thread is running."""
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while not self.stop_event.is_set():
                current_time = time.time()
                
            # Check for faults to execute
                for schedule_id, schedule in list(self.scheduled_faults.items()):
                    if schedule["status"] == "scheduled" and schedule["schedule_time"] <= current_time:
                    try:
                        # Execute the fault
                        fault_id = self.fault_api.inject_fault(
                            schedule["fault_type"],
                            schedule["target"],
                            schedule["params"]
                        )
                        
                        if fault_id:
                        schedule["status"] = "executed"
                        schedule["fault_id"] = fault_id
                        logger.info(f"Executed scheduled fault {schedule_id} with fault ID {fault_id}")
                        else:
                            schedule["status"] = "failed"
                            logger.error(f"Failed to execute scheduled fault {schedule_id}")
            
            except Exception as e:
                        schedule["status"] = "failed"
                        schedule["error"] = str(e)
                        logger.error(f"Error executing scheduled fault {schedule_id}: {e}")
            
            time.sleep(1)
    
    def cancel_scheduled_fault(self, schedule_id: str) -> bool:
        """
        Cancel a scheduled fault.
        
        Args:
            schedule_id: ID of the scheduled fault
        
        Returns:
            True if cancelled successfully, False otherwise
        """
        if schedule_id not in self.scheduled_faults:
            logger.error(f"Scheduled fault {schedule_id} not found")
            return False
        
        schedule = self.scheduled_faults[schedule_id]
        if schedule["status"] != "scheduled":
            logger.error(f"Cannot cancel {schedule_id}, status is {schedule['status']}")
            return False
        
        schedule["status"] = "cancelled"
        logger.info(f"Cancelled scheduled fault {schedule_id}")
        return True
    
    def get_scheduled_faults(self) -> Dict[str, Dict[str, Any]]:
        """Get all scheduled faults that haven't been executed yet."""
        return {k: v for k, v in self.scheduled_faults.items() if v["status"] == "scheduled"}
    
    def get_all_schedules(self) -> Dict[str, Dict[str, Any]]:
        """Get all schedules, including executed and cancelled ones."""
        return self.scheduled_faults.copy()
    
    def stop(self):
        """Stop the scheduler."""
            self.stop_event.set()
        if self.scheduler_thread:
            self.scheduler_thread.join()