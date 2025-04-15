#!/usr/bin/env python3
"""
ThreadMaster - A comprehensive thread management library with GUI
Supports efficient creation, synchronization, and termination of threads
Designed for high-performance computing applications
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import time
import psutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import logging
import uuid
import json
import os
from collections import defaultdict
from enum import Enum, auto
from typing import Dict, List, Callable, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ThreadMaster")

class ThreadPriority(Enum):
    """Thread priority levels"""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()

class ThreadStatus(Enum):
    """Thread status states"""
    PENDING = auto()
    RUNNING = auto()
    WAITING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TERMINATED = auto()

class ThreadGroup:
    """Manages a group of related threads"""
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.threads = []
        self.group_lock = threading.RLock()
        self.creation_time = time.time()
        self.group_id = str(uuid.uuid4())
        
    def add_thread(self, thread):
        """Add a thread to this group"""
        with self.group_lock:
            self.threads.append(thread)
            
    def remove_thread(self, thread):
        """Remove a thread from this group"""
        with self.group_lock:
            if thread in self.threads:
                self.threads.remove(thread)
                
    def terminate_all(self):
        """Terminate all threads in this group"""
        with self.group_lock:
            for thread in self.threads:
                thread.terminate()
                
    def get_stats(self):
        """Get statistics about this thread group"""
        with self.group_lock:
            total = len(self.threads)
            running = sum(1 for t in self.threads if t.status == ThreadStatus.RUNNING)
            waiting = sum(1 for t in self.threads if t.status == ThreadStatus.WAITING)
            completed = sum(1 for t in self.threads if t.status == ThreadStatus.COMPLETED)
            failed = sum(1 for t in self.threads if t.status == ThreadStatus.FAILED)
            terminated = sum(1 for t in self.threads if t.status == ThreadStatus.TERMINATED)
            
            return {
                "total": total,
                "running": running,
                "waiting": waiting,
                "completed": completed,
                "failed": failed,
                "terminated": terminated
            }
            
    def __str__(self):
        return f"ThreadGroup({self.name}, threads={len(self.threads)})"


class ManagedThread:
    """A managed thread with enhanced functionality"""
    def __init__(self, 
                target: Callable, 
                args=(), 
                kwargs=None, 
                name: str = None,
                priority: ThreadPriority = ThreadPriority.NORMAL,
                group: ThreadGroup = None):
        
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.name = name or f"Thread-{uuid.uuid4().hex[:8]}"
        self.priority = priority
        self.group = group
        self.thread_id = str(uuid.uuid4())
        
        # Thread state
        self.status = ThreadStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.exception = None
        self.result = None
        
        # Synchronization primitives
        self.exit_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # Not paused by default
        self.thread_lock = threading.RLock()
        
        # Create the actual thread
        self._thread = threading.Thread(target=self._thread_wrapper, name=self.name)
        self._thread.daemon = True
        
        # Add to group if specified
        if self.group:
            self.group.add_thread(self)
            
        logger.info(f"Thread created: {self.name} ({self.thread_id})")
        
    def _thread_wrapper(self):
        """Wrapper for the target function to handle thread management"""
        self.start_time = time.time()
        self.update_status(ThreadStatus.RUNNING)
        
        try:
            while not self.exit_event.is_set():
                # Wait if paused
                self.pause_event.wait()
                
                # Check for exit signal again after potential pause
                if self.exit_event.is_set():
                    break
                    
                # Execute the target function
                self.result = self.target(*self.args, **self.kwargs)
                self.update_status(ThreadStatus.COMPLETED)
                break
                
        except Exception as e:
            self.exception = e
            self.update_status(ThreadStatus.FAILED)
            logger.error(f"Thread {self.name} failed with exception: {str(e)}")
            
        finally:
            self.end_time = time.time()
            
    def start(self):
        """Start the thread"""
        with self.thread_lock:
            if self.status == ThreadStatus.PENDING:
                self._thread.start()
                logger.info(f"Thread started: {self.name}")
                return True
            return False
            
    def join(self, timeout=None):
        """Join the thread with optional timeout"""
        return self._thread.join(timeout)
    
    def terminate(self):
        """Signal the thread to exit"""
        with self.thread_lock:
            self.exit_event.set()
            self.pause_event.set()  # Unpause if paused
            self.update_status(ThreadStatus.TERMINATED)
            logger.info(f"Thread termination requested: {self.name}")
            
    def pause(self):
        """Pause the thread execution"""
        with self.thread_lock:
            if self.status == ThreadStatus.RUNNING:
                self.pause_event.clear()
                self.update_status(ThreadStatus.WAITING)
                logger.info(f"Thread paused: {self.name}")
                return True
            return False
            
    def resume(self):
        """Resume the thread execution"""
        with self.thread_lock:
            if self.status == ThreadStatus.WAITING:
                self.pause_event.set()
                self.update_status(ThreadStatus.RUNNING)
                logger.info(f"Thread resumed: {self.name}")
                return True
            return False
            
    def is_alive(self):
        """Check if the thread is alive"""
        return self._thread.is_alive()
        
    def update_status(self, status: ThreadStatus):
        """Update the thread status"""
        with self.thread_lock:
            self.status = status
            
    def get_runtime(self):
        """Get the thread's running time in seconds"""
        if self.start_time is None:
            return 0
            
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
        
    def to_dict(self):
        """Convert thread info to dictionary"""
        return {
            'id': self.thread_id,
            'name': self.name,
            'status': self.status.name,
            'priority': self.priority.name,
            'runtime': self.get_runtime(),
            'group': self.group.name if self.group else None,
            'start_time': self.start_time,
            'has_error': self.exception is not None
        }
        
    def __str__(self):
        return f"ManagedThread({self.name}, status={self.status.name})"


class ThreadPoolExecutor:
    """Thread pool for efficient thread management"""
    
    def __init__(self, max_workers=None, name="DefaultPool"):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.name = name
        self.pool_id = str(uuid.uuid4())
        
        # Worker management
        self.workers = []
        self.task_queue = queue.Queue()
        self.pool_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_tasks = 0
        self.creation_time = time.time()
        
        logger.info(f"Thread pool created: {self.name} with {self.max_workers} workers")
        
    def _worker_thread(self):
        """Worker thread that processes tasks from the queue"""
        while not self.shutdown_event.is_set():
            try:
                # Get a task from the queue with timeout
                task = self.task_queue.get(timeout=0.5)
                
                # Process the task
                managed_thread, future = task
                managed_thread.start()
                managed_thread.join()
                
                # Update result or exception in the future
                if managed_thread.exception:
                    future.set_exception(managed_thread.exception)
                    with self.pool_lock:
                        self.failed_tasks += 1
                else:
                    future.set_result(managed_thread.result)
                    with self.pool_lock:
                        self.completed_tasks += 1
                        
                self.task_queue.task_done()
                
            except queue.Empty:
                # No tasks available, continue waiting
                continue
                
            except Exception as e:
                logger.error(f"Worker thread error: {str(e)}")
                
    def start(self):
        """Start the thread pool"""
        with self.pool_lock:
            # Create worker threads
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self._worker_thread,
                    name=f"{self.name}-Worker-{i}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
                
            logger.info(f"Thread pool started: {self.name}")
            
    def submit(self, target, args=(), kwargs=None, priority=ThreadPriority.NORMAL, name=None):
        """Submit a task to the thread pool"""
        with self.pool_lock:
            if self.shutdown_event.is_set():
                raise RuntimeError("Cannot submit tasks to a shutdown thread pool")
                
            # Create future for result
            future = ThreadFuture()
            
            # Create managed thread
            thread = ManagedThread(
                target=target,
                args=args,
                kwargs=kwargs,
                priority=priority,
                name=name
            )
            
            # Add to queue
            self.task_queue.put((thread, future))
            self.total_tasks += 1
            
            return future
            
    def shutdown(self, wait=True):
        """Shutdown the thread pool"""
        with self.pool_lock:
            self.shutdown_event.set()
            
            if wait:
                # Wait for all tasks to complete
                self.task_queue.join()
                
                # Wait for all workers to terminate
                for worker in self.workers:
                    worker.join()
                    
            logger.info(f"Thread pool shutdown: {self.name}")
            
    def get_stats(self):
        """Get statistics about this thread pool"""
        with self.pool_lock:
            return {
                "name": self.name,
                "max_workers": self.max_workers,
                "active_workers": len(self.workers),
                "queue_size": self.task_queue.qsize(),
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "total_tasks": self.total_tasks,
                "uptime": time.time() - self.creation_time
            }


class ThreadFuture:
    """Future object for asynchronous results"""
    
    def __init__(self):
        self._result = None
        self._exception = None
        self._done_event = threading.Event()
        self._callbacks = []
        self._lock = threading.Lock()
        
    def set_result(self, result):
        """Set the result of the future"""
        with self._lock:
            self._result = result
            self._done_event.set()
            self._invoke_callbacks()
            
    def set_exception(self, exception):
        """Set an exception for the future"""
        with self._lock:
            self._exception = exception
            self._done_event.set()
            self._invoke_callbacks()
            
    def _invoke_callbacks(self):
        """Invoke all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Error in future callback: {str(e)}")
                
    def add_done_callback(self, callback):
        """Add a callback to be invoked when the future is done"""
        with self._lock:
            if self._done_event.is_set():
                # Already done, invoke immediately
                try:
                    callback(self)
                except Exception as e:
                    logger.error(f"Error in future callback: {str(e)}")
            else:
                self._callbacks.append(callback)
                
    def result(self, timeout=None):
        """Get the result of the future, waiting if necessary"""
        if not self._done_event.wait(timeout):
            raise TimeoutError("Future operation timed out")
            
        if self._exception:
            raise self._exception
            
        return self._result
        
    def exception(self, timeout=None):
        """Get the exception of the future, waiting if necessary"""
        if not self._done_event.wait(timeout):
            raise TimeoutError("Future operation timed out")
            
        return self._exception
        
    def done(self):
        """Check if the future is done"""
        return self._done_event.is_set()
        
    def cancel(self):
        """Cancel the future (if possible)"""
        # This is a simplified implementation that doesn't actually cancel the task
        return False


class Barrier:
    """A reusable barrier for thread synchronization"""
    
    def __init__(self, parties, action=None, timeout=None):
        self.parties = parties
        self.action = action
        self.timeout = timeout
        self.barrier_lock = threading.RLock()
        self.count = 0
        self.generation = 0
        self.event = threading.Event()
        
    def wait(self, timeout=None):
        """Wait for the barrier"""
        with self.barrier_lock:
            generation = self.generation
            count = self.count
            self.count += 1
            
            if self.count == self.parties:
                # Last thread to arrive
                self.count = 0
                self.generation += 1
                self.event.set()
                
                if self.action:
                    self.action()
                    
                return 0
                
        # Wait for barrier
        timeout = timeout if timeout is not None else self.timeout
        if not self.event.wait(timeout):
            with self.barrier_lock:
                if generation == self.generation:
                    # Timeout occurred
                    self.count -= 1
                    if self.count == 0:
                        self.event.clear()
                    raise TimeoutError("Barrier wait timed out")
                    
        # Check if generation changed while waiting
        if generation != self.generation:
            # Barrier was already broken
            return 0
            
        # Reset event for next generation if all threads have passed
        with self.barrier_lock:
            if generation == self.generation - 1:
                self.event.clear()
                
        return 0


class ThreadSemaphore:
    """Enhanced semaphore with timeout and owner tracking"""
    
    def __init__(self, value=1, name=None):
        self.semaphore = threading.Semaphore(value)
        self.name = name or f"Semaphore-{uuid.uuid4().hex[:8]}"
        self.value = value
        self.owners = set()
        self.owner_lock = threading.Lock()
        
    def acquire(self, blocking=True, timeout=None, owner=None):
        """Acquire the semaphore"""
        result = self.semaphore.acquire(blocking, timeout)
        
        if result and owner:
            with self.owner_lock:
                self.owners.add(owner)
                
        return result
        
    def release(self, owner=None):
        """Release the semaphore"""
        if owner:
            with self.owner_lock:
                if owner in self.owners:
                    self.owners.remove(owner)
                    
        self.semaphore.release()
        
    def get_owners(self):
        """Get the current owners of the semaphore"""
        with self.owner_lock:
            return list(self.owners)
            
    def __enter__(self):
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class EventBus:
    """Event bus for thread communication"""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.lock = threading.RLock()
        
    def subscribe(self, event_type, callback):
        """Subscribe to an event type"""
        with self.lock:
            self.subscribers[event_type].append(callback)
            
    def unsubscribe(self, event_type, callback):
        """Unsubscribe from an event type"""
        with self.lock:
            if event_type in self.subscribers:
                try:
                    self.subscribers[event_type].remove(callback)
                except ValueError:
                    pass
                    
    def publish(self, event_type, data=None):
        """Publish an event"""
        callbacks = []
        with self.lock:
            if event_type in self.subscribers:
                callbacks = self.subscribers[event_type].copy()
                
        # Call callbacks outside the lock
        for callback in callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event subscriber: {str(e)}")


class ThreadMaster:
    """Main thread management system"""
    
    def __init__(self):
        # Thread tracking
        self.threads = {}
        self.groups = {}
        self.pools = {}
        self.master_lock = threading.RLock()
        
        # Communication
        self.event_bus = EventBus()
        
        # CPU usage tracking
        self.cpu_history = []
        self.memory_history = []
        self.tracking_active = False
        self.tracking_interval = 1.0  # seconds
        self._tracking_thread = None
        
        # Synchronization tools
        self.barriers = {}
        self.semaphores = {}
        
        logger.info("ThreadMaster initialized")
        
    def create_thread(self, target, args=(), kwargs=None, name=None, 
                     priority=ThreadPriority.NORMAL, group_name=None,
                     auto_start=False):
        """Create a managed thread"""
        with self.master_lock:
            # Get or create group if specified
            group = None
            if group_name:
                group = self.groups.get(group_name)
                if not group:
                    group = self.create_group(group_name)
                    
            # Create thread
            thread = ManagedThread(
                target=target,
                args=args,
                kwargs=kwargs,
                name=name,
                priority=priority,
                group=group
            )
            
            # Store thread
            self.threads[thread.thread_id] = thread
            
            # Auto-start if requested
            if auto_start:
                thread.start()
                
            return thread
            
    def create_group(self, name, description=""):
        """Create a thread group"""
        with self.master_lock:
            if name in self.groups:
                return self.groups[name]
                
            group = ThreadGroup(name, description)
            self.groups[name] = group
            return group
            
    def create_pool(self, max_workers=None, name=None):
        """Create a thread pool"""
        with self.master_lock:
            pool = ThreadPoolExecutor(max_workers, name)
            self.pools[pool.pool_id] = pool
            pool.start()
            return pool
            
    def create_barrier(self, name, parties, action=None, timeout=None):
        """Create a barrier"""
        with self.master_lock:
            barrier = Barrier(parties, action, timeout)
            self.barriers[name] = barrier
            return barrier
            
    def create_semaphore(self, name, value=1):
        """Create a semaphore"""
        with self.master_lock:
            semaphore = ThreadSemaphore(value, name)
            self.semaphores[name] = semaphore
            return semaphore
            
    def get_thread(self, thread_id):
        """Get a thread by ID"""
        with self.master_lock:
            return self.threads.get(thread_id)
            
    def get_group(self, name):
        """Get a group by name"""
        with self.master_lock:
            return self.groups.get(name)
            
    def get_threads_by_status(self, status):
        """Get all threads with the specified status"""
        with self.master_lock:
            return [t for t in self.threads.values() if t.status == status]
            
    def get_threads_by_group(self, group_name):
        """Get all threads in the specified group"""
        with self.master_lock:
            group = self.groups.get(group_name)
            return group.threads if group else []
            
    def terminate_all(self):
        """Terminate all managed threads"""
        with self.master_lock:
            for thread in self.threads.values():
                thread.terminate()
                
            for pool in self.pools.values():
                pool.shutdown(wait=False)
                
    def start_resource_tracking(self):
        """Start tracking system resources"""
        with self.master_lock:
            if self.tracking_active:
                return False
                
            self.tracking_active = True
            self._tracking_thread = threading.Thread(
                target=self._track_resources,
                daemon=True,
                name="ResourceTracker"
            )
            self._tracking_thread.start()
            logger.info("Resource tracking started")
            return True
            
    def stop_resource_tracking(self):
        """Stop tracking system resources"""
        with self.master_lock:
            self.tracking_active = False
            if self._tracking_thread:
                self._tracking_thread.join(timeout=2.0)
                logger.info("Resource tracking stopped")
                
    def _track_resources(self):
        """Track CPU and memory usage"""
        max_history = 60  # Keep last 60 samples
        
        while self.tracking_active:
            try:
                # Get CPU and memory usage
                cpu = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory().percent
                
                # Add to history
                timestamp = time.time()
                with self.master_lock:
                    self.cpu_history.append((timestamp, cpu))
                    self.memory_history.append((timestamp, memory))
                    
                    # Trim history if needed
                    if len(self.cpu_history) > max_history:
                        self.cpu_history = self.cpu_history[-max_history:]
                    if len(self.memory_history) > max_history:
                        self.memory_history = self.memory_history[-max_history:]
                        
                # Sleep until next sample
                time.sleep(self.tracking_interval)
                
            except Exception as e:
                logger.error(f"Error in resource tracking: {str(e)}")
                time.sleep(1.0)  # Sleep on error
                
    def get_system_stats(self):
        """Get system statistics"""
        with self.master_lock:
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            return {
                "cpu_usage": cpu_usage,
                "memory_used": memory.used,
                "memory_total": memory.total,
                "memory_percent": memory.percent,
                "cpu_history": self.cpu_history,
                "memory_history": self.memory_history
            }
            
    def get_thread_stats(self):
        """Get thread statistics"""
        with self.master_lock:
            total = len(self.threads)
            running = len(self.get_threads_by_status(ThreadStatus.RUNNING))
            waiting = len(self.get_threads_by_status(ThreadStatus.WAITING))
            completed = len(self.get_threads_by_status(ThreadStatus.COMPLETED))
            failed = len(self.get_threads_by_status(ThreadStatus.FAILED))
            terminated = len(self.get_threads_by_status(ThreadStatus.TERMINATED))
            
            return {
                "total": total,
                "running": running,
                "waiting": waiting,
                "completed": completed,
                "failed": failed,
                "terminated": terminated,
                "groups": len(self.groups),
                "pools": len(self.pools)
            }
            
    def cleanup_completed(self):
        """Clean up completed and failed threads"""
        with self.master_lock:
            to_remove = []
            
            for thread_id, thread in self.threads.items():
                if thread.status in (ThreadStatus.COMPLETED, 
                                    ThreadStatus.FAILED,
                                    ThreadStatus.TERMINATED):
                    to_remove.append(thread_id)
                    
            # Remove threads
            for thread_id in to_remove:
                thread = self.threads.pop(thread_id)
                
                # Remove from group if needed
                if thread.group:
                    thread.group.remove_thread(thread)
                    
            return len(to_remove)


class ThreadMonitorGUI:
    """GUI for thread monitoring and management"""
    
    def __init__(self, master):
        self.thread_master = ThreadMaster()
        
        # Example threads/tasks
        self.demo_tasks = {
            "CPU Bound": self._cpu_bound_task,
            "IO Bound": self._io_bound_task,
            "Exception Task": self._exception_task,
            "Random Sleep": self._random_sleep_task,
            "Counter Task": self._counter_task
        }
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("ThreadMaster - Thread Management System")
        self.root.geometry("1280x720")
        self.root.minsize(800, 600)
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use("clam")  # Use a more modern theme
        
        # Configure colors
        self.style.configure("TButton", padding=6, relief="flat",
                           background="#4CAF50", foreground="black")
        self.style.configure("Danger.TButton", padding=6, relief="flat",
                           background="#f44336", foreground="white")
        self.style.configure("Warning.TButton", padding=6, relief="flat", 
                           background="#ff9800", foreground="black")
        self.style.configure("TFrame", background="#f5f5f5")
        self.style.configure("TNotebook", background="#f5f5f5")
        self.style.configure("TNotebook.Tab", padding=[12, 4],
                           background="#e0e0e0", foreground="black")
        self.style.map("TNotebook.Tab", background=[("selected", "#4CAF50")],
                     foreground=[("selected", "white")])
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.threads_tab = ttk.Frame(self.notebook)
        self.groups_tab = ttk.Frame(self.notebook)
        self.create_tab = ttk.Frame(self.notebook)
        self.system_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.threads_tab, text="Threads")
        self.notebook.add(self.groups_tab, text="Groups")
        self.notebook.add(self.create_tab, text="Create")
        self.notebook.add(self.system_tab, text="System")
        
        # Setup each tab
        self._setup_dashboard_tab()
        self._setup_threads_tab()
        self._setup_groups_tab()
        self._setup_create_tab()
        self._setup_system_tab()
        
        # Start resource tracking
        self.thread_master.start_resource_tracking()
        
        # Setup auto-refresh
        self.refresh_interval = 1000  # ms
        self.root.after(self.refresh_interval, self._auto_refresh)
        
        logger.info("ThreadMonitorGUI initialized")
    
    def _setup_dashboard_tab(self):
        """Setup the dashboard tab"""
        # Create frames
        stats_frame = ttk.Frame(self.dashboard_tab)
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Statistics section
        ttk.Label(stats_frame, text="Thread Statistics", font=('Arial', 14, 'bold')).pack(anchor=tk.W)
        
        # Stats display
        self.stats_display = ttk.Frame(stats_frame)
        self.stats_display.pack(fill=tk.X, pady=5)
        
        # Thread counts
        self.thread_counts = {
            "Total": tk.StringVar(value="0"),
            "Running": tk.StringVar(value="0"),
            "Waiting": tk.StringVar(value="0"),
            "Completed": tk.StringVar(value="0"),
            "Failed": tk.StringVar(value="0")
        }
        
        col = 0
        for label, var in self.thread_counts.items():
            frame = ttk.Frame(self.stats_display)
            frame.grid(row=0, column=col, padx=10)
            
            ttk.Label(frame, text=label).pack()
            ttk.Label(frame, textvariable=var, font=('Arial', 16, 'bold')).pack()
            
            col += 1
            
        # Charts section
        charts_frame = ttk.Frame(self.dashboard_tab)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # CPU and memory usage charts
        chart_left = ttk.Frame(charts_frame)
        chart_left = ttk.Frame(charts_frame)
        chart_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chart_right = ttk.Frame(charts_frame)
        chart_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # CPU usage chart
        ttk.Label(chart_left, text="CPU Usage", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.cpu_figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.cpu_plot = self.cpu_figure.add_subplot(111)
        self.cpu_canvas = FigureCanvasTkAgg(self.cpu_figure, chart_left)
        self.cpu_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Memory usage chart
        ttk.Label(chart_right, text="Memory Usage", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.memory_figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.memory_plot = self.memory_figure.add_subplot(111)
        self.memory_canvas = FigureCanvasTkAgg(self.memory_figure, chart_right)
        self.memory_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Recent activity section
        activity_frame = ttk.Frame(self.dashboard_tab)
        activity_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(activity_frame, text="Recent Activity", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        # Activity log
        self.activity_log = scrolledtext.ScrolledText(activity_frame, height=8)
        self.activity_log.pack(fill=tk.BOTH, expand=True, pady=5)
        self.activity_log.config(state=tk.DISABLED)
        
        # Action buttons
        button_frame = ttk.Frame(self.dashboard_tab)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Refresh", command=self._refresh_dashboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Create Demo Threads", command=self._create_demo_threads).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clean Up Completed", command=self._cleanup_completed).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Terminate All", command=self._terminate_all, style="Danger.TButton").pack(side=tk.RIGHT, padx=5)


    # CPU and memory usage charts
        chart_left = ttk.Frame(charts_frame)
        chart_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chart_right = ttk.Frame(charts_frame)
        chart_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # CPU usage chart
        ttk.Label(chart_left, text="CPU Usage", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.cpu_figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.cpu_plot = self.cpu_figure.add_subplot(111)
        self.cpu_canvas = FigureCanvasTkAgg(self.cpu_figure, chart_left)
        self.cpu_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Memory usage chart
        ttk.Label(chart_right, text="Memory Usage", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.memory_figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.memory_plot = self.memory_figure.add_subplot(111)
        self.memory_canvas = FigureCanvasTkAgg(self.memory_figure, chart_right)
        self.memory_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Recent activity section
        activity_frame = ttk.Frame(self.dashboard_tab)
        activity_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(activity_frame, text="Recent Activity", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        # Activity log
        self.activity_log = scrolledtext.ScrolledText(activity_frame, height=8)
        self.activity_log.pack(fill=tk.BOTH, expand=True, pady=5)
        self.activity_log.config(state=tk.DISABLED)
        
        # Action buttons
        button_frame = ttk.Frame(self.dashboard_tab)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Refresh", command=self._refresh_dashboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Create Demo Threads", command=self._create_demo_threads).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clean Up Completed", command=self._cleanup_completed).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Terminate All", command=self._terminate_all, style="Danger.TButton").pack(side=tk.RIGHT, padx=5)
            
    def _setup_threads_tab(self):
        """Setup the threads tab"""
        # Controls
        controls_frame = ttk.Frame(self.threads_tab)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(controls_frame, text="Filter:").pack(side=tk.LEFT, padx=5)
        self.thread_filter = ttk.Combobox(controls_frame, values=["All", "Running", "Waiting", "Completed", "Failed", "Terminated"])
        self.thread_filter.pack(side=tk.LEFT, padx=5)
        self.thread_filter.current(0)
        self.thread_filter.bind("<<ComboboxSelected>>", lambda e: self._refresh_threads())
        
        ttk.Button(controls_frame, text="Refresh", command=self._refresh_threads).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Terminate Selected", command=self._terminate_selected_thread, style="Warning.TButton").pack(side=tk.RIGHT, padx=5)
        
        # Thread list
        list_frame = ttk.Frame(self.threads_tab)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Thread list with scroll
        columns = ("ID", "Name", "Status", "Priority", "Runtime", "Group")
        self.thread_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        
        # Configure headings
        for col in columns:
            self.thread_tree.heading(col, text=col)
            
        # Configure column widths
        self.thread_tree.column("ID", width=80)
        self.thread_tree.column("Name", width=150)
        self.thread_tree.column("Status", width=100)
        self.thread_tree.column("Priority", width=100)
        self.thread_tree.column("Runtime", width=100)
        self.thread_tree.column("Group", width=150)
        
        # Add vertical scrollbar
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.thread_tree.yview)
        self.thread_tree.configure(yscrollcommand=vsb.set)
        
        # Add horizontal scrollbar
        hsb = ttk.Scrollbar(list_frame, orient="horizontal", command=self.thread_tree.xview)
        self.thread_tree.configure(xscrollcommand=hsb.set)
        
        # Pack everything
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.thread_tree.pack(fill=tk.BOTH, expand=True)
        
        # Bind double-click to view details
        self.thread_tree.bind("<Double-1>", self._show_thread_details)
            
    def _setup_groups_tab(self):
        """Setup the groups tab"""
        # Controls
        controls_frame = ttk.Frame(self.groups_tab)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls_frame, text="Refresh", command=self._refresh_groups).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Create Group", command=self._create_new_group).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Terminate Group", command=self._terminate_selected_group, style="Warning.TButton").pack(side=tk.RIGHT, padx=5)
        
        # Group list
        list_frame = ttk.Frame(self.groups_tab)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Group list with scroll
        columns = ("Name", "Thread Count", "Running", "Waiting", "Completed", "Failed")
        self.group_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        
        # Configure headings
        for col in columns:
            self.group_tree.heading(col, text=col)
            
        # Configure column widths
        self.group_tree.column("Name", width=150)
        self.group_tree.column("Thread Count", width=100)
        self.group_tree.column("Running", width=100)
        self.group_tree.column("Waiting", width=100)
        self.group_tree.column("Completed", width=100)
        self.group_tree.column("Failed", width=100)
        
        # Add vertical scrollbar
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.group_tree.yview)
        self.group_tree.configure(yscrollcommand=vsb.set)
        
        # Pack everything
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.group_tree.pack(fill=tk.BOTH, expand=True)
        
        # Bind double-click to view details
        self.group_tree.bind("<Double-1>", self._show_group_details)
            
    def _setup_create_tab(self):
        """Setup the create tab"""
        # Main frame with padding
        main_frame = ttk.Frame(self.create_tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create thread section
        thread_frame = ttk.LabelFrame(main_frame, text="Create Thread", padding=10)
        thread_frame.pack(fill=tk.X, pady=10)
        
        # Thread parameters
        param_frame = ttk.Frame(thread_frame)
        param_frame.pack(fill=tk.X)
        
        # Thread name
        ttk.Label(param_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.thread_name_var = tk.StringVar()
        ttk.Entry(param_frame, textvariable=self.thread_name_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Thread group
        ttk.Label(param_frame, text="Group:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.thread_group_var = tk.StringVar()
        self.thread_group_combo = ttk.Combobox(param_frame, textvariable=self.thread_group_var)
        self.thread_group_combo.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Thread priority
        ttk.Label(param_frame, text="Priority:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.thread_priority_var = tk.StringVar(value="NORMAL")
        ttk.Combobox(param_frame, textvariable=self.thread_priority_var, 
                    values=["LOW", "NORMAL", "HIGH", "CRITICAL"]).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Thread type
        ttk.Label(param_frame, text="Type:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.thread_type_var = tk.StringVar(value="CPU Bound")
        self.thread_type_combo = ttk.Combobox(param_frame, textvariable=self.thread_type_var, 
                                             values=list(self.demo_tasks.keys()))
        self.thread_type_combo.grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Parameters for the task
        ttk.Label(param_frame, text="Duration (s):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.task_duration_var = tk.StringVar(value="5")
        ttk.Entry(param_frame, textvariable=self.task_duration_var).grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create button
        button_frame = ttk.Frame(thread_frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Create Thread", command=self._create_thread).pack(side=tk.RIGHT)
        
        # Set column weights
        param_frame.columnconfigure(1, weight=1)
        
        # Create group section
        group_frame = ttk.LabelFrame(main_frame, text="Create Group", padding=10)
        group_frame.pack(fill=tk.X, pady=10)
        
        # Group parameters
        group_param_frame = ttk.Frame(group_frame)
        group_param_frame.pack(fill=tk.X)
        
        # Group name
        ttk.Label(group_param_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.group_name_var = tk.StringVar()
        ttk.Entry(group_param_frame, textvariable=self.group_name_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Group description
        ttk.Label(group_param_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.group_desc_var = tk.StringVar()
        ttk.Entry(group_param_frame, textvariable=self.group_desc_var).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create button
        group_button_frame = ttk.Frame(group_frame)
        group_button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(group_button_frame, text="Create Group", command=self._create_group).pack(side=tk.RIGHT)
        
        # Set column weights
        group_param_frame.columnconfigure(1, weight=1)
            
    def _setup_system_tab(self):
        """Setup the system tab"""
        # System info section
        info_frame = ttk.LabelFrame(self.system_tab, text="System Information", padding=10)
        info_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # System stats
        self.cpu_label = ttk.Label(info_frame, text="CPU Usage: -")
        self.cpu_label.pack(anchor=tk.W, pady=2)
        
        self.memory_label = ttk.Label(info_frame, text="Memory Usage: -")
        self.memory_label.pack(anchor=tk.W, pady=2)
        
        self.thread_count_label = ttk.Label(info_frame, text="Thread Count: -")
        self.thread_count_label.pack(anchor=tk.W, pady=2)
        
        # Thread pool section
        pool_frame = ttk.LabelFrame(self.system_tab, text="Thread Pool", padding=10)
        pool_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Thread pool controls
        pool_controls = ttk.Frame(pool_frame)
        pool_controls.pack(fill=tk.X)
        
        ttk.Label(pool_controls, text="Max Workers:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.pool_workers_var = tk.StringVar(value=str(os.cpu_count()))
        ttk.Entry(pool_controls, textvariable=self.pool_workers_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(pool_controls, text="Pool Name:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.pool_name_var = tk.StringVar(value="WorkerPool")
        ttk.Entry(pool_controls, textvariable=self.pool_name_var).grid(row=0, column=3, sticky=tk.W+tk.E, padx=5, pady=5)
        
        ttk.Button(pool_controls, text="Create Pool", command=self._create_pool).grid(row=0, column=4, padx=5, pady=5)
        
        # Set column weights
        pool_controls.columnconfigure(3, weight=1)
        
        # Pool list
        ttk.Label(pool_frame, text="Active Pools:").pack(anchor=tk.W, pady=5)
        
        pool_list_frame = ttk.Frame(pool_frame)
        pool_list_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("Name", "Workers", "Queue Size", "Completed", "Failed", "Total")
        self.pool_tree = ttk.Treeview(pool_list_frame, columns=columns, show="headings", height=5)
        
        # Configure headings
        for col in columns:
            self.pool_tree.heading(col, text=col)
            
        # Add vertical scrollbar
        vsb = ttk.Scrollbar(pool_list_frame, orient="vertical", command=self.pool_tree.yview)
        self.pool_tree.configure(yscrollcommand=vsb.set)
        
        # Pack everything
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.pool_tree.pack(fill=tk.BOTH, expand=True)
        
        # Logging section
        log_frame = ttk.LabelFrame(self.system_tab, text="System Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Log level control
        log_control = ttk.Frame(log_frame)
        log_control.pack(fill=tk.X)
        
        ttk.Label(log_control, text="Log Level:").pack(side=tk.LEFT, padx=5)
        self.log_level_var = tk.StringVar(value="INFO")
        ttk.Combobox(log_control, textvariable=self.log_level_var, 
                     values=["DEBUG", "INFO", "WARNING", "ERROR"]).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_control, text="Apply", command=self._set_log_level).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_control, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT, padx=5)
        
        # Log display
        self.system_log = scrolledtext.ScrolledText(log_frame)
        self.system_log.pack(fill=tk.BOTH, expand=True, pady=5)
        self.system_log.config(state=tk.DISABLED)
            
    def _auto_refresh(self):
        """Auto refresh the UI"""
        self._refresh_dashboard()
        self._refresh_system_tab()
        
        # Schedule next refresh
        self.root.after(self.refresh_interval, self._auto_refresh)
        
    def _refresh_dashboard(self):
        """Refresh the dashboard"""
        # Update thread statistics
        stats = self.thread_master.get_thread_stats()
        self.thread_counts["Total"].set(str(stats["total"]))
        self.thread_counts["Running"].set(str(stats["running"]))
        self.thread_counts["Waiting"].set(str(stats["waiting"]))
        self.thread_counts["Completed"].set(str(stats["completed"]))
        self.thread_counts["Failed"].set(str(stats["failed"]))
        
        # Update charts
        self._update_charts()
        
    def _refresh_threads(self):
        """Refresh the threads list"""
        # Clear the list
        for item in self.thread_tree.get_children():
            self.thread_tree.delete(item)
            
        # Get current filter
        filter_status = self.thread_filter.get()
        
        # Add threads to the list
        for thread_id, thread in self.thread_master.threads.items():
            # Apply filter
            if filter_status != "All" and thread.status.name != filter_status:
                continue
                
            # Add to tree
            self.thread_tree.insert("", tk.END, values=(
                thread_id[:8],
                thread.name,
                thread.status.name,
                thread.priority.name,
                f"{thread.get_runtime():.2f}s",
                thread.group.name if thread.group else "-"
            ))
            
    def _refresh_groups(self):
        """Refresh the groups list"""
        # Clear the list
        for item in self.group_tree.get_children():
            self.group_tree.delete(item)
            
        # Update the thread group combo in create tab
        self.thread_group_combo['values'] = [""] + list(self.thread_master.groups.keys())
        
        # Add groups to the list
        for name, group in self.thread_master.groups.items():
            stats = group.get_stats()
            
            self.group_tree.insert("", tk.END, values=(
                name,
                stats["total"],
                stats["running"],
                stats["waiting"],
                stats["completed"],
                stats["failed"]
            ))
            
    def _refresh_system_tab(self):
        """Refresh the system tab"""
        # Update system stats
        stats = self.thread_master.get_system_stats()
        self.cpu_label.config(text=f"CPU Usage: {stats['cpu_usage']:.1f}%")
        self.memory_label.config(text=f"Memory Usage: {stats['memory_percent']:.1f}% ({stats['memory_used'] / (1024**3):.1f} GB / {stats['memory_total'] / (1024**3):.1f} GB)")
        
        thread_stats = self.thread_master.get_thread_stats()
        self.thread_count_label.config(text=f"Thread Count: {thread_stats['total']} (Running: {thread_stats['running']}, Waiting: {thread_stats['waiting']})")
        
        # Update pool list
        self._refresh_pools()
        
    def _refresh_pools(self):
        """Refresh the pools list"""
        # Clear the list
        for item in self.pool_tree.get_children():
            self.pool_tree.delete(item)
            
        # Add pools to the list
        for pool_id, pool in self.thread_master.pools.items():
            stats = pool.get_stats()
            
            self.pool_tree.insert("", tk.END, values=(
                stats["name"],
                stats["max_workers"],
                stats["queue_size"],
                stats["completed_tasks"],
                stats["failed_tasks"],
                stats["total_tasks"]
            ))
            
    def _update_charts(self):
        """Update the charts"""
        # Get system stats
        stats = self.thread_master.get_system_stats()
        
        # CPU chart
        self.cpu_plot.clear()
        if stats["cpu_history"]:
            times, values = zip(*stats["cpu_history"])
            relative_times = [t - times[0] for t in times]
            self.cpu_plot.plot(relative_times, values, 'b-')
            self.cpu_plot.set_ylim(0, 100)
            self.cpu_plot.set_xlabel("Time (s)")
            self.cpu_plot.set_ylabel("CPU (%)")
            self.cpu_plot.grid(True)
        self.cpu_canvas.draw()
        
        # Memory chart
        self.memory_plot.clear()
        if stats["memory_history"]:
            times, values = zip(*stats["memory_history"])
            relative_times = [t - times[0] for t in times]
            self.memory_plot.plot(relative_times, values, 'r-')
            self.memory_plot.set_ylim(0, 100)
            self.memory_plot.set_xlabel("Time (s)")
            self.memory_plot.set_ylabel("Memory (%)")
            self.memory_plot.grid(True)
        self.memory_canvas.draw()
        
    def _create_demo_threads(self):
        """Create some demo threads"""
        group = self.thread_master.create_group("DemoGroup", "Demo threads group")
        
        # Create a variety of threads
        for i, task_type in enumerate(self.demo_tasks.keys()):
            self.thread_master.create_thread(
                target=self.demo_tasks[task_type],
                args=(5,),  # 5 seconds duration
                name=f"Demo-{task_type}-{i}",
                priority=ThreadPriority.NORMAL,
                group=group,
                auto_start=True
            )
            
        self._log_activity(f"Created demo threads in group 'DemoGroup'")
        self._refresh_threads()
        self._refresh_groups()
        
    def _cleanup_completed(self):
        """Clean up completed threads"""
        count = self.thread_master.cleanup_completed()
        self._log_activity(f"Cleaned up {count} completed threads")
        self._refresh_threads()
        self._refresh_groups()
        
    def _terminate_all(self):
        """Terminate all threads"""
        result = messagebox.askokcancel("Terminate All", "Are you sure you want to terminate all threads?")
        if result:
            self.thread_master.terminate_all()
            self._log_activity("Terminated all threads")
            self._refresh_threads()
            self._refresh_groups()
            
    def _terminate_selected_thread(self):
        """Terminate the selected thread"""
        selected = self.thread_tree.selection()
        if not selected:
            messagebox.showinfo("No Selection", "Please select a thread to terminate.")
            return
            
        thread_id = self.thread_tree.item(selected[0])['values'][0]
        
        # Find the full thread ID
        full_id = None
        for tid in self.thread_master.threads.keys():
            if tid.startswith(thread_id):
                full_id = tid
                break
                
        if full_id:
            thread = self.thread_master.get_thread(full_id)
            if thread:
                thread.terminate()
                self._log_activity(f"Terminated thread: {thread.name}")
                self._refresh_threads()
                
    def _terminate_selected_group(self):
        """Terminate the selected group"""
        selected = self.group_tree.selection()
        if not selected:
            messagebox.showinfo("No Selection", "Please select a group to terminate.")
            return
            
        group_name = self.group_tree.item(selected[0])['values'][0]
        group = self.thread_master.get_group(group_name)
        
        if group:
            group.terminate_all()
            self._log_activity(f"Terminated all threads in group: {group_name}")
            self._refresh_threads()
            self._refresh_groups()
            
    def _create_thread(self):
        """Create a new thread with user input"""
        # Get parameters
        name = self.thread_name_var.get() or f"Thread-{uuid.uuid4().hex[:8]}"
        group_name = self.thread_group_var.get()
        priority_name = self.thread_priority_var.get()
        task_type = self.thread_type_var.get()
        
        try:
            duration = float(self.task_duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Duration must be a number.")
            return
            
        # Get the group if specified
        group = None
        if group_name:
            group = self.thread_master.get_group(group_name)
            if not group:
                group = self.thread_master.create_group(group_name)
                
        # Get the task function
        task_func = self.demo_tasks.get(task_type)
        if not task_func:
            messagebox.showerror("Invalid Task", "Please select a valid task type.")
            return
            
        # Get the priority
        priority = ThreadPriority.NORMAL
        try:
            priority = ThreadPriority[priority_name]
        except KeyError:
            pass
            
        # Create and start the thread
        thread = self.thread_master.create_thread(
            target=task_func,
            args=(duration,),
            name=name,
            priority=priority,
            group_name=group_name if group_name else None,
            auto_start=True
        )
        
        self._log_activity(f"Created thread: {name} ({task_type}, {duration}s)")
        self._refresh_threads()
        self._refresh_groups()
        
    def _create_group(self):
        """Create a new group with user input"""
        # Get parameters
        name = self.group_name_var.get()
        description = self.group_desc_var.get()
        
        if not name:
            messagebox.showerror("Invalid Input", "Group name is required.")
            return
            
        # Create the group
        group = self.thread_master.create_group(name, description)
        
        self._log_activity(f"Created group: {name}")
        self._refresh_groups()
        
        # Clear inputs
        self.group_name_var.set("")
        self.group_desc_var.set("")
        
    def _create_new_group(self):
        """Create a new group from the groups tab"""
        # Show a dialog to get group name
        dialog = tk.Toplevel(self.root)
        dialog.title("Create Group")
        dialog.geometry("300x150")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center on parent
        dialog.geometry("+{}+{}".format(
            self.root.winfo_x() + int(self.root.winfo_width()/2 - 150),
            self.root.winfo_y() + int(self.root.winfo_height()/2 - 75)
        ))
        
        # Create widgets
        ttk.Label(dialog, text="Group Name:").pack(pady=(15, 5))
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var).pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(dialog, text="Description:").pack(pady=5)
        desc_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=desc_var).pack(fill=tk.X, padx=20, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=15)
        
        def on_cancel():
            dialog.destroy()
            
        def on_create():
            name = name_var.get()
            desc = desc_var.get()
            
            if not name:
                messagebox.showerror("Invalid Input", "Group name is required.", parent=dialog)
                return
                
            self.thread_master.create_group(name, desc)
            self._log_activity(f"Created group: {name}")
            self._refresh_groups()
            dialog.destroy()
            
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Create", command=on_create).pack(side=tk.RIGHT)
        
    def _create_pool(self):
        """Create a new thread pool"""
        # Get parameters
        try:
            max_workers = int(self.pool_workers_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Max workers must be a number.")
            return
            
        name = self.pool_name_var.get() or "Pool"
        
        # Create the pool
        pool = self.thread_master.create_pool(max_workers, name)
        
        self._log_activity(f"Created thread pool: {name} with {max_workers} workers")
        self._refresh_pools()
        
    



    def _show_thread_details(self, event):
        """Show details for the selected thread"""
        selected = self.thread_tree.selection()
        if not selected:
            return
            
        thread_id = self.thread_tree.item(selected[0])['values'][0]
        
        # Find the full thread ID
        full_id = None
        for tid in self.thread_master.threads.keys():
            if tid.startswith(thread_id):
                full_id = tid
                break
                
        if full_id:
            thread = self.thread_master.get_thread(full_id)
            if thread:
                self._show_thread_dialog(thread)
                
    def _show_group_details(self, event):
        """Show details for the selected group"""
        selected = self.group_tree.selection()
        if not selected:
            return
            
        group_name = self.group_tree.item(selected[0])['values'][0]
        group = self.thread_master.get_group(group_name)
        
        if group:
            self._show_group_dialog(group)
            
    def _show_thread_dialog(self, thread):
        """Show a dialog with thread details"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Thread Details: {thread.name}")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center on parent
        dialog.geometry("+{}+{}".format(
            self.root.winfo_x() + int(self.root.winfo_width()/2 - 250),
            self.root.winfo_y() + int(self.root.winfo_height()/2 - 200)
        ))
        
        # Thread info
        info_frame = ttk.LabelFrame(dialog, text="Thread Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create grid for thread info
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill=tk.X, expand=True)
        
        # Add thread properties
        row = 0
        for label, value in [
            ("ID", thread.thread_id),
            ("Name", thread.name),
            ("Status", thread.status.name),
            ("Priority", thread.priority.name),
            ("Runtime", f"{thread.get_runtime():.2f} seconds"),
            ("Group", thread.group.name if thread.group else "None"),
            ("Started", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(thread.start_time)) if thread.start_time else "Not started"),
            ("Ended", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(thread.end_time)) if thread.end_time else "Running"),
        ]:
            ttk.Label(info_grid, text=f"{label}:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Label(info_grid, text=str(value)).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            row += 1
            
        # Thread controls
        control_frame = ttk.Frame(dialog)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Add control buttons based on thread state
        if thread.status == ThreadStatus.RUNNING:
            ttk.Button(control_frame, text="Pause", command=lambda: self._pause_thread(thread, dialog)).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="Terminate", command=lambda: self._terminate_thread(thread, dialog), style="Warning.TButton").pack(side=tk.LEFT, padx=5)
        elif thread.status == ThreadStatus.WAITING:
            ttk.Button(control_frame, text="Resume", command=lambda: self._resume_thread(thread, dialog)).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="Terminate", command=lambda: self._terminate_thread(thread, dialog), style="Warning.TButton").pack(side=tk.LEFT, padx=5)
            
        # Exception info if any
        if thread.exception:
            exc_frame = ttk.LabelFrame(dialog, text="Exception Information", padding=10)
            exc_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            exc_text = scrolledtext.ScrolledText(exc_frame, height=8)
            exc_text.pack(fill=tk.BOTH, expand=True)
            exc_text.insert(tk.END, str(thread.exception))
            exc_text.config(state=tk.DISABLED)
            
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=10, pady=10)
        
    def _show_group_dialog(self, group):
        """Show a dialog with group details"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Group Details: {group.name}")
        dialog.geometry("700x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center on parent
        dialog.geometry("+{}+{}".format(
            self.root.winfo_x() + int(self.root.winfo_width()/2 - 350),
            self.root.winfo_y() + int(self.root.winfo_height()/2 - 250)
        ))
        
        # Group info
        info_frame = ttk.LabelFrame(dialog, text="Group Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create grid for group info
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill=tk.X, expand=True)
        
        # Add group properties
        stats = group.get_stats()
        row = 0
        for label, value in [
            ("Name", group.name),
            ("Description", group.description or "No description"),
            ("Created", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(group.creation_time))),
            ("Thread Count", stats["total"]),
            ("Running", stats["running"]),
            ("Waiting", stats["waiting"]),
            ("Completed", stats["completed"]),
            ("Failed", stats["failed"])
        ]:
            ttk.Label(info_grid, text=f"{label}:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Label(info_grid, text=str(value)).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            row += 1
            
        # Group threads
        threads_frame = ttk.LabelFrame(dialog, text="Group Threads", padding=10)
        threads_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Thread list with scroll
        columns = ("ID", "Name", "Status", "Priority", "Runtime")
        group_threads_tree = ttk.Treeview(threads_frame, columns=columns, show="headings")
        
        # Configure headings
        for col in columns:
            group_threads_tree.heading(col, text=col)
            
        # Configure column widths
        group_threads_tree.column("ID", width=80)
        group_threads_tree.column("Name", width=150)
        group_threads_tree.column("Status", width=100)
        group_threads_tree.column("Priority", width=100)
        group_threads_tree.column("Runtime", width=100)
        
        # Add vertical scrollbar
        vsb = ttk.Scrollbar(threads_frame, orient="vertical", command=group_threads_tree.yview)
        group_threads_tree.configure(yscrollcommand=vsb.set)
        
        # Pack everything
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        group_threads_tree.pack(fill=tk.BOTH, expand=True)
        
        # Populate thread list
        for thread in group.threads:
            group_threads_tree.insert("", tk.END, values=(
                thread.thread_id[:8],
                thread.name,
                thread.status.name,
                thread.priority.name,
                f"{thread.get_runtime():.2f}s"
            ))
            
        # Group controls
        control_frame = ttk.Frame(dialog)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Terminate All", command=lambda: self._terminate_group(group, dialog), style="Warning.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh", command=lambda: self._refresh_group_dialog(dialog, group, group_threads_tree)).pack(side=tk.LEFT, padx=5)
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=10, pady=10)
        
    def _refresh_group_dialog(self, dialog, group, tree):
        """Refresh the group dialog thread list"""
        # Clear the list
        for item in tree.get_children():
            tree.delete(item)
            
        # Populate thread list
        for thread in group.threads:
            tree.insert("", tk.END, values=(
                thread.thread_id[:8],
                thread.name,
                thread.status.name,
                thread.priority.name,
                f"{thread.get_runtime():.2f}s"
            ))
            
    def _pause_thread(self, thread, dialog=None):
        """Pause a thread"""
        if thread.pause():
            self._log_activity(f"Paused thread: {thread.name}")
            self._refresh_threads()
            if dialog:
                dialog.destroy()
                self._show_thread_dialog(thread)
                
    def _resume_thread(self, thread, dialog=None):
        """Resume a thread"""
        if thread.resume():
            self._log_activity(f"Resumed thread: {thread.name}")
            self._refresh_threads()
            if dialog:
                dialog.destroy()
                self._show_thread_dialog(thread)
                
    def _terminate_thread(self, thread, dialog=None):
        """Terminate a thread"""
        thread.terminate()
        self._log_activity(f"Terminated thread: {thread.name}")
        self._refresh_threads()
        if dialog:
            dialog.destroy()
            
    def _terminate_group(self, group, dialog=None):
        """Terminate all threads in a group"""
        result = messagebox.askokcancel("Terminate Group", f"Are you sure you want to terminate all threads in group '{group.name}'?", parent=dialog)
        if result:
            group.terminate_all()
            self._log_activity(f"Terminated all threads in group: {group.name}")
            self._refresh_threads()
            self._refresh_groups()
            if dialog:
                dialog.destroy()
                
    def _set_log_level(self):
        """Set the log level"""
        level_name = self.log_level_var.get()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)
        self._log_activity(f"Set log level to {level_name}")
        
    def _clear_log(self):
        """Clear the system log"""
        self.system_log.config(state=tk.NORMAL)
        self.system_log.delete(1.0, tk.END)
        self.system_log.config(state=tk.DISABLED)
        
    def _log_activity(self, message):
        """Log an activity message"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        log_message = f"{timestamp} - {message}"
        
        # Add to activity log
        self.activity_log.config(state=tk.NORMAL)
        self.activity_log.insert(tk.END, log_message + "\n")
        self.activity_log.see(tk.END)
        self.activity_log.config(state=tk.DISABLED)
        
        # Add to system log
        self.system_log.config(state=tk.NORMAL)
        self.system_log.insert(tk.END, log_message + "\n")
        self.system_log.see(tk.END)
        self.system_log.config(state=tk.DISABLED)
        
        # Log to logger
        logger.info(message)
        
    # Demo task functions
    def _cpu_bound_task(self, duration):
        """CPU-bound task for testing"""
        start_time = time.time()
        result = 0
        
        while time.time() - start_time < duration:
            # Simulate CPU-intensive work
            for i in range(10000):
                result += i * i
                
        return result
        
    def _io_bound_task(self, duration):
        """IO-bound task for testing"""
        chunks = int(duration)
        
        for i in range(chunks):
            # Simulate IO operation
            time.sleep(1)
            
        return f"Completed {chunks} IO operations"
        
    def _exception_task(self, duration):
        """Task that raises an exception"""
        time.sleep(duration / 2)
        raise ValueError("This is a test exception")
        
    def _random_sleep_task(self, duration):
        """Task with random sleeps"""
        import random
        start_time = time.time()
        sleeps = []
        
        while time.time() - start_time < duration:
            sleep_time = random.uniform(0.1, 0.5)
            time.sleep(sleep_time)
            sleeps.append(sleep_time)
            
        return f"Completed {len(sleeps)} random sleeps"
        
    def _counter_task(self, duration):
        """Simple counter task"""
        count = 0
        end_time = time.time() + duration
        
        while time.time() < end_time:
            count += 1
            time.sleep(0.01)
            
        return f"Counted to {count}"
        
    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()
        
    def _on_close(self):
        """Handle window close event"""
        # Stop resource tracking
        self.thread_master.stop_resource_tracking()
        
        # Terminate all threads
        self.thread_master.terminate_all()
        
        # Destroy the window
        self.root.destroy()


def run_demo():
    """Run the thread management demo"""
    # Create and run the GUI
    gui = ThreadMonitorGUI(None)
    gui.run()
    
    
if __name__ == "__main__":
    run_demo()