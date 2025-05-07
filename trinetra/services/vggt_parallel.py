#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import threading
import queue
import time
import numpy as np
import open3d as o3d
import os
from datetime import datetime

from typing import List, Dict, Any, Optional
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from utils.pcd_utils import save_point_cloud_to_pcd

from schemas.pcd_schemas import PointCloudResult, TimestampedFrameSet

# Import the PCDQueueVisualizer
from utils.pcd_queue_visualizer import PCDQueueVisualizer

class VGGTParallel:
    """
    Parallel processor for VGGT model that takes sets of frames,
    processes them on multiple GPU threads and saves point clouds.
    """
    
    def __init__(self, num_workers=3, output_dir="./output_pcd", save_interval=5.0, conf_threshold=25.0, visualize=False):
        """
        Initialize the parallel VGGT processor.
        
        Args:
            num_workers: Number of parallel GPU threads
            output_dir: Directory to save point cloud files
            save_interval: How often to save individual point clouds (seconds)
            conf_threshold: Confidence threshold percentile for filtering points
            visualize: Whether to visualize point clouds in Open3D renderer
        """
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.conf_threshold = conf_threshold
        self.running = False
        self.visualize = visualize
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if (self.device == "cuda" and 
                                    torch.cuda.get_device_capability()[0] >= 8) else torch.float16
        print(f"Using device: {self.device}, dtype: {self.dtype}")
        
        # Input queue for frame sets to be processed
        self.input_queue = queue.Queue(maxsize=30000)
        
        # Worker job queues - one queue per worker
        self.worker_queues = [queue.Queue(maxsize=10) for _ in range(num_workers)]
        
        # Available workers queue
        self.available_workers = queue.Queue(maxsize=num_workers)
        for i in range(num_workers):
            self.available_workers.put(i)
        
        # Worker tracking
        self.worker_status = {i: "idle" for i in range(num_workers)}
        self.worker_status_lock = threading.Lock()
        
        # Output queue for sorted results
        self.output_queue = queue.Queue(maxsize=300)
        
        # Results buffer for sorting by timestamp
        self.results_buffer = []
        self.results_lock = threading.Lock()
        
        # Point cloud visualization using the queue visualizer
        self.queue_visualizer = None
        if self.visualize:
            # Initialize the queue visualizer for real-time visualization
            self.queue_visualizer = PCDQueueVisualizer(delay=0.05)
        
        # Thread references
        self.processing_threads = []
        self.output_thread = None
        self.dispatcher_thread = None
        self.vis_thread = None
        
        # Last saved time
        self.last_save_time = time.time()
        
        # Initialize VGGT models for each worker
        print(f"Initializing {num_workers} VGGT models on {self.device}...")
        self.models = []
        for i in range(num_workers):
            model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
            model.eval()  # Set to evaluation mode
            self.models.append(model)
        print("All VGGT models initialized successfully")

    def _process_frames(self, worker_id):
        """
        Process frame sets through VGGT model in a worker thread.
        
        Args:
            worker_id: Worker thread ID
        """
        model = self.models[worker_id]
        print(f"Worker {worker_id} started on {self.device}")
        
        while self.running:
            try:
                # Get frame set from input queue
                frame_set = self.input_queue.get(timeout=0.5)
                
                # Extract frames and metadata
                frames = frame_set.frames
                timestamp = frame_set.timestamp
                batch_id = frame_set.batch_id
                
                try:
                    # Process frames through VGGT to generate point cloud
                    points, colors = self._process_point_cloud(model, frames, batch_id)
                    
                    if points is not None and len(points) > 0:
                        # Create result with timestamp
                        result = PointCloudResult(
                            points=points,
                            colors=colors,
                            timestamp=timestamp,
                            batch_id=batch_id
                        )
                        
                        # Add to results buffer with lock
                        with self.results_lock:
                            self.results_buffer.append(result)
                        
                        print(f"Worker {worker_id} processed batch {batch_id} with {len(points)} points")
                    else:
                        print(f"Worker {worker_id}: No valid points generated for batch {batch_id}")
                    
                except Exception as e:
                    print(f"Error in worker {worker_id} processing batch {batch_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Mark task as done
                self.input_queue.task_done()
                
            except queue.Empty:
                # No frames to process
                pass
    
    def _process_point_cloud(self, model, frames, batch_id):
        """
        Process frames through VGGT model and generate point cloud.
        Based on the provided code examples.
        
        Args:
            model: VGGT model instance
            frames: List of frames to process
            batch_id: Batch identifier for logging
            
        Returns:
            Tuple of (points, colors) arrays or (None, None) if processing failed
        """
        # Preprocess frames for VGGT
        preprocessed = self._preprocess_frames(frames)
        if preprocessed is None:
            return None, None
        
        # Run inference
        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast('cuda'):
                    predictions = model(preprocessed)
            else:
                predictions = model(preprocessed)
        
        # Convert pose encoding
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], 
            preprocessed.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        
        # Convert to numpy
        pred_dict = {}
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                pred_dict[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dim
        
        # Get points and confidence
        try:
            world_points = pred_dict["world_points"]  # (S, H, W, 3)
            conf = pred_dict["world_points_conf"]  # (S, H, W)
        except KeyError:
            # Fallback: compute world points from depth map
            world_points = unproject_depth_map_to_point_map(
                pred_dict["depth"], 
                pred_dict["extrinsic"], 
                pred_dict["intrinsic"]
            )
            conf = pred_dict["depth_conf"]
        
        # Get colors from images
        colors = pred_dict["images"].transpose(0, 2, 3, 1)  # (S, H, W, 3)
        
        # Flatten everything
        S, H, W, _ = world_points.shape
        points = world_points.reshape(-1, 3)
        colors_flat = colors.reshape(-1, 3)
        conf_flat = conf.reshape(-1)
        
        # Compute scene center
        scene_center = np.mean(points, axis=0)
        points_centered = points - scene_center
        
        # Filter by confidence
        threshold_val = np.percentile(conf_flat, self.conf_threshold)
        mask = (conf_flat >= threshold_val) & (conf_flat > 0.1)
        
        # Ensure we have valid points
        filtered_points = points_centered[mask]
        filtered_colors = colors_flat[mask]
        
        # Additional check to ensure we have valid data
        if len(filtered_points) == 0:
            print(f"Warning: No points passed confidence threshold for batch {batch_id}!")
            return None, None
            
        # Check for NaN or Inf values
        if np.isnan(filtered_points).any() or np.isinf(filtered_points).any():
            print(f"Warning: NaN or Inf values in point cloud for batch {batch_id}! Fixing...")
            valid_mask = ~(np.isnan(filtered_points).any(axis=1) | np.isinf(filtered_points).any(axis=1))
            filtered_points = filtered_points[valid_mask]
            filtered_colors = filtered_colors[valid_mask]
            
            if len(filtered_points) == 0:
                print(f"Error: All points were invalid after NaN/Inf check for batch {batch_id}!")
                return None, None
        
        # Ensure colors are in [0,1] range
        if filtered_colors.max() > 1.0:
            filtered_colors = filtered_colors / 255.0
        
        # Make sure colors are in [0,1] range
        filtered_colors = np.clip(filtered_colors, 0.0, 1.0)
        
        return filtered_points, filtered_colors
    
    def _preprocess_frames(self, frames):
        """
        Preprocess frames for VGGT model.
        
        Args:
            frames: List of numpy array frames
            
        Returns:
            Tensor ready for VGGT model
        """
        from vggt.utils.load_fn import load_and_preprocess_images
        import cv2
        import tempfile
        
        # Save frames to temporary files
        temp_dir = tempfile.mkdtemp()
        temp_paths = []
        
        try:
            for i, frame in enumerate(frames):
                temp_path = os.path.join(temp_dir, f"frame_{i}.png")
                cv2.imwrite(temp_path, frame)
                temp_paths.append(temp_path)
            
            # Use VGGT's preprocessing function
            images = load_and_preprocess_images(temp_paths).to(self.device)
            
            return images
            
        except Exception as e:
            print(f"Error preprocessing frames: {str(e)}")
            return None
        finally:
            # Clean up temporary files
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)
            try:
                os.rmdir(temp_dir)
            except:
                pass
    
    def _process_results(self):
        """Process results in timestamp order and send to the queue visualizer."""
        while self.running:
            # Get results from buffer
            results_to_process = []
            with self.results_lock:
                if self.results_buffer:
                    # Sort by timestamp
                    self.results_buffer.sort(key=lambda x: x.timestamp)
                    
                    # Get the earliest result
                    result = self.results_buffer.pop(0)
                    results_to_process.append(result)
            
            # Process results
            for result in results_to_process:
                timestamp = result.timestamp
                points = result.points
                colors = result.colors
                batch_id = result.batch_id
                
                # Print message
                print(f"Point Cloud with timestamp {timestamp:.3f} received")
                
                # Send to queue visualizer if enabled
                if self.visualize and self.queue_visualizer is not None:
                    self.queue_visualizer.add_point_cloud(
                        points=points,
                        colors=colors,
                        timestamp=timestamp, 
                        batch_id=batch_id
                    )
                
                # Optionally save to disk
                current_time = time.time()
                if current_time - self.last_save_time >= self.save_interval:
                    output_path = os.path.join(self.output_dir, f"point_cloud_{timestamp:.6f}.pcd")
                    self._save_point_cloud(points, colors, output_path)
                    self.last_save_time = current_time
            
            # Sleep to prevent high CPU usage
            time.sleep(0.1)
    
    def _save_point_cloud(self, points, colors, output_path):
        """Save point cloud to disk."""
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Write to PCD file
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"Saved point cloud to {output_path}")
        except Exception as e:
            print(f"Error saving point cloud: {str(e)}")
    
    def process_frame_set(self, frames, timestamp=None, batch_id=None):
        """
        Add a set of frames to be processed.
        
        Args:
            frames: List of numpy array frames (BGR format from OpenCV)
            timestamp: Optional timestamp for the frame set (default: current time)
            batch_id: Optional batch identifier (default: auto-generated)
            
        Returns:
            bool: True if frames were added to the queue, False otherwise
        """
        if not self.running:
            print("Processor not running")
            return False
        
        if not frames or len(frames) == 0:
            print("Empty frame set")
            return False
        
        # Set default timestamp and batch ID if not provided
        if timestamp is None:
            timestamp = time.time()
        
        if batch_id is None:
            batch_id = int(timestamp * 1000)
        
        # Create frame set
        frame_set = TimestampedFrameSet(
            frames=frames,
            timestamp=timestamp,
            batch_id=batch_id
        )
        
        # Add to input queue
        try:
            if not self.input_queue.full():
                self.input_queue.put(frame_set, block=False)
                return True
            else:
                print("Input queue full, dropping frame set")
                return False
        except queue.Full:
            print("Failed to add frame set to queue")
            return False
    def _dispatch_jobs(self):
        """
        Dispatcher thread that takes jobs from input queue and assigns them to available workers.
        """
        print("Dispatcher thread started")
        
        while self.running:
            try:
                # Get an available worker
                worker_id = self.available_workers.get(timeout=0.5)
                
                # Get a frame set from the input queue
                try:
                    frame_set = self.input_queue.get(timeout=0.5)
                    
                    # Put the frame set in the worker's job queue
                    self.worker_queues[worker_id].put(frame_set)
                    
                    print(f"Assigned batch {frame_set.batch_id} to worker {worker_id}")
                    
                    # Mark input queue task as done
                    self.input_queue.task_done()
                except queue.Empty:
                    # No frames to process, put the worker back
                    self.available_workers.put(worker_id)
            
            except queue.Empty:
                # No available workers
                pass
            
            except Exception as e:
                print(f"Error in dispatcher: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Don't lose an available worker if there's an error
                try:
                    self.available_workers.put(worker_id)
                except:
                    pass
            

    def _worker_thread(self, worker_id):
        """Worker thread function that processes frames from its own job queue."""
        model = self.models[worker_id]
        print(f"Worker {worker_id} started on {self.device}")
        
        while self.running:
            try:
                # Get frame set from worker's job queue instead of input queue
                frame_set = self.worker_queues[worker_id].get(timeout=0.5)
                
                # Update worker status
                with self.worker_status_lock:
                    self.worker_status[worker_id] = "busy"
                
                try:
                    # Extract frames and metadata
                    frames = frame_set.frames
                    timestamp = frame_set.timestamp
                    batch_id = frame_set.batch_id
                    
                    print(f"Worker {worker_id} processing batch {batch_id}")
                    
                    # Process frames through VGGT to generate point cloud
                    start_time = time.time()
                    points, colors = self._process_point_cloud(model, frames, batch_id)
                    current_time = time.time()
                    print(f'The time taken for the worker id {worker_id} is {current_time-start_time}')
                    if points is not None and len(points) > 0:
                        # Create result with timestamp
                        result = PointCloudResult(
                            points=points,
                            colors=colors,
                            timestamp=timestamp,
                            batch_id=batch_id
                        )
                        
                        # Add to results buffer with lock
                        with self.results_lock:
                            self.results_buffer.append(result)
                        
                        print(f"Worker {worker_id} completed batch {batch_id} with {len(points)} points")
                    else:
                        print(f"Worker {worker_id}: No valid points generated for batch {batch_id}")
                    
                except Exception as e:
                    print(f"Error in worker {worker_id} processing batch {batch_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Mark worker's job queue task as done
                self.worker_queues[worker_id].task_done()
                
                # Mark worker as available again
                with self.worker_status_lock:
                    self.worker_status[worker_id] = "idle"
                self.available_workers.put(worker_id)
                
            except queue.Empty:
                # No frames to process
                pass
            
            except Exception as e:
                print(f"Unexpected error in worker {worker_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Make sure worker becomes available again after an exception
                try:
                    with self.worker_status_lock:
                        self.worker_status[worker_id] = "idle"
                    self.available_workers.put(worker_id)
                except:
                    pass
    
    def start(self):
        """Start processing."""
        if self.running:
            print("Already running")
            return
        
        self.running = True
        
        # Start the queue visualizer if enabled
        if self.visualize and self.queue_visualizer is not None:
            self.queue_visualizer.start()
        
        # Start the dispatcher thread first
        self.dispatcher_thread = threading.Thread(target=self._dispatch_jobs)
        self.dispatcher_thread.daemon = True
        self.dispatcher_thread.start()
        
        # Start processing threads
        for i in range(self.num_workers):
            thread = threading.Thread(target=self._worker_thread, args=(i,))
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
        
        # Start result processing thread
        self.output_thread = threading.Thread(target=self._process_results)
        self.output_thread.daemon = True
        self.output_thread.start()
        
        print(f"VGGT parallel processor started with {self.num_workers} workers")
    
    def stop(self):
        """Stop processing and clear CUDA memory."""
        if not self.running:
            print("Not running")
            return
        
        self.running = False
        print("Stopping VGGT parallel processor...")
        
        # Stop the queue visualizer if enabled
        if self.visualize and self.queue_visualizer is not None:
            self.queue_visualizer.stop()
        
        # Wait for threads to finish
        if self.dispatcher_thread:
            self.dispatcher_thread.join(timeout=1.0)
        
        for thread in self.processing_threads:
            thread.join(timeout=1.0)
        
        if self.output_thread:
            self.output_thread.join(timeout=1.0)
        
        # Clear CUDA memory for each model
        print("Clearing CUDA memory...")
        for i, model in enumerate(self.models):
            # Move model to CPU first to free GPU memory
            model.to("cpu")
            # Delete model reference
            self.models[i] = None
        
        # Force CUDA garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("CUDA memory cleared")
        
        # Clear model list
        self.models = []
        
        print("VGGT parallel processor stopped")