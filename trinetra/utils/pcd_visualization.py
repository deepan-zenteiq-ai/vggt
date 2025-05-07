#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
import threading
import queue
import time
import os

class PointCloudVisualizer:
    """
    Thread-safe Open3D point cloud visualizer with non-blocking updates.
    """
    
    def __init__(self, window_name="Point Cloud Viewer", width=1280, height=720):
        """
        Initialize the point cloud visualizer.
        
        Args:
            window_name: Title of the visualization window
            width: Window width in pixels
            height: Window height in pixels
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        
        # Visualization components
        self.vis = None
        self.current_pcd = None
        
        # Thread-safe queuing
        self.vis_queue = queue.Queue(maxsize=10)
        self.vis_lock = threading.Lock()
        
        # State flags
        self.running = False
        self.vis_thread = None
    
    def initialize(self):
        """
        Initialize the Open3D visualization window and components.
        Returns:
            bool: True if visualization was initialized successfully, False otherwise
        """
        try:
            # Check if we're in a headless environment
            is_headless = os.environ.get('DISPLAY', '') == ''
            if is_headless:
                print("Detected headless environment, visualization not available")
                return False
                
            with self.vis_lock:
                # Create the visualizer with specific settings
                self.vis = o3d.visualization.Visualizer()
                
                try:
                    self.vis.create_window(
                        window_name=self.window_name, 
                        width=self.width, 
                        height=self.height,
                        visible=True,
                        left=50,
                        top=50
                    )
                except Exception as e:
                    print(f"Failed to create visualization window: {e}")
                    if self.vis is not None:
                        self.vis.destroy_window()
                    self.vis = None
                    return False
                
                # Create an empty point cloud
                self.current_pcd = o3d.geometry.PointCloud()
                self.vis.add_geometry(self.current_pcd)
                
                # Configure view and rendering options
                view_control = self.vis.get_view_control()
                view_control.set_zoom(0.8)
                render_option = self.vis.get_render_option()
                render_option.point_size = 2.0
                render_option.background_color = [0.1, 0.1, 0.1]  # Dark background
                
                # Test if we can render
                if not self.vis.poll_events():
                    print("Failed to poll visualization events, visualization not available")
                    self.vis.destroy_window()
                    self.vis = None
                    return False
                
                self.running = True
                print("Visualization initialized successfully")
                return True
                
        except Exception as e:
            print(f"Visualization initialization error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Clean up if partially initialized
            if self.vis is not None:
                try:
                    self.vis.destroy_window()
                except:
                    pass
                self.vis = None
                
            return False
    
    def update_point_cloud(self, points, colors):
        """
        Queue a point cloud for visualization.
        
        Args:
            points: Numpy array of 3D points (Nx3)
            colors: Numpy array of RGB colors (Nx3)
        """
        if not self.running or self.vis is None:
            return
        
        # Create a tuple of points and colors for visualization
        pcd_data = (points, colors)
        
        # Add to visualization queue, non-blocking (replace oldest if full)
        try:
            if not self.vis_queue.full():
                self.vis_queue.put(pcd_data, block=False)
            else:
                # Replace oldest item if queue is full
                try:
                    self.vis_queue.get_nowait()
                    self.vis_queue.put(pcd_data, block=False)
                except queue.Empty:
                    pass
        except queue.Full:
            pass
    
    def start_visualization_thread(self):
        """
        Start the visualization thread for continuous updates.
        """
        if self.vis_thread is not None and self.vis_thread.is_alive():
            return  # Already running
            
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.daemon = True
        self.vis_thread.start()
        print("Visualization thread started")
    
    def _visualization_thread(self):
        """Thread to handle visualization updates."""
        if not self.running or self.vis is None:
            print("Visualization components not initialized, thread exiting")
            return
            
        last_update_time = time.time()
        update_interval = 0.05  # 20 FPS maximum
        
        while self.running:
            current_time = time.time()
            
            # Limit update rate to avoid excessive CPU usage
            if current_time - last_update_time >= update_interval:
                # Try to get new point cloud data (non-blocking)
                try:
                    points, colors = self.vis_queue.get(block=False, timeout=0.01)
                    
                    with self.vis_lock:
                        # Update point cloud
                        self.current_pcd.points = o3d.utility.Vector3dVector(points)
                        self.current_pcd.colors = o3d.utility.Vector3dVector(colors)
                        
                        # Update geometry
                        self.vis.update_geometry(self.current_pcd)
                    
                    self.vis_queue.task_done()
                except (queue.Empty, Exception) as e:
                    # No new data or error, that's okay
                    if not isinstance(e, queue.Empty):
                        print(f"Error getting point cloud from queue: {str(e)}")
                
                # Update visualization
                try:
                    with self.vis_lock:
                        if not self.vis.poll_events():
                            print("Visualization window closed, stopping visualization")
                            self.running = False
                            break
                        self.vis.update_renderer()
                    last_update_time = current_time
                except Exception as e:
                    print(f"Error updating visualization: {str(e)}")
                    # If we encounter OpenGL/GLFW errors, try to continue but log them
                    time.sleep(0.5)  # Wait a bit longer to potentially recover
            
            # Sleep briefly to prevent high CPU usage
            # time.sleep(0.01)
        
        # Clean up when exiting
        self._cleanup()
    
    def update_visualization(self):
        """
        Update visualization from the main thread.
        
        Returns:
            bool: True if visualization should continue, False otherwise
        """
        if not self.running or self.vis is None:
            return False
        
        try:
            # Check for new point cloud data
            try:
                points, colors = self.vis_queue.get_nowait()
                
                with self.vis_lock:
                    # Update point cloud
                    self.current_pcd.points = o3d.utility.Vector3dVector(points)
                    self.current_pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    # Update geometry
                    self.vis.update_geometry(self.current_pcd)
                
                self.vis_queue.task_done()
            except queue.Empty:
                # No new data, that's okay
                pass
            
            # Update visualization
            with self.vis_lock:
                if not self.vis.poll_events():
                    print("Visualization window closed, stopping visualization")
                    self.running = False
                    return False
                self.vis.update_renderer()
            
            return True
            
        except Exception as e:
            print(f"Visualization error in main thread: {str(e)}")
            self.running = False
            return False
    
    def _cleanup(self):
        """Clean up visualization resources."""
        if self.vis is not None:
            try:
                with self.vis_lock:
                    self.vis.destroy_window()
            except Exception as e:
                print(f"Error closing visualization: {str(e)}")
            self.vis = None
        self.running = False
    
    def stop(self):
        """Stop visualization and clean up resources."""
        self.running = False
        
        # Wait for visualization thread to finish
        if self.vis_thread is not None and self.vis_thread.is_alive():
            self.vis_thread.join(timeout=1.0)
        
        self._cleanup()
        print("Visualization stopped")