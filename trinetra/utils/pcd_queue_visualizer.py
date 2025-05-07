#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import queue
import threading
import open3d as o3d
import numpy as np

class PointCloudData:
    """Representation of a point cloud with timestamp."""
    def __init__(self, points, colors, timestamp=None, batch_id=None):
        self.points = points
        self.colors = colors
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.batch_id = batch_id
        
        # Generate a filename for debugging
        self.filename = f"point_cloud_{self.timestamp:.6f}.pcd"
    
    # Implement comparison methods for priority queue
    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    def __eq__(self, other):
        return self.timestamp == other.timestamp


class PCDQueueVisualizer:
    """Visualizer for point clouds using a priority queue."""
    
    def __init__(self, delay=0.1):
        """
        Initialize the PCD Queue Visualizer.
        
        Args:
            delay: Delay between processing each point cloud (seconds)
        """
        self.delay = delay
        self.running = False
        
        # Create a priority queue for the point clouds
        self.pcd_queue = queue.PriorityQueue()
        
        # Visualization components
        self.vis = None
        self.current_pcd = None
        
        # Visualization thread
        self.vis_thread = None
    
    def add_point_cloud(self, points, colors, timestamp=None, batch_id=None):
        """
        Add a point cloud to the visualization queue.
        
        Args:
            points: numpy array of 3D points (N, 3)
            colors: numpy array of RGB colors (N, 3)
            timestamp: Optional timestamp for the point cloud
            batch_id: Optional batch identifier
            
        Returns:
            bool: True if point cloud was added to the queue, False otherwise
        """
        if not self.running:
            print("Visualizer not running")
            return False
        
        if points is None or len(points) == 0:
            print("Empty point cloud data")
            return False
        
        # Create point cloud object
        pcd_data = PointCloudData(
            points=points,
            colors=colors,
            timestamp=timestamp,
            batch_id=batch_id
        )
        
        # Add to queue
        try:
            self.pcd_queue.put(pcd_data)
            print(f"Added to queue: Point cloud with timestamp {pcd_data.timestamp:.3f}")
            return True
        except Exception as e:
            print(f"Failed to add point cloud to queue: {str(e)}")
            return False
    
    def process_point_clouds(self):
        """Process point clouds from the queue and visualize them."""
        print("Starting point cloud visualization thread...")
        
        # Set up the visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        # Process the first point cloud to set up initial visualization
        first_pcd = None
        while self.running and first_pcd is None:
            try:
                first_pcd = self.pcd_queue.get(timeout=0.5)
            except queue.Empty:
                print("Waiting for initial point cloud...")
                time.sleep(0.1)
        
        if not self.running:
            self.vis.destroy_window()
            return
        
        print(f"Processing first point cloud with timestamp {first_pcd.timestamp:.3f}")
        
        # Create and add the first point cloud
        self.current_pcd = o3d.geometry.PointCloud()
        self.current_pcd.points = o3d.utility.Vector3dVector(first_pcd.points)
        self.current_pcd.colors = o3d.utility.Vector3dVector(first_pcd.colors)
        
        # Add geometry to visualizer
        self.vis.add_geometry(self.current_pcd)
        
        # Removed the coordinate frame/axis marker
        # No longer adding: coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        
        # Configure view and rendering options
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)
        render_option = self.vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = [0.1, 0.1, 0.1]  # Dark background
        
        # Process remaining point clouds from the queue in timestamp order
        print("\nProcessing point clouds in timestamp order:")
        
        while self.running:
            try:
                # Get the next point cloud
                pcd_data = self.pcd_queue.get(timeout=0.5)
                print(f"Point Cloud with timestamp {pcd_data.timestamp:.3f} received")
                
                # Update the point cloud data
                self.current_pcd.points = o3d.utility.Vector3dVector(pcd_data.points)
                self.current_pcd.colors = o3d.utility.Vector3dVector(pcd_data.colors)
                
                # Update geometry in the visualizer
                self.vis.update_geometry(self.current_pcd)
                
                # Update visualization
                if not self.vis.poll_events():
                    print("Visualization window closed")
                    break
                self.vis.update_renderer()
                
                # Mark as done
                self.pcd_queue.task_done()
                
                # Delay to simulate real-time processing or to control visualization speed
                time.sleep(self.delay)
                
            except queue.Empty:
                # No point clouds to process
                if not self.vis.poll_events():
                    print("Visualization window closed")
                    break
                self.vis.update_renderer()
                time.sleep(0.01)  # Short sleep to prevent high CPU usage
            
            except Exception as e:
                print(f"Error processing point cloud: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Delay to prevent error spam
        
        # Clean up
        self.vis.destroy_window()
        print("Visualization thread stopped")
    
    def start(self):
        """Start the visualizer."""
        if self.running:
            print("Visualizer already running")
            return
        
        self.running = True
        
        # Start processing thread
        self.vis_thread = threading.Thread(target=self.process_point_clouds)
        self.vis_thread.daemon = True
        self.vis_thread.start()
        
        print("PCD Queue Visualizer started")
    
    def stop(self):
        """Stop the visualizer."""
        if not self.running:
            print("Visualizer not running")
            return
        
        self.running = False
        
        # Wait for thread to finish
        if self.vis_thread:
            self.vis_thread.join(timeout=1.0)
        
        print("PCD Queue Visualizer stopped")