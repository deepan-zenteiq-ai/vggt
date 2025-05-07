#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import queue
import argparse
import open3d as o3d
import numpy as np
import threading

class PointCloudFile:
    """Representation of a point cloud file with timestamp."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        
        # Extract timestamp from filename (assuming format point_cloud_TIMESTAMP.pcd)
        try:
            # Split by underscore and get the last part before .pcd
            timestamp_str = self.filename.split('_')[-1].split('.')[0:2]
            timestamp_str = '.'.join(timestamp_str)  # Join with period for full timestamp
            self.timestamp = float(timestamp_str)
        except (IndexError, ValueError):
            # If filename doesn't follow expected format, use file creation time
            self.timestamp = os.path.getctime(file_path)
    
    # Implement comparison methods for priority queue
    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    def __eq__(self, other):
        return self.timestamp == other.timestamp


def process_and_visualize_point_clouds(directory, delay=0.1):
    """Read point cloud files from a directory, queue them, and visualize them."""
    # Create a priority queue for the files
    file_queue = queue.PriorityQueue()
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return
    
    # Scan for PCD files
    pcd_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.pcd'):
            file_path = os.path.join(directory, filename)
            pcd_files.append(file_path)
    
    if not pcd_files:
        print(f"No point cloud files found in {directory}")
        return
    
    print(f"Found {len(pcd_files)} point cloud files in {directory}")
    
    # Add files to the queue
    for file_path in pcd_files:
        point_cloud_file = PointCloudFile(file_path)
        file_queue.put(point_cloud_file)
        print(f"Added to queue: {point_cloud_file.filename} with timestamp {point_cloud_file.timestamp:.3f}")
    
    # Set up the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Process the first file to set up initial visualization
    if not file_queue.empty():
        first_file = file_queue.get()
        print(f"Processing first point cloud: {first_file.filename}")
        
        # Read and add the first point cloud
        current_pcd = o3d.io.read_point_cloud(first_file.file_path)
        
        # Set a color for better visualization (optional)
        current_pcd.paint_uniform_color([0.5, 0.5, 0.9])
        
        # Add geometry to visualizer
        vis.add_geometry(current_pcd)
    
    # Process remaining files from the queue in timestamp order
    print("\nProcessing files in timestamp order:")
    
    while not file_queue.empty():
        # Get the next file
        file_obj = file_queue.get()
        print(f"Point Cloud with timestamp {file_obj.timestamp:.3f} received - File: {file_obj.filename}")
        
        # Read the new point cloud
        new_pcd = o3d.io.read_point_cloud(file_obj.file_path)
        
        # Update the point cloud data
        current_pcd.points = new_pcd.points
        current_pcd.colors = new_pcd.colors if new_pcd.has_colors() else o3d.utility.Vector3dVector(np.array([[0.5, 0.5, 0.9]] * len(new_pcd.points)))
        current_pcd.normals = new_pcd.normals if new_pcd.has_normals() else current_pcd.normals
        
        # Update geometry in the visualizer
        vis.update_geometry(current_pcd)
        vis.poll_events()
        vis.update_renderer()
        
        # Delay to simulate real-time processing or to control visualization speed
        time.sleep(delay)
    
    # Keep the visualization window open until manually closed
    print("Finished processing all point clouds. Close the visualization window to exit.")
    vis.run()  # This will block until the window is closed
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Process and visualize point cloud files from a directory")
    parser.add_argument("--dir", type=str, default="./output_pcd", 
                        help="Directory containing point cloud files")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between processing each point cloud (seconds)")
    args = parser.parse_args()
    
    process_and_visualize_point_clouds(args.dir, args.delay)


if __name__ == "__main__":
    main()