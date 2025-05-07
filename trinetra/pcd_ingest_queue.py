#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import queue
import argparse

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

def process_point_clouds(directory):
    """Read point cloud files from a directory, queue them, and process them."""
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
        # Add to queue with PointCloudFile object directly
        file_queue.put(point_cloud_file)
        print(f"Added to queue: {point_cloud_file.filename} with timestamp {point_cloud_file.timestamp:.3f}")
    
    print("\nProcessing files in timestamp order:")
    
    # Process files from the queue in timestamp order
    while not file_queue.empty():
        file_obj = file_queue.get()
        print(f"Point Cloud with timestamp {file_obj.timestamp:.3f} received - File: {file_obj.filename}")
        
        # Small delay to simulate processing time
        time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser(description="Process point cloud files from a directory")
    parser.add_argument("--dir", type=str, default="./output_pcd", 
                        help="Directory containing point cloud files")
    args = parser.parse_args()
    
    process_point_clouds(args.dir)

if __name__ == "__main__":
    main()