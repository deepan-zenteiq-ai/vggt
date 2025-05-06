import os
import glob
import numpy as np
import open3d as o3d
import re
import time

def natural_sort_key(s):
    """Function to ensure numerical sorting of filenames"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def read_and_visualize_realtime_pcds(folder_path):
    """
    Reads PCD files in sequential order from a folder and visualizes them
    with no delay between frames for real-time visualization.
    
    Args:
        folder_path (str): Path to the folder containing PCD files
    """
    # Find all PCD files in the folder
    pcd_files = glob.glob(os.path.join(folder_path, "*.pcd"))
    
    # Sort the files to ensure they're processed in the correct order
    pcd_files.sort(key=natural_sort_key)
    
    if not pcd_files:
        print(f"No PCD files found in {folder_path}")
        return
    
    print(f"Found {len(pcd_files)} PCD files. Starting real-time visualization...")
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Real-time Point Cloud Viewer", width=1024, height=768)
    
    # Add options to the renderer
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background
    opt.point_size = 1.0
    
    # For measuring frame rate
    start_time = time.time()
    frame_count = 0
    
    # Prepare the first point cloud to set up the view
    if pcd_files:
        first_pcd = o3d.io.read_point_cloud(pcd_files[0])
        vis.add_geometry(first_pcd)
        
        # Set initial view based on the first point cloud
        view_control = vis.get_view_control()
        view_params = view_control.convert_to_pinhole_camera_parameters()
    
    # Remove the first point cloud before starting the loop
    vis.clear_geometries()
    
    # Create a placeholder for the current point cloud
    current_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(current_pcd)
    
    try:
        # Render each point cloud sequentially with no delay
        for i, pcd_file in enumerate(pcd_files):
            # Read the point cloud
            try:
                pcd = o3d.io.read_point_cloud(pcd_file)
                
                # Skip if point cloud is empty
                if len(np.asarray(pcd.points)) == 0:
                    print(f"Skipping empty point cloud: {pcd_file}")
                    continue
                
                # Update points and colors of the current point cloud
                current_pcd.points = pcd.points
                current_pcd.colors = pcd.colors if hasattr(pcd, 'colors') else None
                current_pcd.normals = pcd.normals if hasattr(pcd, 'normals') else None
                
                # Update the geometry
                vis.update_geometry(current_pcd)
                
                # Update view
                vis.poll_events()
                vis.update_renderer()
                
                # Calculate and display frame rate every 10 frames
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"Frame {i+1}/{len(pcd_files)} | FPS: {fps:.2f}")
                
            except Exception as e:
                print(f"Error reading {pcd_file}: {e}")
                continue
        
        # Final statistics
        total_time = time.time() - start_time
        total_fps = len(pcd_files) / total_time
        print(f"Visualization complete. Average FPS: {total_fps:.2f}")
        
        # Keep the window open until manually closed
        print("Finished sequence. Keeping the last frame visible.")
        vis.run()
        
    finally:
        vis.destroy_window()

if __name__ == "__main__":
    # Change this to your folder path containing the PCD files
    folder_path = "output_pcd/"
    
    # Start visualization with no delay between frames
    read_and_visualize_realtime_pcds(folder_path)