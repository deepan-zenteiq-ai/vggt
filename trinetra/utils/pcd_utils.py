import open3d as o3d
import numpy as np
def save_point_cloud_to_pcd(points, colors, output_path):
    """
    Save 3D points and their colors to a PCD file.
    
    Args:
        points (numpy.ndarray): Point coordinates with shape (N, 3)
        colors (numpy.ndarray): RGB colors with shape (N, 3), values in range [0, 1]
        output_path (str): Path where to save the PCD file
    """
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Set points
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Convert colors from [0, 255] to [0, 1] float format if needed
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255.0
    
    # Make sure colors are in [0,1] range
    colors = np.clip(colors, 0.0, 1.0)
    
    # Set colors
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save to PCD file
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to {output_path}")
    
    return True