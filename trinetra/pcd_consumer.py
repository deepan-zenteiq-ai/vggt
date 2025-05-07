import os
import numpy as np
import open3d as o3d
import time
import json
import binascii
from confluent_kafka import Consumer, KafkaError
import tempfile

def visualize_pcd_from_kafka(kafka_topic, bootstrap_servers='localhost:9092'):
    """
    Consumes PCD data from a Kafka topic and visualizes the point clouds in real-time
    
    Args:
        kafka_topic (str): Kafka topic to consume from
        bootstrap_servers (str): Kafka bootstrap servers
    """
    print(f"Starting consumer for topic '{kafka_topic}'...")
    
    # Create Confluent Kafka consumer
    conf = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': 'pcd_visualizer',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(conf)
    consumer.subscribe([kafka_topic])
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Kafka Point Cloud Viewer", width=1024, height=768)
    
    # Add options to the renderer
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background
    opt.point_size = 1.0
    
    # Create a placeholder for the current point cloud
    current_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(current_pcd)
    
    # For measuring frame rate
    start_time = time.time()
    frame_count = 0
    
    # Create a temporary directory for storing the PCD files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Process messages from Kafka
        while True:
            # Poll for messages
            msg = consumer.poll(1.0)
            
            if msg is None:
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    print(f"Reached end of partition {msg.partition()}")
                else:
                    print(f"Error: {msg.error()}")
            else:
                try:
                    # Parse the message value
                    pcd_data = json.loads(msg.value().decode('utf-8'))
                    frame_number = pcd_data.get('frame_number', frame_count)
                    
                    # Convert hex string back to binary
                    binary_data = binascii.unhexlify(pcd_data['data'])
                    
                    # Write to a temporary file
                    temp_file = os.path.join(temp_dir, f"temp_{frame_number}.pcd")
                    with open(temp_file, 'wb') as f:
                        f.write(binary_data)
                    
                    # Read the point cloud from the temporary file
                    pcd = o3d.io.read_point_cloud(temp_file)
                    
                    # Skip if point cloud is empty
                    if len(np.asarray(pcd.points)) == 0:
                        print(f"Skipping empty point cloud: frame {frame_number}")
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
                        print(f"Frame {frame_count} | FPS: {fps:.2f}")
                    
                    # Remove the temporary file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                    
                except Exception as e:
                    print(f"Error processing message: {e}")
                    
            # Check if the visualization window is still open
            if not vis.poll_events():
                break
    
    except KeyboardInterrupt:
        print("Visualization stopped by user")
    
    finally:
        # Clean up
        consumer.close()
        vis.destroy_window()
        
        # Try to remove the temporary directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
        
        # Final statistics
        if frame_count > 0:
            total_time = time.time() - start_time
            total_fps = frame_count / total_time
            print(f"Visualization complete. Average FPS: {total_fps:.2f}")

if __name__ == "__main__":
    # Kafka topic name
    kafka_topic = "pcd_ingest"
    
    # Start visualization from Kafka messages
    visualize_pcd_from_kafka(kafka_topic)