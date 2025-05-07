import os
import glob
import re
import time
import json
from confluent_kafka import Producer

def natural_sort_key(s):
    """Function to ensure numerical sorting of filenames"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def delivery_report(err, msg):
    """Delivery callback for Kafka producer"""
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        pass  # Successful delivery

def produce_pcd_files(folder_path, kafka_topic, bootstrap_servers='localhost:9092'):
    """
    Reads PCD files from a folder and publishes them to a Kafka topic
    
    Args:
        folder_path (str): Path to the folder containing PCD files
        kafka_topic (str): Kafka topic to publish to
        bootstrap_servers (str): Kafka bootstrap servers
    """
    # Find all PCD files in the folder
    pcd_files = glob.glob(os.path.join(folder_path, "*.pcd"))
    
    # Sort the files to ensure they're processed in the correct order
    pcd_files.sort(key=natural_sort_key)
    
    if not pcd_files:
        print(f"No PCD files found in {folder_path}")
        return
    
    print(f"Found {len(pcd_files)} PCD files. Starting to produce messages...")
    
    # Create Confluent Kafka producer with increased message size limits
    conf = {
        'bootstrap.servers': bootstrap_servers,
        'client.id': 'pcd_producer',
        'message.max.bytes': 314572800  # 300MB limit
    }
    producer = Producer(conf)
    
    # Process each PCD file
    for i, pcd_file in enumerate(pcd_files):
        try:
            # Read the PCD file as binary
            with open(pcd_file, 'rb') as f:
                pcd_data = f.read()
            
            # Create a message with the file path and hex encoded data
            message = {
                'file_path': pcd_file,
                'data': pcd_data.hex(),  # Convert binary to hex string for JSON serialization
                'timestamp': time.time(),
                'frame_number': i
            }
            
            # Convert message to JSON string
            json_payload = json.dumps(message).encode('utf-8')
            
            # Send the message to Kafka
            producer.produce(
                kafka_topic, 
                value=json_payload,
                callback=delivery_report
            )
            print(f"Sent frame {i+1}/{len(pcd_files)}: {os.path.basename(pcd_file)}")
            
            # Force the producer to send the message immediately
            producer.poll(0)
            
            # Small delay to control flow rate if needed
            time.sleep(0.01)  # Adjust as needed for your visualization speed
            
        except Exception as e:
            print(f"Error processing {pcd_file}: {e}")
    
    # Wait for any outstanding messages to be delivered
    producer.flush()
    print(f"Finished producing {len(pcd_files)} PCD files to topic '{kafka_topic}'")

if __name__ == "__main__":
    # Change this to your folder path containing the PCD files
    folder_path = "output_pcd/"
    
    # Kafka topic name
    kafka_topic = "pcd_ingest"
    
    # Start producing PCD files to Kafka
    produce_pcd_files(folder_path, kafka_topic)