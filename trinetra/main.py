import cv2
import threading
import time
import numpy as np
from services.rtsp_client import RTSPClient
# List of RTSP stream URLs
if __name__ == "__main__":
    # Example stream URLs
    stream_urls = [
        'rtsp://admin:Zenteiq2mai!@192.168.0.65:554/h265',
        'rtsp://admin:Zenteiq1mai!@192.168.0.66:554/h265',
        'rtsp://admin:Zenteiq1mai!@192.168.0.64:554/h265'
    ]
    
    # Create and start the client
    client = RTSPClient(stream_urls)
    client.start()
    
    try:
        # Keep the main thread alive
        while client.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Stop client on Ctrl+C
        client.stop()
    
    print("Program terminated")