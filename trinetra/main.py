#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
from services.vggt_parallel import VGGTParallel

def process_rtsp_streams(rtsp_urls, vggt_processor):
    """
    Capture frames from RTSP streams and feed them to the VGGT processor.
    
    Args:
        rtsp_urls: List of RTSP stream URLs
        vggt_processor: Initialized VGGTParallel instance
    """
    # Initialize video captures for each stream
    caps = []
    for url in rtsp_urls:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream {url}")
            continue
        caps.append(cap)
    
    if len(caps) == 0:
        print("No streams could be opened")
        return
    
    print(f"Successfully opened {len(caps)} camera streams")
    
    # Batch counter
    batch_id = 0
    
    try:
        while True:
            # Capture frames from all streams
            frames = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    print(f"Could not read from stream {i+1}, reconnecting...")
                    cap.release()
                    cap = cv2.VideoCapture(rtsp_urls[i], cv2.CAP_FFMPEG)
                    caps[i] = cap
            
            # Only process if we have frames from all streams
            if len(frames) == len(caps):
                # Process frames through VGGT
                batch_id += 1
                print('The frames are', frames)
                vggt_processor.process_frame_set(frames, timestamp=time.time(), batch_id=batch_id)
                print(f"Processed batch {batch_id}")
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        for cap in caps:
            cap.release()
        
        print("RTSP processing stopped")

def main():
    # RTSP stream URLs
    rtsp_urls = [
        'rtsp://admin:Zenteiq2mai!@192.168.0.65:554/h265',
        'rtsp://admin:Zenteiq1mai!@192.168.0.66:554/h265',
        'rtsp://admin:Zenteiq1mai!@192.168.0.64:554/h265'
    ]
    
    # Initialize VGGT processor with 3 parallel workers
    vggt_processor = VGGTParallel(
        num_workers=4,
        output_dir="./output_pcd",
        save_interval=5.0,  # Save combined point cloud every 5 seconds
        conf_threshold=25.0  # Confidence threshold percentile
    )
    
    # Start the VGGT processor
    vggt_processor.start()
    
    try:
        # Process RTSP streams
        process_rtsp_streams(rtsp_urls, vggt_processor)
    finally:
        # Stop the VGGT processor and save the final point cloud
        vggt_processor.stop()

if __name__ == "__main__":
    main()