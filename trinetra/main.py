#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
import threading
import os
from services.vggt_parallel import VGGTParallel

def process_rtsp_streams(rtsp_urls, vggt_processor, resize_width=640, resize_height=360, target_fps=15):
    """Process RTSP streams and feed frames to VGGT processor."""
    # Initialize video captures for each stream
    caps = []
    for url in rtsp_urls:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream {url}")
            continue
        
        # Set properties to ensure 15 FPS capture
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        
        caps.append(cap)
    
    if len(caps) == 0:
        print("No streams could be opened")
        return
    
    # Batch counter
    batch_id = 0
    
    try:
        while True:
            # Capture timestamp at the beginning of frame collection
            current_timestamp = time.time()
            
            # Capture frames from all streams
            frames = []
            all_frames_captured = True
            
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    # Resize the frame to reduce dimensionality
                    frame = cv2.resize(frame, (resize_width, resize_height))
                    height, width = frame.shape[:2]
                    frames.append(frame)
                else:
                    print(f"Could not read from stream {i+1}, reconnecting...")
                    cap.release()
                    cap = cv2.VideoCapture(rtsp_urls[i], cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_FPS, target_fps)  # Set FPS again after reconnection
                    caps[i] = cap
                    all_frames_captured = False
                    break  # Break early if any frame capture fails
            
            # Only process if we have frames from all streams
            if all_frames_captured and len(frames) == len(caps):
                # Process frames through VGGT
                batch_id += 1
                success = vggt_processor.process_frame_set(frames, timestamp=current_timestamp, batch_id=batch_id)
                if success:
                    print(f"Queued batch {batch_id} with timestamp {current_timestamp:.3f}")
                else:
                    print(f"Failed to queue batch {batch_id}")
            
            # Adjust timing to maintain target FPS
            target_frame_time = 1.0/target_fps  # seconds per frame
            elapsed = time.time() - current_timestamp
            sleep_time = max(0, target_frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Print actual FPS (for monitoring)
            actual_fps = 1.0 / (time.time() - current_timestamp) if elapsed > 0 else target_fps
            if batch_id % 30 == 0:  # Print every 30 frames to avoid console spam
                print(f"Actual FPS: {actual_fps:.2f}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        for cap in caps:
            cap.release()
        
        print("RTSP processing stopped")

def main():
    # RTSP stream URLs - using the exact URLs from your original code
    rtsp_urls = [
        'rtsp://admin:Zenteiq2mai!@192.168.0.65:554/h265',
        'rtsp://admin:Zenteiq1mai!@192.168.0.66:554/h265',
        'rtsp://admin:Zenteiq1mai!@192.168.0.64:554/h265'
    ]
    
    # Initialize VGGT processor with parallel workers and visualization enabled
    vggt_processor = VGGTParallel(
        num_workers=5,
        output_dir="./output_pcd",
        save_interval=1.0,  # Save point clouds every second
        conf_threshold=60.0,  # Confidence threshold percentile
        visualize=True  # Enable visualization
    )
    
    # Start the VGGT processor
    vggt_processor.start()
    
    try:
        # Start RTSP processing in a separate thread - with resize parameters and target FPS
        rtsp_thread = threading.Thread(
            target=process_rtsp_streams, 
            args=(rtsp_urls, vggt_processor, 640, 360, 5)  # 640x360 resolution, 15 FPS
        )
        rtsp_thread.daemon = True
        rtsp_thread.start()
        
        # Main thread monitoring loop
        try:
            while vggt_processor.running:
                # # Let the main thread sleep to avoid high CPU usage
                time.sleep(0.1)
                
                # Check for keyboard interrupt
                if not rtsp_thread.is_alive():
                    print("RTSP processing thread stopped unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        # Wait for RTSP thread to finish
        rtsp_thread.join(timeout=2.0)
        
    finally:
        # Stop the VGGT processor
        vggt_processor.stop()

if __name__ == "__main__":
    main()