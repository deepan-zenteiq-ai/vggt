#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import threading
import time
import queue
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass

@dataclass
class TimestampedFrame:
    """
    Data class for storing a frame with its timestamp and stream ID.
    """
    frame: np.ndarray
    timestamp: float
    stream_id: int

class RTSPClient:
    """
    A modular RTSP client that handles multiple streams simultaneously
    using thread-safe queues to avoid H.264 decoding issues.
    """
    
    def __init__(self, stream_urls: List[str], queue_size: int = 300):
        """
        Initialize the RTSP client with multiple stream URLs.
        
        Args:
            stream_urls: List of RTSP stream URLs to connect to
            queue_size: Maximum size of the frame queue for each stream
        """
        self.stream_urls = stream_urls
        self.num_streams = len(stream_urls)
        self.running = False
        
        # Create a queue for each stream
        self.frame_queues = {}
        for i in range(self.num_streams):
            self.frame_queues[i] = queue.Queue(maxsize=queue_size)
        
        # Output queue for frames to be processed
        self.output_queue = queue.Queue(maxsize=queue_size * self.num_streams)
        
        # Thread references
        self.capture_threads = []
        self.display_thread = None
        
        # For frame rate calculation
        self.fps_counters = {}
        for i in range(self.num_streams):
            self.fps_counters[i] = {
                'last_time': time.time(),
                'count': 0,
                'fps': 0
            }
    
    def _capture_stream(self, stream_url: str, stream_id: int):
        """
        Thread function to capture frames from a stream and put them in the queue.
        
        Args:
            stream_url: RTSP URL to capture
            stream_id: Identifier for this stream
        """
        # Create a VideoCapture object with FFMPEG backend to handle RTSP
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        
        # Check if the connection was successful
        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream {stream_url}")
            return
        
        print(f"Successfully connected to {stream_url}")
        
        # Continuously capture frames while running
        while self.running:
            ret, frame = cap.read()
            
            # If frame was not retrieved, try to reconnect
            if not ret:
                print(f"Error: Could not read frame from {stream_url}, attempting to reconnect...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                continue
            
            # Create timestamped frame
            current_time = time.time()
            timestamped_frame = TimestampedFrame(
                frame=frame.copy(),
                timestamp=current_time,
                stream_id=stream_id
            )
            
            # If queue is full, remove oldest frame to prevent blocking
            if self.frame_queues[stream_id].full():
                try:
                    self.frame_queues[stream_id].get_nowait()
                except queue.Empty:
                    pass
            
            # Put the new frame in the stream queue
            try:
                self.frame_queues[stream_id].put(timestamped_frame, block=False)
            except queue.Full:
                pass  # Skip this frame if queue is still full
            
            # Put the frame in the output queue for processing
            try:
                if not self.output_queue.full():
                    self.output_queue.put(timestamped_frame, block=False)
                else:
                    # Remove oldest item if full
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put(timestamped_frame, block=False)
                    except queue.Empty:
                        pass
            except queue.Full:
                pass
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)
        
        # Release resources
        cap.release()
        print(f"Capture thread for {stream_url} terminated")
    
    def _display_streams(self):
        """
        Thread function to display all streams from their respective queues.
        This runs in a separate thread from the capture threads to avoid H.264 decoding issues.
        """
        # Create windows for each stream (all done in the same thread)
        for i in range(self.num_streams):
            cv2.namedWindow(f"Stream {i+1}", cv2.WINDOW_NORMAL)
        
        # Display loop
        while self.running:
            for i in range(self.num_streams):
                # Calculate FPS
                current_time = time.time()
                elapsed = current_time - self.fps_counters[i]['last_time']
                if elapsed > 1.0:  # Update FPS every second
                    self.fps_counters[i]['fps'] = self.fps_counters[i]['count'] / elapsed
                    self.fps_counters[i]['count'] = 0
                    self.fps_counters[i]['last_time'] = current_time
                
                # Check if we have a frame for this stream
                if not self.frame_queues[i].empty():
                    try:
                        frame_data = self.frame_queues[i].get_nowait()
                        display_frame = frame_data.frame.copy()
                        
                        # Add timestamp to the frame
                        timestamp_str = f"Time: {time.strftime('%H:%M:%S', time.localtime(frame_data.timestamp))}"
                        cv2.putText(display_frame, timestamp_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        # Add FPS info
                        fps_str = f"FPS: {self.fps_counters[i]['fps']:.1f}"
                        cv2.putText(display_frame, fps_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        # Display the frame
                        cv2.imshow(f"Stream {i+1}", display_frame)
                        
                        # Increment frame counter for FPS calculation
                        self.fps_counters[i]['count'] += 1
                        
                    except queue.Empty:
                        pass  # Queue became empty between check and get
            
            # Process GUI events and check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        # Cleanup
        cv2.destroyAllWindows()
        print("Display thread terminated")
    
    def get_frame(self, timeout: float = 0.5) -> Optional[TimestampedFrame]:
        """
        Get a frame from the output queue.
        
        Args:
            timeout: Time to wait for a frame (seconds)
            
        Returns:
            TimestampedFrame or None if queue is empty
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def start(self, display: bool = True):
        """
        Start capturing streams and optionally display them.
        
        Args:
            display: Whether to start the display thread
        """
        if self.running:
            print("RTSP client is already running")
            return
        
        self.running = True
        
        # Start capture threads
        for i, url in enumerate(self.stream_urls):
            thread = threading.Thread(target=self._capture_stream, args=(url, i))
            thread.daemon = True
            thread.start()
            self.capture_threads.append(thread)
        
        # Give some time for the capture threads to start
        time.sleep(1)
        
        # Start display thread if requested
        if display:
            self.display_thread = threading.Thread(target=self._display_streams)
            self.display_thread.daemon = True
            self.display_thread.start()
        
        print("RTSP client started")
    
    def stop(self):
        """
        Stop all threads and release resources.
        """
        if not self.running:
            print("RTSP client is not running")
            return
        
        self.running = False
        
        # Wait for threads to finish
        for thread in self.capture_threads:
            thread.join(timeout=1.0)
        
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        
        # Clear all queues
        for q in self.frame_queues.values():
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        
        self.capture_threads = []
        self.display_thread = None
        
        print("RTSP client stopped")

# Usage exa