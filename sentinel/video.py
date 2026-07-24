import cv2
import os

class VideoReader:
    """
    A utility class to handle loading, reading, and querying video files using OpenCV,
    supporting frame skipping and random-access seeking.
    """
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found at: {path}")
        
        self.path = path
        self.cap = cv2.VideoCapture(path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {path}")
            
        # Cache video metadata
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.frame_count / self.fps if self.fps > 0 else 0.0

    def get_frame(self, frame_idx: int):
        """
        Retrieve a specific frame by its 0-based index.
        Returns the frame (numpy array) or None if index is out of bounds or read fails.
        """
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = self.cap.read()
        if success:
            return frame
        return None

    def get_frame_at_time(self, seconds: float):
        """
        Retrieve a frame at a specific timestamp in seconds.
        """
        frame_idx = int(seconds * self.fps)
        return self.get_frame(frame_idx)

    def iter_frames(self, step: int = 1):
        """
        A generator that yields (frame_index, frame) from the video.
        
        Args:
            step: Skip frames to process every `step` frame. 
                  e.g., step=5 processes frame 0, 5, 10, etc.
        """
        if step < 1:
            raise ValueError("Step size must be at least 1.")
            
        # Reset reader to the start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
                
            if frame_idx % step == 0:
                yield frame_idx, frame
                
            frame_idx += 1

    def release(self):
        """Release the video capture object."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
