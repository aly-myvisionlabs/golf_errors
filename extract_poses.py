from ultralytics import YOLO
import numpy as np
import cv2

def extract_video_keypoints(video_path, model_path=r"C:\Users\opsiclear_user\projects\golf_errors\models\yolo11x-pose.pt"):
    """
    Extract keypoints from a video using YOLO pose estimation.
    
    Args:
        video_path (str): Path to the input video file
        model_path (str): Path to the YOLO model weights file
        
    Returns:
        numpy.ndarray: Array of keypoints with shape (num_frames, num_keypoints, 3)
                      where 3 represents (x, y, confidence) for each keypoint
    """
    # Load YOLO model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # List to store keypoints for each frame
    keypoints_list = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get pose estimation results
            results = model(frame, verbose=False)[0]
            kpts = results.keypoints.data
            kpts = kpts.to('cpu').numpy()
            
            # Handle multiple detections by selecting the one with highest confidence
            if len(kpts) > 1:
                kpts_probs = kpts[:,:,-1]
                best_idx = np.argmax(np.sum(kpts_probs, axis=1))
                keypoints_list.append(kpts[best_idx])
            else:
                keypoints_list.append(kpts[0])
                
    finally:
        cap.release()
    
    # Stack all keypoints into a single array
    keypoints_array = np.stack(keypoints_list, axis=0)
    
    return keypoints_array

if __name__ == "__main__":
    # Example usage
    video_path = "path/to/your/video.mp4"
    try:
        keypoints = extract_video_keypoints(video_path)
        print(f"Extracted keypoints shape: {keypoints.shape}")
    except Exception as e:
        print(f"Error processing video: {e}") 