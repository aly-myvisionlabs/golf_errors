import numpy as np
from scipy.interpolate import CubicSpline
import cv2
import os

def calculate_swing_angles(keypoints):
    """
    Calculate swing angles from keypoints data.
    
    Args:
        keypoints (numpy.ndarray): Array of keypoints with shape (num_frames, num_keypoints, 3)
        
    Returns:
        numpy.ndarray: Array of angles for each frame
    """
    angles = []
    angles_del = []
    last_angle = 0
    
    for i in range(len(keypoints)):
        coords = keypoints[i]
        shoulder_coords = coords[5:7].copy()  # Shoulder keypoints
        hands = coords[9:11].copy()  # Hand keypoints
        
        # Check if shoulder and hand detections are reliable
        shoulder_yes = (shoulder_coords[0,2] > 0.7) and (shoulder_coords[1,2] > 0.7)
        hands_yes = (hands[0,2] > 0.7) and (hands[1,2] > 0.7)
        
        angle_here = 'NULL'
        if hands_yes and shoulder_yes:
            # Calculate mid points
            x_shoulder_mid = (shoulder_coords[0,0] + shoulder_coords[1,0]) / 2.
            y_shoulder_mid = (shoulder_coords[0,1] + shoulder_coords[1,1]) / 2.
            x_hands_mid = (hands[0,0] + hands[1,0]) / 2.
            y_hands_mid = (hands[0,1] + hands[1,1]) / 2.
            
            # Calculate angle
            angle_here = np.arctan2(-(y_hands_mid - y_shoulder_mid), 
                                  x_hands_mid - x_shoulder_mid)
            
            # Handle angle wrapping
            if i < 450:
                try:
                    if angle_here > 2:
                        angle_here -= 2 * np.pi
                except:
                    pass
        
        # Handle first frame
        if i == 0:
            angles_del.append(0)
            if angle_here == 'NULL':
                angles.append(angle_here)
            else:
                angles.append(float(angle_here))
        else:
            if angle_here == 'NULL':
                angles.append(angle_here)
                angles_del.append(0)
            else:
                if last_angle == 0:
                    last_angle = float(angle_here)
                    del_here = 0
                else:
                    del_here = float(angle_here) - last_angle
                    if (abs(del_here) > np.pi) and del_here > 0:
                        del_here = float(angle_here) - (last_angle + 2*np.pi)
                    elif (abs(del_here) > np.pi) and del_here < 0:
                        del_here = float(angle_here) - (last_angle - 2 * np.pi)
                    last_angle = float(angle_here)
                angles_del.append(del_here)
                angles.append(float(angle_here))
    
    return np.array(angles), np.array(angles_del)

def identify_swing_phases(keypoints, video_path=None, output_path=None):
    """
    Identify golf swing phases from keypoints data.
    
    Args:
        keypoints (numpy.ndarray): Array of keypoints with shape (num_frames, num_keypoints, 3)
        video_path (str, optional): Path to the source video file. Required if output_path is provided.
        output_path (str, optional): Path to save the phase frames. If None, frames won't be saved.
        
    Returns:
        dict: Dictionary mapping phase names to frame numbers
    """
    # Calculate angles
    angles, angles_del = calculate_swing_angles(keypoints)
    angles_sum = np.cumsum(angles_del)
    
    # Get valid angles (non-NULL)
    inds = np.arange(len(angles))
    keep = np.array(angles) != 'NULL'
    angles_keep = np.array(angles)[keep].astype(float)
    inds_keep = inds[keep]
    angles_sum_keep = angles_sum[keep]
    
    # Interpolate angles
    spl = CubicSpline(inds_keep, angles_sum_keep)
    new_x = np.arange(len(angles))
    new_y = np.array([spl(new_x[i]) for i in range(len(new_x))]) - np.pi/2.
    angles_interp = new_y.copy()
    
    # Define phases and their target angles
    phases = ['address', 'early_backswing', 'takeaway', 'half_backswing', 'top_backswing',
              'mid_downswing', 'low_downswing', 'prior_impact', 'impact',
              'early_follow_through', 'early_mid_follow through', 'mid_follow_through',
              'top_follow_through', 'finish']

    
    # Get top and finish angles
    top_frame = np.argmin(angles_interp)
    finish_frame = np.argmax(angles_interp)
    top_angle = np.amin(angles_interp)
    finish_angle = np.amax(angles_interp)
    
    # Define target angles for each phase
    seq = [-np.pi/2., -5*np.pi/8, -3*np.pi/4., -np.pi, top_angle,
           -np.pi, -3*np.pi/4., -4.5*np.pi/8, -np.pi/2.,
           -3*np.pi/8., -np.pi/4., 0, np.pi/4., finish_angle]
    
    # Find frames matching each phase
    phase_frames = {}
    seq_inds = []
    seq_angles = []
    
    for i, angle in enumerate(angles_interp):
        if len(seq_angles) == len(seq):
            break
        if i > 150:  # Skip initial frames
            target_angle = seq[len(seq_inds)]
            delta = target_angle - angle
            
            # Different matching criteria for different phases
            if len(seq_inds) < 4:  # Backswing phases
                if (np.abs(delta) < np.pi/32.) or (angle < target_angle):
                    seq_inds.append(i)
                    seq_angles.append(angle)
            elif len(seq_inds) == 4:  # Top of swing
                if (np.abs(delta) < np.pi/32.):
                    seq_inds.append(i)
                    seq_angles.append(angle)
            else:  # Downswing and follow-through phases
                if (np.abs(delta) < np.pi/32.) or (angle > target_angle):
                    seq_inds.append(i)
                    seq_angles.append(angle)
    
    # Create phase dictionary
    if len(seq_inds) == len(phases):
        phase_frames = {phase: frame for phase, frame in zip(phases, seq_inds)}
        
        # Save frames if output path is provided
        if output_path is not None:
            if video_path is None:
                raise ValueError("video_path must be provided when output_path is specified")
                
            # Create output directory if it doesn't exist
            print(f"Creating output directory: {output_path}")
            os.makedirs(output_path, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            try:
                frame_num = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_num in seq_inds:
                        phase_name = phases[seq_inds.index(frame_num)]
                        # Add phase name text to frame
                        cv2.putText(frame, phase_name, (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # Save frame
                        output_file = os.path.join(output_path, f"{frame_num}.png")
                        cv2.imwrite(output_file, frame)
                    
                    frame_num += 1
            finally:
                cap.release()
    
    return phase_frames

if __name__ == "__main__":
    # Example usage
    from extract_poses import extract_video_keypoints
    
    video_path = "path/to/your/video.mp4"
    output_path = "path/to/output/frames"  # Optional: set to None if you don't want to save frames
    
    try:
        # Extract keypoints
        keypoints = extract_video_keypoints(video_path)
        
        # Identify phases and optionally save frames
        phases = identify_swing_phases(keypoints, video_path=video_path, output_path=output_path)
        
        # Print results
        print("Swing phases and their frame numbers:")
        for phase, frame in phases.items():
            print(f"{phase}: frame {frame}")
            
    except Exception as e:
        print(f"Error analyzing video: {e}") 