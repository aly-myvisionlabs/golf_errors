import numpy as np
from utils import smooth_keypoints, interpolate_missing_keypoints
from glob import glob
import os
import cv2

def load_pose_data(pose_file_path, smooth=True, interpolate=True):
    """Load pose keypoints sequence from a single .npy file"""
    try:
        pose_data = np.load(pose_file_path, allow_pickle=True)
        
        # Handle different possible formats
        if isinstance(pose_data, np.ndarray):
            # Check shape
            if len(pose_data.shape) == 3 and pose_data.shape[1] == 17:
                # Format: (N, 17, x) where N is number of frames, x is 2 or 3
                keypoints_sequence = [frame[:, :2] for frame in pose_data]
            elif len(pose_data.shape) == 4 and pose_data.shape[1] == 1 and pose_data.shape[2] == 17:
                # Format: (N, 1, 17, x) where N is number of frames, x is 2 or 3
                keypoints_sequence = [frame[0, :, :2] for frame in pose_data]
            else:
                # Unknown format, create dummy data
                print(f"Warning: Unrecognized pose data shape {pose_data.shape}. Creating dummy data.")
                keypoints_sequence = [np.zeros((17, 2))]
        else:
            print(f"Warning: Pose data is not a numpy array. Creating dummy data.")
            keypoints_sequence = [np.zeros((17, 2))]
        
        print(f"Loaded {len(keypoints_sequence)} pose frames")
        
        # Validate that we have valid keypoint data
        if not check_keypoints_validity(keypoints_sequence):
            print("Warning: Keypoint data appears invalid. Results may be inaccurate.")
        
        # Interpolate missing keypoints if requested
        if interpolate:
            keypoints_sequence = interpolate_missing_keypoints(keypoints_sequence)
            print("Applied interpolation to missing keypoints")
        
        # Smooth keypoints if requested
        if smooth:
            keypoints_sequence = smooth_keypoints(keypoints_sequence)
            print("Applied smoothing to keypoints")
            
        return keypoints_sequence
    except Exception as e:
        print(f"Error loading pose data from {pose_file_path}: {str(e)}")
        return [np.zeros((17, 2))]

def check_keypoints_validity(keypoints_sequence):
    """Check if the keypoints data is valid (has enough points and non-zero values)"""
    if not keypoints_sequence:
        return False
    
    # Check a sample of frames
    sample_size = min(10, len(keypoints_sequence))
    sample_indices = np.linspace(0, len(keypoints_sequence) - 1, sample_size, dtype=int)
    
    valid_frames = 0
    for idx in sample_indices:
        keypoints = keypoints_sequence[idx]
        # Check if keypoints has proper shape
        if keypoints.shape[0] >= 17 and keypoints.shape[1] >= 2:
            # Check if it has non-zero values
            if np.sum(np.abs(keypoints)) > 0:
                valid_frames += 1
    
    # Consider valid if at least 50% of sampled frames are valid
    return valid_frames >= sample_size * 0.5

def load_both_views_data(front_pose_file, dtl_pose_file=None):
    """Load pose data from both front and DTL views if available"""
    # Load front view data
    front_keypoints = load_pose_data(front_pose_file)
    
    # Load DTL view data if available
    dtl_keypoints = load_pose_data(dtl_pose_file) if dtl_pose_file else None
    
    return front_keypoints, dtl_keypoints

def map_swing_phases_to_events(phases_folder):
    """Map the swing phases to the events used in the error detection code"""
    swing_phase_files = sorted(glob(os.path.join(phases_folder, '*.png')))
    
    # Check if we have enough swing phase files
    if len(swing_phase_files) == 0:
        raise ValueError(f"No swing phase files found in {phases_folder}")
    
    print(f"Found {len(swing_phase_files)} swing phase files")
    
    # Get frame numbers from filenames
    frame_numbers = []
    for file in swing_phase_files:
        try:
            frame_number = int(os.path.basename(file).split('.')[0])
            frame_numbers.append(frame_number)
        except ValueError:
            # Handle invalid filenames
            print(f"Warning: Could not parse frame number from {os.path.basename(file)}. Skipping.")
    
    # Make sure we have at least one frame number
    if len(frame_numbers) == 0:
        raise ValueError(f"Could not parse any frame numbers from swing phase files.")
    
    # Map the swing phases to the events needed for error detection
    # If we have all 14 phases, use the standard mapping
    if len(frame_numbers) >= 14:
        events = {
            'address': frame_numbers[0],
            'takeaway': frame_numbers[2],
            'half_backswing': frame_numbers[3],
            'top_backswing': frame_numbers[4],
            'transition': frame_numbers[5],
            'mid_downswing': frame_numbers[6],
            'impact': frame_numbers[8],
            'follow_through': frame_numbers[9],
            'finish': frame_numbers[13]
        }
    else:
        # If we have fewer phases, create a simplified mapping
        print("Warning: Fewer than 14 swing phases found. Creating simplified mapping.")
        
        # Distribute available frames across key events
        n_frames = len(frame_numbers)
        
        if n_frames == 1:
            # Just use the single frame for all events
            single_frame = frame_numbers[0]
            events = {key: single_frame for key in ['address', 'takeaway', 'half_backswing', 
                                                  'top_backswing', 'transition', 'mid_downswing', 
                                                  'impact', 'follow_through', 'finish']}
        else:
            # Distribute frames evenly
            step = (n_frames - 1) / 8  # 8 intervals between 9 points
            
            # Compute indices by even distribution
            indices = [min(int(i * step), n_frames - 1) for i in range(9)]
            
            # Map to events
            event_keys = ['address', 'takeaway', 'half_backswing', 'top_backswing', 
                          'transition', 'mid_downswing', 'impact', 'follow_through', 'finish']
            events = {event_keys[i]: frame_numbers[indices[i]] for i in range(9)}
    
    print(f"Successfully mapped {len(swing_phase_files)} swing phases to {len(events)} events")
    print(f"Event frames: {events}")
    
    return events

def define_key_frames(keypoints_sequence, events):
    """Extract the key frame indices from events dictionary"""
    address = events.get('address', 0)
    takeaway = events.get('takeaway', 0)
    half_backswing = events.get('half_backswing', 0)
    top_backswing = events.get('top_backswing', 0)
    transition = events.get('transition', 0)
    mid_downswing = events.get('mid_downswing', 0)
    impact = events.get('impact', 0)
    follow_through = events.get('follow_through', 0)
    finish = events.get('finish', 0)
    
    return address, takeaway, half_backswing, top_backswing, transition, mid_downswing, impact, follow_through, finish


def load_frames_from_video(video_path, start_frame, end_frame):
    """Load frames from a video file between start and end frames"""
    if not video_path or not os.path.exists(video_path):
        print(f"Warning: Video file {video_path} not found")
        return []
    
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check if start and end frames are valid
    if start_frame >= total_frames or end_frame >= total_frames:
        print(f"Warning: Requested frames ({start_frame} to {end_frame}) exceed video length ({total_frames})")
        end_frame = min(end_frame, total_frames - 1)
        start_frame = min(start_frame, end_frame)
    
    # Extract the requested frames
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Failed to read frame {frame_idx} from video")
            break
    
    cap.release()
    return frames
