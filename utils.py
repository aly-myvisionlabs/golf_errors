import numpy as np

BODY_KEYPOINTS= {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
# Utility functions for vector operations
def get_vector(point1, point2):
    try:
        # Check if points are valid
        if len(point1) < 2 or len(point2) < 2:
            print("Warning: Invalid points for vector calculation")
            return [0, 0]
        
        # Check for NaN or None values
        if any(isinstance(v, (float, int)) and np.isnan(v) for v in point1 + point2):
            print("Warning: NaN values in points for vector calculation")
            return [0, 0]
        
        return [point2[0] - point1[0], point2[1] - point1[1]]
    except Exception as e:
        print(f"Error in get_vector: {str(e)}")
        return [0, 0]

def get_midpoint(point1, point2):
    return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]

def get_spine_vector(keypoints):
    """Get vector from hips to shoulders (spine direction)"""
    try:
        # Check if keypoints array is valid
        if len(keypoints) < 13:  # Need indices 5, 6, 11, 12 for shoulders and hips
            print("Warning: Not enough keypoints for spine vector calculation")
            return [0, 1]  # Default vertical vector
            
        # Get midpoint between hips
        hip_mid = get_midpoint(keypoints[11], keypoints[12])
        
        # Get midpoint between shoulders
        shoulder_mid = get_midpoint(keypoints[5], keypoints[6])
        
        # Create vector from hips to shoulders
        return get_vector(hip_mid, shoulder_mid)
    except Exception as e:
        print(f"Error in get_spine_vector: {str(e)}")
        return [0, 1]  # Default vertical vector

def normalize_vector(vector):
    # Avoid division by zero
    magnitude = np.sqrt(vector[0]**2 + vector[1]**2)
    if magnitude == 0:
        return [0, 0]
    return [vector[0] / magnitude, vector[1] / magnitude]

def dot_product(vector1, vector2):
    return vector1[0] * vector2[0] + vector1[1] * vector2[1]

def vector_angle(vector1, vector2):
    # Return angle in degrees between two vectors
    dot = dot_product(vector1, vector2)
    # Clamp dot to [-1, 1] to avoid numerical errors
    dot = max(min(dot, 1.0), -1.0)
    return np.degrees(np.arccos(dot))

def smooth_keypoints(keypoints_sequence, window_size=5):
    """
    Apply a moving average filter to smooth keypoints across frames.
    
    Args:
        keypoints_sequence: List of keypoint arrays
        window_size: Size of the smoothing window
    
    Returns:
        Smoothed keypoints sequence
    """
    if not keypoints_sequence or len(keypoints_sequence) < 3:
        return keypoints_sequence  # Not enough frames to smooth
    
    num_frames = len(keypoints_sequence)
    num_keypoints = keypoints_sequence[0].shape[0]
    num_coords = keypoints_sequence[0].shape[1]  # Usually 2 (x, y)
    
    # Create a copy of the original sequence to avoid modifying it
    smoothed_sequence = [np.copy(frame) for frame in keypoints_sequence]
    
    # Apply moving average filter to each keypoint coordinate
    half_window = window_size // 2
    
    for i in range(num_frames):
        for j in range(num_keypoints):
            for k in range(num_coords):
                # Define window bounds
                start_idx = max(0, i - half_window)
                end_idx = min(num_frames, i + half_window + 1)
                
                # Collect valid values within the window
                values = []
                weights = []
                
                for t in range(start_idx, end_idx):
                    # Check if this keypoint is valid (non-zero)
                    if np.sum(np.abs(keypoints_sequence[t][j])) > 0:
                        # Add the value with a weight based on distance from the center frame
                        distance = abs(t - i)
                        weight = 1.0 / (distance + 1)  # Higher weight for closer frames
                        values.append(keypoints_sequence[t][j][k] * weight)
                        weights.append(weight)
                
                # Only update if we have valid values
                if values:
                    smoothed_sequence[i][j][k] = sum(values) / sum(weights)
    
    return smoothed_sequence

def interpolate_missing_keypoints(keypoints_sequence, max_gap=10):
    """
    Interpolate missing keypoints if they're missing for max_gap frames or fewer.
    
    Args:
        keypoints_sequence: List of keypoint arrays
        max_gap: Maximum number of consecutive frames to interpolate
    
    Returns:
        Keypoints sequence with interpolated values
    """
    if not keypoints_sequence or len(keypoints_sequence) < 3:
        return keypoints_sequence  # Not enough frames to interpolate
    
    num_frames = len(keypoints_sequence)
    num_keypoints = keypoints_sequence[0].shape[0]
    
    # Create a copy of the original sequence to avoid modifying it
    interpolated_sequence = [np.copy(frame) for frame in keypoints_sequence]
    
    # For each keypoint
    for kp_idx in range(num_keypoints):
        # Find runs of missing keypoints
        i = 0
        while i < num_frames:
            # If current keypoint is missing (marked by zeros)
            if np.sum(np.abs(keypoints_sequence[i][kp_idx])) == 0:
                # Find the end of this run of missing keypoints
                start_missing = i
                while i < num_frames and np.sum(np.abs(keypoints_sequence[i][kp_idx])) == 0:
                    i += 1
                end_missing = i
                
                # Calculate the gap size
                gap_size = end_missing - start_missing
                
                # Only interpolate if the gap is not too large and we have valid keypoints before and after
                if gap_size <= max_gap and start_missing > 0 and end_missing < num_frames:
                    # Get the valid keypoints before and after the gap
                    before_kp = keypoints_sequence[start_missing - 1][kp_idx]
                    after_kp = keypoints_sequence[end_missing][kp_idx]
                    
                    # Interpolate linearly between them
                    for j in range(start_missing, end_missing):
                        alpha = (j - start_missing + 1) / (gap_size + 1)
                        interpolated_sequence[j][kp_idx] = before_kp * (1 - alpha) + after_kp * alpha
            else:
                i += 1
    
    return interpolated_sequence