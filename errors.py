import os
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm  # For progress bars
from utils import get_vector, get_midpoint, get_spine_vector, normalize_vector, interpolate_missing_keypoints, smooth_keypoints, BODY_KEYPOINTS
from golf_parser import load_pose_data, load_both_views_data, map_swing_phases_to_events, define_key_frames
from extract_masks import extract_masks_from_points

# Constants for error thresholds
SWAY_THRESHOLD = 3  # Pixels for sway detection
SLIDE_THRESHOLD = 2  # Pixels for slide detection
POSTURE_THRESHOLD = 10  # Degrees for posture angle changes
HEAD_EXTENSION_THRESHOLD = 20  # Pixels for early extension detection
HIP_EXTENSION_THRESHOLD = 15  # Pixels for early extension detection



def process_swing_data(front_keypoints, dtl_keypoints, events, front_video_path, dtl_video_path, output_dir, is_right_handed=True):
    """Process the pose data and swing phases to detect errors."""
    
    try:
        
        # Make sure we have valid pose data
        if len(front_keypoints) == 0:
            print("Error: No pose frames were loaded")
            return {"error": "No pose frames were loaded"}
        
        
        address_frame = events["address"]
        dtl_address_keypoints = dtl_keypoints[address_frame]
        front_address_keypoints = front_keypoints[address_frame]

        # Filter out keypoints with [0,0] coordinates (undetected points)
        valid_dtl_keypoints = [kp for kp in dtl_address_keypoints if not (kp[0] == 0 and kp[1] == 0)]
        valid_front_keypoints = [kp for kp in front_address_keypoints if not (kp[0] == 0 and kp[1] == 0)]

        # Initialize masks variables
        masks_dtl = None
        masks_front = None
        # Check if we have at least one valid keypoint
        if len(valid_dtl_keypoints) > 0:
            print(f"Using {len(valid_dtl_keypoints)} valid keypoints out of {len(dtl_address_keypoints)} total keypoints")
            masks_dtl = extract_masks_from_points(
                video_path=dtl_video_path,
                points=valid_dtl_keypoints,  # Use filtered keypoints
                frame_idx=address_frame,
                output_dir=os.path.join(output_dir, "masks", "dtl")
            )

        if len(valid_front_keypoints) > 0:
            print(f"Using {len(valid_front_keypoints)} valid keypoints out of {len(front_address_keypoints)} total keypoints")
            masks_front = extract_masks_from_points(
                video_path=front_video_path,
                points=valid_front_keypoints,  # Use filtered keypoints
                frame_idx=address_frame,
                output_dir=os.path.join(output_dir, "masks", "front")
            )
        
        # Detect errors
        print("Running error detection...")
        errors = detect_all_errors_from_data(front_keypoints, dtl_keypoints, events, masks_front, masks_dtl, is_right_handed)

        return errors, masks_front, masks_dtl
        
    except Exception as e:
        print(f"Error processing swing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, None

def detect_swaying(keypoints_sequence, events, masks=None, is_right_handed=True):
    """
    Detect swaying using keypoints.
    Sway occurs when the trail hip moves outward during backswing.
    """
    try:
        address = events["address"]
        top_backswing = events["top_backswing"]
        
        # Get trail side foot and hip indices based on handedness
        if is_right_handed:
            foot_idx = BODY_KEYPOINTS["right_ankle"]  # Right ankle (trail side for right-handed)
            hip_idx = BODY_KEYPOINTS["right_hip"]   # Right hip
        else:
            foot_idx = BODY_KEYPOINTS["left_ankle"]  # Left ankle (trail side for left-handed)
            hip_idx = BODY_KEYPOINTS["left_hip"]   # Left hip
            
        # Get reference points at address
        address_keypoints = keypoints_sequence[address]
        trail_foot_x = address_keypoints[foot_idx][0]
        trail_foot_y = address_keypoints[foot_idx][1]
        trail_hip_x = address_keypoints[hip_idx][0]
        trail_hip_y = address_keypoints[hip_idx][1]
        
        # Store mask edges if available (for visualization only)
        mask_foot_x = trail_foot_x
        mask_hip_x = trail_hip_x
        
        # Get mask edges if available (for visualization only)
        if masks is not None and address in masks:
            address_mask = masks[address]
            if address_mask is not None:
                foot_y = int(trail_foot_y)
                hip_y = int(trail_hip_y)
                
                # Find foot edge from mask (for visualization)
                if 0 <= foot_y < address_mask.shape[0]:
                    foot_rows = address_mask[foot_y:]
                    edge_x = None
                    
                    # Scan each row from foot_y downward
                    for row_idx, row in enumerate(foot_rows):
                        non_zero_indices = np.where(row > 0)[0]
                        if len(non_zero_indices) > 0:
                            if is_right_handed:
                                # For right-handed golfers: leftmost point in image
                                current_edge = non_zero_indices[0]
                            else:
                                # For left-handed golfers: rightmost point in image
                                current_edge = non_zero_indices[-1]
                            
                            # Update edge_x if this is the first edge found or it's more extreme than current edge
                            if edge_x is None:
                                original_edge_x = current_edge
                                edge_x = current_edge
                                edge_y = row_idx
                            elif (is_right_handed and current_edge < edge_x) or (not is_right_handed and current_edge > edge_x):
                                edge_x = current_edge
                                edge_y = row_idx
                    
                    # Use the found edge if any
                    if edge_x is not None:
                        hip_x_diff = abs(original_edge_x - edge_x)
                        mask_foot_x = edge_x
                        foot_y = foot_y + edge_y
                # Find hip edge from mask (for visualization)
                if 0 <= hip_y < address_mask.shape[0]:
                    hip_row = address_mask[hip_y]
                    non_zero_indices = np.where(hip_row > 0)[0]
                    if len(non_zero_indices) > 0:
                        if is_right_handed:
                            mask_hip_x = non_zero_indices[0] - hip_x_diff  # Leftmost point
                        else:
                            mask_hip_x = non_zero_indices[-1] + hip_x_diff  # Rightmost point
        
        # Track maximum sway distance
        max_sway_distance = 0
        max_sway_frame = address
        
        frame_data = []
        is_sway_detected = False
        
        # Check all frames between address and top backswing
        for frame in range(address, top_backswing + 1):
            current_keypoints = keypoints_sequence[frame]
            current_hip_x = current_keypoints[hip_idx][0]
            current_hip_y = current_keypoints[hip_idx][1]
            
            # Calculate sway distance using keypoints - sway is when hip moves outward
            if is_right_handed:
                # For right-handed, sway is when hip_x > address_hip_x (moving right)
                sway_distance = current_hip_x - trail_hip_x
                sway_detected = sway_distance > SWAY_THRESHOLD
            else:
                # For left-handed, sway is when hip_x < address_hip_x (moving left)
                sway_distance = trail_hip_x - current_hip_x
                sway_detected = sway_distance > SWAY_THRESHOLD
            
            # Check if this frame has more sway than previous max
            if sway_distance > max_sway_distance:
                max_sway_distance = sway_distance
                max_sway_frame = frame
            
            # Record if sway is detected
            if sway_detected:
                is_sway_detected = True
            
            # Get current mask edge for visualization (if available)
            current_mask_hip_x = current_hip_x
            if masks is not None and frame in masks:
                current_mask = masks[frame]
                if current_mask is not None:
                    hip_y_int = int(current_hip_y)
                    if 0 <= hip_y_int < current_mask.shape[0]:
                        hip_row = current_mask[hip_y_int]
                        non_zero_indices = np.where(hip_row > 0)[0]
                        if len(non_zero_indices) > 0:
                            if is_right_handed:
                                current_mask_hip_x = non_zero_indices[0]  # Leftmost point
                            else:
                                current_mask_hip_x = non_zero_indices[-1]  # Rightmost point
            
            # Store data for this frame
            frame_data.append({
                'frame': frame,
                'foot_x': trail_foot_x,
                'hip_x': current_hip_x,
                'address_hip_x': trail_hip_x,
                'mask_foot_x': mask_foot_x,
                'mask_hip_x': mask_hip_x,
                'current_mask_hip_x': current_mask_hip_x,
                'sway_distance': sway_distance,
                'sway_detected': sway_detected,
                'hip_y': current_hip_y
            })
                
        # Return detailed metrics
        return {
            'result': is_sway_detected,
            'threshold': SWAY_THRESHOLD,
            'max_sway_distance': max_sway_distance,
            'max_sway_frame': max_sway_frame,
            'trail_foot_x': trail_foot_x,
            'trail_hip_x': trail_hip_x,
            'mask_foot_x': mask_foot_x,
            'mask_hip_x': mask_hip_x,
            'foot_idx': foot_idx,
            'hip_idx': hip_idx,
            'frame_data': frame_data,
            'address': address,
            'top_backswing': top_backswing
        }
    except Exception as e:
        print(f"Error in detect_swaying: {str(e)}")
        return {'result': False, 'error': str(e)}

def detect_sliding(keypoints_sequence, events, masks=None, is_right_handed=True):
    """
    Detect sliding and return detailed metrics.
    """
    try:
        address = events["address"]
        top_backswing = events["top_backswing"]
        impact = events["impact"]
        
        # Get lead side foot and hip indices based on handedness
        if is_right_handed:
            foot_idx = BODY_KEYPOINTS["left_ankle"]  # Left ankle (lead side for right-handed)
            hip_idx = BODY_KEYPOINTS["left_hip"]   # Left hip
        else:
            foot_idx = BODY_KEYPOINTS["right_ankle"]  # Right ankle (lead side for left-handed)
            hip_idx = BODY_KEYPOINTS["right_hip"]   # Right hip
        
        # Get reference points at address for visualization
        address_keypoints = keypoints_sequence[address]
        address_foot_x = address_keypoints[foot_idx][0]
        address_foot_y = address_keypoints[foot_idx][1]
        address_hip_y = address_keypoints[hip_idx][1]
        
        # Store mask edges at address if available (for visualization only)
        mask_foot_x = address_foot_x
        
        # Get mask edge at address if available (for visualization only)
        if masks is not None and address in masks:
            address_mask = masks[address]
            if address_mask is not None:
                foot_y = int(address_foot_y)
                
                # Find foot edge from mask (for visualization)
                if 0 <= foot_y < address_mask.shape[0]:
                    foot_rows = address_mask[foot_y:]
                    edge_x = None
                    
                    # Scan each row from foot_y downward
                    for row_idx, row in enumerate(foot_rows):
                        non_zero_indices = np.where(row > 0)[0]
                        if len(non_zero_indices) > 0:
                            if is_right_handed:
                                # For right-handed golfers: rightmost point (largest x) for left foot
                                current_edge = non_zero_indices[-1]
                            else:
                                # For left-handed golfers: leftmost point (smallest x) for right foot
                                current_edge = non_zero_indices[0]
                            
                            # Update edge_x if this is the first edge found or it's more extreme than current edge
                            if edge_x is None:
                                edge_x = current_edge
                                edge_y = row_idx
                            elif (is_right_handed and current_edge > edge_x) or (not is_right_handed and current_edge < edge_x):
                                edge_x = current_edge
                                edge_y = row_idx
                    
                    # Use the found edge if any
                    if edge_x is not None:
                        mask_foot_x = edge_x
                        foot_y = foot_y + edge_y
        # Get reference points at top backswing for detection
        top_keypoints = keypoints_sequence[top_backswing]
        lead_foot_x = top_keypoints[foot_idx][0]
        
        # Track maximum slide distance
        max_slide_distance = 0
        max_slide_frame = top_backswing
        
        frame_data = []
        is_slide_detected = False
        
        # Check all frames between top backswing and impact
        for frame in range(top_backswing, impact + 1):
            current_keypoints = keypoints_sequence[frame]
            current_hip_y = int(current_keypoints[hip_idx][1])
            # Find hip edge from mask (for visualization)
            if 0 <= current_hip_y < address_mask.shape[0]:
                hip_row = address_mask[current_hip_y]
                non_zero_indices = np.where(hip_row > 0)[0]
                if len(non_zero_indices) > 0:
                    if is_right_handed:
                        current_hip_x = non_zero_indices[-1]   # Leftmost point
                    else:
                        current_hip_x = non_zero_indices[0]  # Rightmost point
            # Calculate slide distance based on handedness
            if is_right_handed:
                slide_distance = current_hip_x - mask_foot_x
            else:
                slide_distance = mask_foot_x - current_hip_x
            
            # Check if this frame has more slide than previous max
            if slide_distance > max_slide_distance:
                max_slide_distance = slide_distance
                max_slide_frame = frame
            
            # Check if slide is detected for this frame
            frame_slide_detected = slide_distance > SLIDE_THRESHOLD
            if frame_slide_detected:
                is_slide_detected = True
            
            # Store data for this frame
            frame_data.append({
                'frame': frame,
                'foot_x': lead_foot_x,
                'hip_x': current_hip_x,
                'slide_distance': slide_distance,
                'slide_detected': frame_slide_detected
            })
                
        # Return detailed metrics
        return {
            'result': is_slide_detected,
            'threshold': SLIDE_THRESHOLD,
            'max_slide_distance': max_slide_distance,
            'max_slide_frame': max_slide_frame,
            'lead_foot_x': lead_foot_x,
            'mask_foot_x': mask_foot_x,
            'address_foot_y': address_foot_y,
            'address_hip_y': address_hip_y,
            'foot_idx': foot_idx,
            'hip_idx': hip_idx,
            'frame_data': frame_data,
            'address': address,
            'top_backswing': top_backswing,
            'impact': impact
        }
    except Exception as e:
        print(f"Error in detect_sliding: {str(e)}")
        return {'result': False, 'error': str(e)}

def detect_loss_of_posture_dtl(keypoints_sequence, events, is_right_handed=True):
    """
    Detect loss of posture using trail side and measuring hip and knee angles.
    """
    try:
        address = events["address"]
        impact = events["impact"]
        
        # Get initial angles at address
        address_keypoints = keypoints_sequence[address]
        
        # Get trail side keypoints
        hip = get_midpoint(address_keypoints[BODY_KEYPOINTS["right_hip"]], address_keypoints[BODY_KEYPOINTS["left_hip"]])
        knee = get_midpoint(address_keypoints[BODY_KEYPOINTS["right_knee"]], address_keypoints[BODY_KEYPOINTS["left_knee"]])
        ankle = get_midpoint(address_keypoints[BODY_KEYPOINTS["right_ankle"]], address_keypoints[BODY_KEYPOINTS["left_ankle"]])
        shoulder = get_midpoint(address_keypoints[BODY_KEYPOINTS["right_shoulder"]], address_keypoints[BODY_KEYPOINTS["left_shoulder"]])
        
        # Calculate vectors
        address_spine_vector = get_vector(hip, shoulder)
        address_femur_vector = get_vector(hip, knee)
        address_tibia_vector = get_vector(knee, ankle)
        
        # Calculate initial angles
        # Hip angle (between spine and femur)
        address_hip_angle = np.degrees(np.arccos(
            np.dot(normalize_vector(address_spine_vector), normalize_vector(address_femur_vector))
        ))
        
        # Knee angle (between femur and tibia)
        address_knee_angle = np.degrees(np.arccos(
            np.dot(normalize_vector(address_femur_vector), normalize_vector(address_tibia_vector))
        ))
        
        required_indices = [BODY_KEYPOINTS["right_hip"], BODY_KEYPOINTS["right_knee"], BODY_KEYPOINTS["right_ankle"], BODY_KEYPOINTS["right_shoulder"], 
                            BODY_KEYPOINTS["left_hip"], BODY_KEYPOINTS["left_knee"], BODY_KEYPOINTS["left_ankle"], BODY_KEYPOINTS["left_shoulder"]]
        
        # Track frame data and maximum angle changes
        frame_data = []
        is_posture_loss_detected = False
        max_hip_diff = 0
        max_knee_diff = 0
        max_hip_frame = address
        max_knee_frame = address
        
        # Check all frames between address and impact
        for frame in range(address, impact + 1):
            current_keypoints = keypoints_sequence[frame]
            
            # Skip frame if any required keypoint is not detected
            missing_keypoint = False
            for idx in required_indices:
                if current_keypoints[idx][0] == 0 and current_keypoints[idx][1] == 0:
                    missing_keypoint = True
                    break
            
            if missing_keypoint:
                # Add frame with missing data
                frame_data.append({
                    'frame': frame,
                    'hip_angle_diff': 0,
                    'knee_angle_diff': 0,
                    'missing_data': True,
                    'posture_loss_detected': False
                })
                continue
            
            # Calculate current angles
            current_hip = get_midpoint(current_keypoints[BODY_KEYPOINTS["right_hip"]], current_keypoints[BODY_KEYPOINTS["left_hip"]])
            current_knee = get_midpoint(current_keypoints[BODY_KEYPOINTS["right_knee"]], current_keypoints[BODY_KEYPOINTS["left_knee"]])
            current_ankle = get_midpoint(current_keypoints[BODY_KEYPOINTS["right_ankle"]], current_keypoints[BODY_KEYPOINTS["left_ankle"]])
            current_shoulder = get_midpoint(current_keypoints[BODY_KEYPOINTS["right_shoulder"]], current_keypoints[BODY_KEYPOINTS["left_shoulder"]])
            
            # Calculate vectors
            current_spine_vector = get_vector(current_hip, current_shoulder)
            current_femur_vector = get_vector(current_hip, current_knee)
            current_tibia_vector = get_vector(current_knee, current_ankle)
            
            # Calculate current angles
            # Hip angle (between spine and femur)
            current_hip_angle = np.degrees(np.arccos(
                np.dot(normalize_vector(current_spine_vector), normalize_vector(current_femur_vector))
            ))
            
            # Knee angle (between femur and tibia)
            current_knee_angle = np.degrees(np.arccos(
                np.dot(normalize_vector(current_femur_vector), normalize_vector(current_tibia_vector))
            ))
            
            # Calculate angle differences
            hip_angle_diff = abs(current_hip_angle - address_hip_angle)
            knee_angle_diff = abs(current_knee_angle - address_knee_angle)
            
            # Update max values
            if hip_angle_diff > max_hip_diff:
                max_hip_diff = hip_angle_diff
                max_hip_frame = frame
                
            if knee_angle_diff > max_knee_diff:
                max_knee_diff = knee_angle_diff
                max_knee_frame = frame
        
            # Check for significant changes in either angle
            frame_posture_loss = hip_angle_diff > POSTURE_THRESHOLD or knee_angle_diff > POSTURE_THRESHOLD
            if frame_posture_loss:
                is_posture_loss_detected = True
                
            # Store frame data
            frame_data.append({
                'frame': frame,
                'hip_angle_diff': hip_angle_diff,
                'knee_angle_diff': knee_angle_diff,
                'hip_angle': current_hip_angle,
                'knee_angle': current_knee_angle,
                'missing_data': False,
                'posture_loss_detected': frame_posture_loss
            })
                
        # Return detailed metrics
        return {
            'result': is_posture_loss_detected,
            'threshold': POSTURE_THRESHOLD,
            'max_hip_diff': max_hip_diff,
            'max_hip_frame': max_hip_frame,
            'max_knee_diff': max_knee_diff,
            'max_knee_frame': max_knee_frame,
            'address_hip_angle': address_hip_angle,
            'address_knee_angle': address_knee_angle,
            'trail_side': 'right' if is_right_handed else 'left',
            'frame_data': frame_data,
            'address': address,
            'impact': impact,
            'required_indices': required_indices
        }
    except Exception as e:
        print(f"Error in detect_loss_of_posture_dtl: {str(e)}")
        return {'result': False, 'error': str(e)}


def detect_early_extension_dtl(keypoints_sequence, events, masks):
    """
    Detect early extension in DTL view using masks for hip tracking and average head keypoints.
    """
    try:
        address = events["address"]
        impact = events["impact"]
        
        # Define head keypoints to use
        head_keypoint_indices = [
            BODY_KEYPOINTS["nose"],
            BODY_KEYPOINTS["left_eye"],
            BODY_KEYPOINTS["right_eye"],
            BODY_KEYPOINTS["left_ear"],
            BODY_KEYPOINTS["right_ear"]
        ]
        
        # Get initial positions at address
        address_keypoints = keypoints_sequence[address]
        
        # Calculate average position of detected head keypoints at address
        detected_head_keypoints = []
        for idx in head_keypoint_indices:
            kp = address_keypoints[idx]
            if not (kp[0] == 0 and kp[1] == 0):  # Only include detected keypoints
                detected_head_keypoints.append(kp)
        
        # If no head keypoints are detected, use nose as fallback
        if not detected_head_keypoints:
            print("Warning: No head keypoints detected at address, using nose as fallback")
            initial_head_pos = address_keypoints[BODY_KEYPOINTS["nose"]]
        else:
            # Calculate average position
            initial_head_pos = np.mean(detected_head_keypoints, axis=0)
        
        # Get hip keypoints for y-coordinate reference
        left_hip = address_keypoints[BODY_KEYPOINTS["left_hip"]]
        right_hip = address_keypoints[BODY_KEYPOINTS["right_hip"]]
        hip_y = get_midpoint(left_hip, right_hip)[1]
        
        # Get initial hip edge from mask at address frame
        address_mask = masks[address+1]
        initial_hip_edge_x = None
        
        if address_mask is not None:
            # Find leftmost point (x_min) at hip_y coordinate in the mask
            hip_row = address_mask[int(hip_y)] if 0 <= int(hip_y) < address_mask.shape[0] else None
            if hip_row is not None:
                non_zero_indices = np.where(hip_row > 0)[0]
                if len(non_zero_indices) > 0:
                    initial_hip_edge_x = non_zero_indices[0]
        
        # If mask approach fails, fall back to keypoint method
        if initial_hip_edge_x is None:
            initial_hip_edge_x = get_midpoint(left_hip, right_hip)[0]
            print("Warning: Using keypoint for hip tracking (mask failed)")
        
        # Track frame data and maximum values
        frame_data = []
        is_extension_detected = False
        max_head_extension = 0
        max_hip_extension = 0
        max_head_frame = address
        max_hip_frame = address
        
        # Check all frames between address and impact
        for frame in range(address, impact + 1):
            current_keypoints = keypoints_sequence[frame]
            current_mask = masks[frame]
            
            # Calculate average position of detected head keypoints for current frame
            current_detected_head_keypoints = []
            for idx in head_keypoint_indices:
                kp = current_keypoints[idx]
                if not (kp[0] == 0 and kp[1] == 0):
                    current_detected_head_keypoints.append(kp)
            
            # Skip frame if no head keypoints are detected
            if not current_detected_head_keypoints:
                frame_data.append({
                    'frame': frame,
                    'head_position': [0, 0],
                    'hip_edge_x': 0,
                    'head_extension': 0,
                    'hip_extension': 0,
                    'num_head_keypoints': 0,
                    'missing_data': True,
                    'extension_detected': False
                })
                continue
                
            current_head_pos = np.mean(current_detected_head_keypoints, axis=0)
            
            # Get current hip y-coordinate for reference
            current_left_hip = current_keypoints[BODY_KEYPOINTS["left_hip"]]
            current_right_hip = current_keypoints[BODY_KEYPOINTS["right_hip"]]
            current_hip_y = get_midpoint(current_left_hip, current_right_hip)[1]
            
            # Get current hip edge from mask
            current_hip_edge_x = None
            if current_mask is not None:
                # Find leftmost point at hip_y coordinate in the mask
                hip_row = current_mask[int(current_hip_y)] if 0 <= int(current_hip_y) < current_mask.shape[0] else None
                if hip_row is not None:
                    non_zero_indices = np.where(hip_row > 0)[0]
                    if len(non_zero_indices) > 0:
                        current_hip_edge_x = non_zero_indices[0]
            
            # Fall back to keypoint method if mask approach fails
            if current_hip_edge_x is None:
                current_hip_edge_x = get_midpoint(current_left_hip, current_right_hip)[0]
            
            # Calculate extensions
            # Use radial distance (Euclidean) for head movement
            head_extension = np.sqrt((current_head_pos[0] - initial_head_pos[0])**2 + 
                                    (current_head_pos[1] - initial_head_pos[1])**2)
            hip_extension = current_hip_edge_x - initial_hip_edge_x
            
            # Update max values
            if head_extension > max_head_extension:
                max_head_extension = head_extension
                max_head_frame = frame
                
            if hip_extension > max_hip_extension:
                max_hip_extension = hip_extension
                max_hip_frame = frame
            
            # Check for extension threshold
            frame_extension_detected = (head_extension > HEAD_EXTENSION_THRESHOLD or 
                                       hip_extension > HIP_EXTENSION_THRESHOLD)
            if frame_extension_detected:
                is_extension_detected = True
                
            # Store frame data
            frame_data.append({
                'frame': frame,
                'head_position': current_head_pos.tolist(),
                'hip_edge_x': current_hip_edge_x,
                'head_extension': head_extension,
                'hip_extension': hip_extension,
                'num_head_keypoints': len(current_detected_head_keypoints),
                'missing_data': False,
                'extension_detected': frame_extension_detected
            })
                
        # Return detailed metrics
        return {
            'result': is_extension_detected,
            'head_extension_threshold': HEAD_EXTENSION_THRESHOLD,
            'hip_extension_threshold': HIP_EXTENSION_THRESHOLD,
            'max_head_extension': max_head_extension,
            'max_head_frame': max_head_frame,
            'max_hip_extension': max_hip_extension,
            'max_hip_frame': max_hip_frame,
            'initial_head_pos': initial_head_pos.tolist(),
            'initial_hip_edge_x': initial_hip_edge_x,
            'head_keypoints_used': len(detected_head_keypoints),
            'frame_data': frame_data,
            'address': address,
            'impact': impact
        }
    except Exception as e:
        print(f"Error in detect_early_extension_dtl: {str(e)}")
        return {'result': False, 'error': str(e)}


def detect_all_errors_from_data(front_keypoints, dtl_keypoints, events, masks_front, masks_dtl, is_right_handed=True):
    """
    Run all error detection methods using already loaded data.
    
    Args:
        front_keypoints: Sequence of keypoints from front view
        dtl_keypoints: Sequence of keypoints from down-the-line view
        events: Dictionary mapping swing phases to frame numbers
        masks_front: Masks for front view
        masks_dtl: Masks for DTL view
        is_right_handed: Boolean indicating if golfer is right-handed
    
    Returns:
        Dictionary containing detailed metrics for each error type
    """
    metrics = {}
    
    try:
        # Detect swaying using front view
        metrics["swaying"] = detect_swaying(front_keypoints, events, masks_front, is_right_handed)
        
        # Detect sliding using front view
        metrics["sliding"] = detect_sliding(front_keypoints, events, masks_front, is_right_handed)
        
        # Detect loss of posture using DTL view
        if dtl_keypoints is not None:
            metrics["loss_of_posture"] = detect_loss_of_posture_dtl(dtl_keypoints, events, is_right_handed)
        else:
            metrics["loss_of_posture"] = {'result': False, 'error': 'No DTL view data available'}
    
        
        if dtl_keypoints is not None:
            metrics["early_extension_dtl"] = detect_early_extension_dtl(dtl_keypoints, events, masks_dtl)
        else:
            metrics["early_extension_dtl"] = {'result': False, 'error': 'No DTL view data available'}
        
        # Combine early extension results
        metrics["early_extension"] = {
            'result': metrics["early_extension_dtl"]['result'] if 'result' in metrics["early_extension_dtl"] else False
        }
        
        # Print summary of detected errors
        print("\nError Detection Summary:")
        for error in ["swaying", "sliding", "loss_of_posture", "early_extension"]:
            detected = metrics[error]['result'] if 'result' in metrics[error] else False
            print(f"{error}: {'Detected' if detected else 'Not detected'}")
            
        return metrics
        
    except Exception as e:
        print(f"Error in detect_all_errors_from_data: {str(e)}")
        return {'error': str(e)}




