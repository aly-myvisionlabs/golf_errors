import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# Import constants and detection function from errors
from errors import process_swing_data, SWAY_THRESHOLD, SLIDE_THRESHOLD, POSTURE_THRESHOLD, HEAD_EXTENSION_THRESHOLD, HIP_EXTENSION_THRESHOLD
# Import data loading and processing functions from parser
from golf_parser import map_swing_phases_to_events, define_key_frames, load_pose_data, load_frames_from_video
# Import utility functions from utils
from utils import get_midpoint, get_vector, BODY_KEYPOINTS
FPS = 15


def overlay_mask(img, mask, color=(220, 50, 50), alpha=0.3):
    """Overlay mask on image with specified color and opacity"""
    if mask is None or not isinstance(mask, np.ndarray) or len(mask.shape) != 2:
        return img
    
    if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
        return img
    
    mask_overlay = np.zeros_like(img)
    mask_bool = mask > 0
    if mask_bool.any():
        mask_overlay[mask_bool] = color
        img = cv2.addWeighted(img, 1.0, mask_overlay, alpha, 0)
    
    return img



def draw_skeleton(img, keypoints, color=(255, 0, 0)):
    """Draw skeleton (connections between keypoints) on the image"""
    # Define connections between keypoints
    connections = [
        # Face
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Upper body
        (5, 6), (5, 11), (6, 12), (11, 12),
        # Arms
        (5, 7), (7, 9), (6, 8), (8, 10),
        # Legs
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    # Draw connections
    for connection in connections:
        pt1 = keypoints[connection[0]]
        pt2 = keypoints[connection[1]]
        
        # Check if keypoints are valid
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(img, 
                    (int(pt1[0]), int(pt1[1])), 
                    (int(pt2[0]), int(pt2[1])), 
                    color, 1)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if kp[0] > 0 and kp[1] > 0:
            cv2.circle(img, (int(kp[0]), int(kp[1])), 1, color, -1)
    
    return img

def visualize_swaying(keypoints_sequence, swaying_metrics, images, output_path, masks=None, is_right_handed=True):
    """Visualize swaying using the pre-computed metrics"""
    if 'error' in swaying_metrics:
        print(f"Error in swaying metrics: {swaying_metrics['error']}")
        return
    
    # Extract key metrics
    threshold = swaying_metrics['threshold']
    foot_idx = swaying_metrics['foot_idx']
    hip_idx = swaying_metrics['hip_idx']
    address = swaying_metrics['address']
    top_backswing = swaying_metrics['top_backswing']
    frame_data = swaying_metrics['frame_data']
    
    # Get mask-based coordinates for visualization
    mask_foot_x = swaying_metrics.get('mask_foot_x', swaying_metrics['trail_foot_x'])
    mask_hip_x = swaying_metrics.get('mask_hip_x', swaying_metrics['trail_hip_x'])
    
    # Create output video
    if not images or len(images) == 0:
        print("No images available, creating blank frames")
        height, width = 480, 640
        images = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(len(frame_data))]
    
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
    
    # Get address keypoints for drawing the reference line
    address_keypoints = keypoints_sequence[address]
    foot_y = address_keypoints[foot_idx][1]
    hip_y = address_keypoints[hip_idx][1]
    
    # Create frames for visualization
    for i, frame_info in enumerate(tqdm(frame_data, desc="Creating swaying visualization")):
        frame = frame_info['frame']
        if i < len(images):
            img = images[i].copy()
        else:
            # Create blank image if we run out of images
            img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Overlay mask if available
        if masks is not None and frame in masks:
            mask = masks[frame]
            img = overlay_mask(img, mask)
        
        # Get current keypoints
        current_keypoints = keypoints_sequence[frame]
        current_hip_y = frame_info.get('hip_y', current_keypoints[hip_idx][1])
        
        # Get mask-based hip edge for visualization (if available)
        current_mask_hip_x = frame_info.get('current_mask_hip_x', current_keypoints[hip_idx][0])
        
        # Draw skeleton
        img = draw_skeleton(img, current_keypoints, color=(255, 0, 0))
        
        # Draw reference line from foot to hip at address using mask edges (yellow)
        cv2.line(img, (int(mask_foot_x), int(foot_y)), (int(mask_hip_x), int(hip_y)), (0, 255, 255), 2)
        
        # Display metrics
        distance = frame_info['sway_distance']
        sway_detected = frame_info['sway_detected']
        
        cv2.putText(img, f"Sway Distance: {distance:.2f} px", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Threshold: {threshold} px", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Sway Detected: {sway_detected}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if sway_detected else (0, 255, 0), 2)
        
        video_out.write(img)
    
    video_out.release()
    print(f"Saved swaying visualization to {output_path}")

def visualize_sliding(keypoints_sequence, sliding_metrics, images, output_path, masks=None, is_right_handed=True):
    """Visualize sliding using the pre-computed metrics"""
    if 'error' in sliding_metrics:
        print(f"Error in sliding metrics: {sliding_metrics['error']}")
        return
    
    # Extract key metrics
    threshold = sliding_metrics['threshold']
    foot_idx = sliding_metrics['foot_idx']
    hip_idx = sliding_metrics['hip_idx']
    top_backswing = sliding_metrics['top_backswing']
    impact = sliding_metrics['impact']
    frame_data = sliding_metrics['frame_data']
    
    # Get address position details for the fixed reference line
    mask_foot_x = sliding_metrics.get('mask_foot_x', sliding_metrics['lead_foot_x'])
    address_foot_y = sliding_metrics.get('address_foot_y', None)
    address_hip_y = sliding_metrics.get('address_hip_y', None)
    
    # Create output video
    if not images or len(images) == 0:
        print("No images available, creating blank frames")
        height, width = 480, 640
        images = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(len(frame_data))]
    
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
    
    # If we don't have address hip/foot y values, get from keypoints
    if address_foot_y is None or address_hip_y is None:
        address_keypoints = keypoints_sequence[sliding_metrics['address']]
        address_foot_y = address_keypoints[foot_idx][1]
        address_hip_y = address_keypoints[hip_idx][1]
    
    # Create frames for visualization
    for i, frame_info in enumerate(tqdm(frame_data, desc="Creating sliding visualization")):
        frame = frame_info['frame']
        if i < len(images):
            img = images[i].copy()
        else:
            # Create blank image if we run out of images
            img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Overlay mask if available
        if masks is not None and frame in masks:
            mask = masks[frame]
            img = overlay_mask(img, mask)
        
        # Get current keypoints for skeleton
        current_keypoints = keypoints_sequence[frame]
        
        # Draw skeleton
        img = draw_skeleton(img, current_keypoints, color=(255, 0, 0))
        
        # Draw fixed vertical line at address foot position from hip to foot height (yellow)
        cv2.line(img, 
                (int(mask_foot_x), int(address_hip_y)), 
                (int(mask_foot_x), int(address_foot_y)), 
                (0, 255, 255), 2)
        
        # Display metrics
        distance = frame_info['slide_distance']
        slide_detected = frame_info['slide_detected']
        
        cv2.putText(img, f"Slide Distance: {distance:.2f} px", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Threshold: {threshold} px", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Slide Detected: {slide_detected}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if slide_detected else (0, 255, 0), 2)
        
        video_out.write(img)
    
    video_out.release()
    print(f"Saved sliding visualization to {output_path}")

def visualize_loss_of_posture_dtl(keypoints_sequence, posture_metrics, images, output_path, is_right_handed=True):
    """Visualize loss of posture using the pre-computed metrics (trail side only)"""
    if 'error' in posture_metrics:
        print(f"Error in posture metrics: {posture_metrics['error']}")
        return
    
    # Extract key metrics
    threshold = posture_metrics['threshold']
    address = posture_metrics['address']
    impact = posture_metrics['impact']
    frame_data = posture_metrics['frame_data']
    
    # Handle both old and new format metrics
    if 'trail_side' in posture_metrics:
        trail_side = posture_metrics['trail_side']
    else:
        trail_side = 'right' if is_right_handed else 'left'
    
    address_hip_angle = posture_metrics.get('address_hip_angle', 0)
    address_knee_angle = posture_metrics.get('address_knee_angle', 0)
    
    if not images or len(images) == 0:
        print("No images available, creating blank frames")
        height, width = 480, 640
        images = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(len(frame_data))]
    
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
    
    # Get reference keypoints for address
    address_keypoints = keypoints_sequence[address]
    
    # Get trail side keypoints at address
    hip_address = get_midpoint(address_keypoints[BODY_KEYPOINTS["right_hip"]], address_keypoints[BODY_KEYPOINTS["left_hip"]])
    knee_address = get_midpoint(address_keypoints[BODY_KEYPOINTS["right_knee"]], address_keypoints[BODY_KEYPOINTS["left_knee"]])
    ankle_address = get_midpoint(address_keypoints[BODY_KEYPOINTS["right_ankle"]], address_keypoints[BODY_KEYPOINTS["left_ankle"]])
    shoulder_address = get_midpoint(address_keypoints[BODY_KEYPOINTS["right_shoulder"]], address_keypoints[BODY_KEYPOINTS["left_shoulder"]])
    
    # Create frames for visualization
    for i, frame_info in enumerate(tqdm(frame_data, desc="Creating posture visualization")):
        frame = frame_info['frame']
        if i < len(images):
            img = images[i].copy()
        else:
            # Create blank image if we run out of images
            img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Skip frames with missing data
        if frame_info['missing_data']:
            video_out.write(img)
            continue
        
        # Get current keypoints
        current_keypoints = keypoints_sequence[frame]
        
        # Draw skeleton
        img = draw_skeleton(img, current_keypoints, color=(255, 0, 0))
        
        # Get current trail side keypoints
        current_hip = get_midpoint(current_keypoints[BODY_KEYPOINTS["right_hip"]], current_keypoints[BODY_KEYPOINTS["left_hip"]])
        current_knee = get_midpoint(current_keypoints[BODY_KEYPOINTS["right_knee"]], current_keypoints[BODY_KEYPOINTS["left_knee"]])
        current_ankle = get_midpoint(current_keypoints[BODY_KEYPOINTS["right_ankle"]], current_keypoints[BODY_KEYPOINTS["left_ankle"]])
        current_shoulder = get_midpoint(current_keypoints[BODY_KEYPOINTS["right_shoulder"]], current_keypoints[BODY_KEYPOINTS["left_shoulder"]])
        
        # Draw address (reference) lines in green
        cv2.line(img, 
                (int(hip_address[0]), int(hip_address[1])), 
                (int(shoulder_address[0]), int(shoulder_address[1])), 
                (0, 255, 0), 2)  # Spine (hip to neck)
        cv2.line(img, 
                (int(hip_address[0]), int(hip_address[1])), 
                (int(knee_address[0]), int(knee_address[1])), 
                (0, 255, 0), 2)  # Femur (hip to knee)
        cv2.line(img, 
                (int(knee_address[0]), int(knee_address[1])), 
                (int(ankle_address[0]), int(ankle_address[1])), 
                (0, 255, 0), 2)  # Tibia (knee to ankle)
        
        # Draw current lines in red
        cv2.line(img, 
                (int(current_hip[0]), int(current_hip[1])), 
                (int(current_shoulder[0]), int(current_shoulder[1])), 
                (0, 0, 255), 2)  # Spine (hip to neck)
        cv2.line(img, 
                (int(current_hip[0]), int(current_hip[1])), 
                (int(current_knee[0]), int(current_knee[1])), 
                (0, 0, 255), 2)  # Femur (hip to knee)
        cv2.line(img, 
                (int(current_knee[0]), int(current_knee[1])), 
                (int(current_ankle[0]), int(current_ankle[1])), 
                (0, 0, 255), 2)  # Tibia (knee to ankle)
        
        # Handle both old and new format metrics
        if 'hip_angle_diff' in frame_info:
            hip_angle_diff = frame_info['hip_angle_diff']
            knee_angle_diff = frame_info['knee_angle_diff']
            current_hip_angle = frame_info.get('hip_angle', 0)
            current_knee_angle = frame_info.get('knee_angle', 0)
        else:
            # Use old format metrics if new ones don't exist
            hip_angle_diff = frame_info.get('spine_angle_diff', 0)
            knee_angle_diff = frame_info.get('femur_angle_diff', 0)
            current_hip_angle = hip_angle_diff  # Just show diff as we don't have absolute
            current_knee_angle = knee_angle_diff
        
        # Determine if posture loss is detected
        posture_loss_detected = frame_info['posture_loss_detected']
        
        # Display metrics
        cv2.putText(img, f"Hip Angle: {current_hip_angle:.2f}° (Diff: {hip_angle_diff:.2f}°)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Knee Angle: {current_knee_angle:.2f}° (Diff: {knee_angle_diff:.2f}°)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Address Hip: {address_hip_angle:.2f}°, Knee: {address_knee_angle:.2f}°", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Threshold: {threshold}°", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Loss of Posture: {posture_loss_detected}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if posture_loss_detected else (0, 255, 0), 2)
        
        video_out.write(img)
    
    video_out.release()
    print(f"Saved posture visualization to {output_path}")

def visualize_early_extension_dtl(keypoints_sequence, extension_metrics, images, output_path, masks=None):
    """Visualize early extension in DTL view using masks for hip tracking and radial head movement."""
    if 'error' in extension_metrics:
        print(f"Error in DTL extension metrics: {extension_metrics['error']}")
        return
    
    # Extract key metrics
    threshold = extension_metrics.get('head_extension_threshold', HEAD_EXTENSION_THRESHOLD)
    address = extension_metrics['address']
    impact = extension_metrics['impact']
    frame_data = extension_metrics['frame_data']
    

    
    # Get initial positions (updated for new metrics format)
    if 'initial_head_pos' in extension_metrics:
        initial_head_pos = extension_metrics['initial_head_pos']
    else:
        # Fallback for older metrics format
        initial_head_pos = [0, extension_metrics.get('initial_nose_y', 0)]
    
    initial_hip_edge_x = extension_metrics.get('initial_hip_edge_x', 0)
    
    # Create output video
    if not images or len(images) == 0:
        print("No images available, creating blank frames")
        height, width = 480, 640
        images = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(len(frame_data))]
    
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
    
    # Create frames for visualization
    for i, frame_info in enumerate(tqdm(frame_data, desc="Creating DTL early extension visualization")):
        frame = frame_info['frame']
        if i < len(images):
            img = images[i].copy()
        else:
            # Create blank image if we run out of images
            img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Skip frames with missing data
        if frame_info.get('missing_data', False):
            video_out.write(img)
            continue
        
        # Get current keypoints
        current_keypoints = keypoints_sequence[frame]
        
        # Overlay mask if available
        if masks is not None and frame in masks:
            mask = masks[frame]
            img = overlay_mask(img, mask)
        
        # Draw skeleton
        img = draw_skeleton(img, current_keypoints, color=(255, 0, 0))
        
        # Get current positions from metrics
        if 'head_position' in frame_info:
            current_head_pos = frame_info['head_position']
        else:
            # Fallback for older metrics format
            current_head_pos = [current_keypoints[0][0], frame_info.get('nose_y', 0)]
            
        current_hip_edge_x = frame_info.get('hip_edge_x', 0)
        
        # Calculate mid hip position for y-coordinate reference
        current_mid_hips = get_midpoint(current_keypoints[11], current_keypoints[12])
        
        # Draw vertical line at initial hip edge position (yellow)
        cv2.line(img, (int(initial_hip_edge_x), 0), (int(initial_hip_edge_x), height), (0, 255, 255), 2)
        
        # Draw circle with center at initial head position and radius equal to threshold (yellow)
        cv2.circle(img, (int(initial_head_pos[0]), int(initial_head_pos[1])), int(threshold), (0, 255, 255), 1)
        
        # Draw dot for current head position (red)
        cv2.circle(img, (int(current_head_pos[0]), int(current_head_pos[1])), 1, (255, 0, 0), -1)
        
        
        # Get extensions from metrics
        hip_extension = frame_info.get('hip_extension', 0)
        head_extension = frame_info.get('head_extension', 0)
        extension_detected = frame_info.get('extension_detected', False)
        
        # Display number of head keypoints used if available
        num_head_keypoints = frame_info.get('num_head_keypoints', 1)
        
        # Display metrics
        cv2.putText(img, f"Hip Forward: {hip_extension:.2f} px", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Head Movement: {head_extension:.2f} px", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Threshold: {threshold} px", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Head Keypoints: {num_head_keypoints}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Early Extension: {extension_detected}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if extension_detected else (0, 255, 0), 2)
        
        video_out.write(img)
    
    video_out.release()
    print(f"Saved DTL early extension visualization to {output_path}")


def create_error_visualizations(front_keypoints, dtl_keypoints, events, front_video=None, 
                              dtl_video=None, output_folder="visualizations", is_right_handed=True, error_metrics=None, masks_front=None, masks_dtl=None):
    """Create visualizations for all error types using error metrics"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get error metrics if not provided
    if error_metrics is None:
        error_metrics, masks_front, masks_dtl = process_swing_data(front_keypoints, dtl_keypoints, events, front_video, dtl_video, output_folder, is_right_handed)
    
    
    
    
    
    # Create output paths
    sway_output = os.path.join(output_folder, "sway_visualization.mp4")
    slide_output = os.path.join(output_folder, "slide_visualization.mp4")
    posture_output = os.path.join(output_folder, "posture_visualization.mp4")
    ee_dtl_output = os.path.join(output_folder, "early_extension_dtl_visualization.mp4")
    
    # Extract key frames from metrics for loading video frames
    swaying = error_metrics.get('swaying', {})
    sliding = error_metrics.get('sliding', {})
    loss_of_posture = error_metrics.get('loss_of_posture', {})
    ee_dtl = error_metrics.get('early_extension_dtl', {})
    
    # Load front view frames for different phases
    if 'address' in swaying and 'top_backswing' in swaying:
        sway_images = load_frames_from_video(front_video, swaying['address'], swaying['top_backswing']) if front_video else []
    else:
        sway_images = []
    
    if 'top_backswing' in sliding and 'impact' in sliding:
        slide_images = load_frames_from_video(front_video, sliding['top_backswing'], sliding['impact']) if front_video else []
    else:
        slide_images = []
    
    # Load DTL view frames
    if 'address' in loss_of_posture and 'impact' in loss_of_posture:
        posture_images = load_frames_from_video(dtl_video, loss_of_posture['address'], loss_of_posture['impact']) if dtl_video else []
    else:
        posture_images = []
    
    if 'address' in ee_dtl and 'impact' in ee_dtl:
        ee_dtl_images = load_frames_from_video(dtl_video, ee_dtl['address'], ee_dtl['impact']) if dtl_video else []
    else:
        ee_dtl_images = []
    
    # Create visualizations
    print("Creating swaying visualization...")
    visualize_swaying(front_keypoints, error_metrics['swaying'], sway_images, sway_output, masks_front, is_right_handed)
    
    print("Creating sliding visualization...")
    visualize_sliding(front_keypoints, error_metrics['sliding'], slide_images, slide_output, masks_front, is_right_handed)
    
    if dtl_keypoints is not None and 'error' not in error_metrics.get('loss_of_posture', {}):
        print("Creating posture visualization...")
        visualize_loss_of_posture_dtl(dtl_keypoints, error_metrics['loss_of_posture'], posture_images, posture_output, is_right_handed)
        
        print("Creating DTL early extension visualization...")
        visualize_early_extension_dtl(dtl_keypoints, error_metrics['early_extension_dtl'], ee_dtl_images, ee_dtl_output, masks_dtl)
    
    print("All visualizations complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations for golf swing errors")
    parser.add_argument("--front_pose_file", help="Path to front view pose data (.npy file)")
    parser.add_argument("--phases_folder", help="Path to folder containing swing phase images")
    parser.add_argument("--dtl_pose_file", help="Path to DTL view pose data (.npy file)", default=None)
    parser.add_argument("--front_video", help="Path to front view video file", default=None)
    parser.add_argument("--dtl_video", help="Path to DTL view video file", default=None)
    parser.add_argument("--output_folder", help="Path to output folder", default="visualizations")
    parser.add_argument("--left_handed", help="Specify if golfer is left-handed", action="store_true")
    
    args = parser.parse_args()
    
    create_error_visualizations(
        args.front_pose_file,
        args.dtl_pose_file,
        args.phases_folder,
        args.front_video,
        args.dtl_video,
        args.output_folder,
        not args.left_handed  # is_right_handed is True if not left_handed
    )