import os
# Set environment variable to handle OpenMP runtime conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import cv2
import torch
import gc
import sys
from tqdm import tqdm
import os.path as osp

# Get the absolute path to the project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMURAI_DIR = os.path.join(ROOT_DIR, "third_party", "samurai")
SAM2_DIR = os.path.join(SAMURAI_DIR, "sam2")

# Add SAMURAI to path using absolute path
sys.path.append(SAM2_DIR)
from sam2.build_sam import build_sam2_video_predictor

def determine_model_cfg(model_path):
    """Determine which config file to use based on model path"""
    # Build absolute path to configs
    config_base = os.path.join(SAM2_DIR, "sam2", "configs", "samurai")
    
    if "large" in model_path:
        return os.path.join(config_base, "sam2.1_hiera_l.yaml")
    elif "base_plus" in model_path:
        return os.path.join(config_base, "sam2.1_hiera_b+.yaml")
    elif "small" in model_path:
        return os.path.join(config_base, "sam2.1_hiera_s.yaml")
    elif "tiny" in model_path:
        return os.path.join(config_base, "sam2.1_hiera_t.yaml")
    else:
        return os.path.join(config_base, "sam2.1_hiera_b+.yaml")  # Default

def extract_masks_from_points(
    video_path, 
    points, 
    frame_idx=0, 
    output_dir=None, 
    model_path=None,
    device="cuda:0"
):
    """
    Extract masks from a video using keypoints for seeding.
    
    Args:
        video_path: Path to input video
        points: List of [x, y] coordinates to use as seed points
        frame_idx: Frame number to use for seeding
        output_dir: Directory to save masks (if None, masks aren't saved to disk)
        model_path: Path to SAMURAI model checkpoint (if None, uses default)
        device: Device to run inference on ("cuda:0" or "cpu")
        
    Returns:
        List of binary masks, one per frame
    """
    try:
        # Set default model path if not provided
        if model_path is None:
            model_path = os.path.join(SAM2_DIR, "checkpoints", "sam2.1_hiera_base_plus.pt")
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize video capture to get properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Important: Reset frame_idx to 0 for SAM2 API
        # SAM2 uses 0-indexed frames internally regardless of the actual frame number in the video
        # Store the original frame_idx for later reference
        original_frame_idx = frame_idx
        working_frame_idx = 0  # Always use 0 for the SAM2 API
        
        # Initialize SAMURAI predictor - using the determine_model_cfg function
        model_cfg = determine_model_cfg(model_path)
        print(f"Using model config: {model_cfg}")
        print(f"Using model checkpoint: {model_path}")
        try:
            predictor = build_sam2_video_predictor(model_cfg, model_path, device=device)
            print("Successfully built SAM2 video predictor")
        except Exception as e:
            print(f"Error building SAM2 video predictor: {str(e)}")
            if "HYDRA_FULL_ERROR" in str(e):
                print("Set the environment variable HYDRA_FULL_ERROR=1 to see more details")
            raise
        
        # Collect all masks
        all_masks = {}  # Initialize as empty dictionary
        
        # Run SAMURAI
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # Initialize state with the video
            state = predictor.init_state(video_path, offload_video_to_cpu=True)
            
            # Convert points to tensor
            points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
            labels = torch.ones(len(points), dtype=torch.int, device=device)  # All foreground
            
            # Add points to initialize tracking
            _, _, masks = predictor.add_new_points_or_box(
                state, 
                points=points_tensor, 
                labels=labels, 
                frame_idx=working_frame_idx,  # Always use 0 here
                obj_id=0
            )
            
            # Save the mask for the initial frame
            if len(masks) > 0:
                mask = masks[0][0].cpu().numpy() > 0.0
                all_masks[original_frame_idx] = mask.astype(np.uint8)
                
                # Save mask if output directory is specified
                if output_dir:
                    mask_filename = os.path.join(output_dir, f"{original_frame_idx:04d}.png")
                    cv2.imwrite(mask_filename, mask.astype(np.uint8) * 255)
            
            # Process each frame
            print(f"Processing {frame_count} frames...")

            # We'll need to map the frames that SAM2 processes back to our original frame indices
            for i, (current_frame_idx, object_ids, masks) in enumerate(tqdm(predictor.propagate_in_video(state), total=frame_count)):
                    
                # Check if we have a valid mask
                if len(masks) > 0 and len(object_ids) > 0 and current_frame_idx not in all_masks: 
                    mask = masks[0][0].cpu().numpy() > 0.0
                    all_masks[current_frame_idx] = mask.astype(np.uint8)
                    
                    # Save mask if output directory is specified
                    if output_dir:
                        mask_filename = os.path.join(output_dir, f"{current_frame_idx:04d}.png")
                        cv2.imwrite(mask_filename, mask.astype(np.uint8) * 255)
        
        # Clean up
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
        
        return all_masks
    except Exception as e:
        print(f"Error in extract_masks_from_points: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def extract_masks_from_bbox(
    video_path, 
    bbox, 
    frame_idx=0, 
    output_dir=None, 
    model_path=None,
    device="cuda:0"
):
    """
    Extract masks from a video using a bounding box for seeding.
    
    Args:
        video_path: Path to input video
        bbox: Tuple of (x1, y1, x2, y2) coordinates for bounding box
        frame_idx: Frame number to use for seeding
        output_dir: Directory to save masks (if None, masks aren't saved to disk)
        model_path: Path to SAMURAI model checkpoint (if None, uses default)
        device: Device to run inference on ("cuda:0" or "cpu")
        
    Returns:
        List of binary masks, one per frame
    """
    try:
        # Set default model path if not provided
        if model_path is None:
            model_path = os.path.join(SAM2_DIR, "checkpoints", "sam2.1_hiera_base_plus.pt")
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize video capture to get properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Important: Reset frame_idx to 0 for SAM2 API
        # SAM2 uses 0-indexed frames internally regardless of the actual frame number in the video
        # Store the original frame_idx for later reference
        original_frame_idx = frame_idx
        working_frame_idx = 0  # Always use 0 for the SAM2 API
        
        # Initialize SAMURAI predictor - using the determine_model_cfg function
        model_cfg = determine_model_cfg(model_path)
        print(f"Using model config: {model_cfg}")
        print(f"Using model checkpoint: {model_path}")
        try:
            predictor = build_sam2_video_predictor(model_cfg, model_path, device=device)
            print("Successfully built SAM2 video predictor")
        except Exception as e:
            print(f"Error building SAM2 video predictor: {str(e)}")
            if "HYDRA_FULL_ERROR" in str(e):
                print("Set the environment variable HYDRA_FULL_ERROR=1 to see more details")
            raise
        
        # Collect all masks
        all_masks = {}  # Initialize as empty dictionary
        
        # Run SAMURAI
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # Initialize state with the video
            state = predictor.init_state(video_path, offload_video_to_cpu=True)
            
            # Add bounding box to initialize tracking
            _, _, masks = predictor.add_new_points_or_box(
                state, 
                box=bbox, 
                frame_idx=working_frame_idx,  # Always use 0 here
                obj_id=0
            )
            
            # Save the mask for the initial frame
            if len(masks) > 0:
                mask = masks[0][0].cpu().numpy() > 0.0
                all_masks[original_frame_idx] = mask.astype(np.uint8)
                
                # Save mask if output directory is specified
                if output_dir:
                    mask_filename = os.path.join(output_dir, f"{original_frame_idx:04d}.png")
                    cv2.imwrite(mask_filename, mask.astype(np.uint8) * 255)
            
            # Process each frame
            print(f"Processing {frame_count} frames...")
            frame_mapping = {}  # Map SAM2 frame indices to original video frame indices

            # We'll need to map the frames that SAM2 processes back to our original frame indices
            for i, (current_frame_idx, object_ids, masks) in enumerate(tqdm(predictor.propagate_in_video(state), total=frame_count)):
                # Map the current SAM2 frame index to our original video frame index
                # For the first frame, SAM2 uses working_frame_idx (0), which maps to our original_frame_idx
                # For subsequent frames, we need to calculate the offset relative to our original frame
                if i == 0:
                    mapped_frame_idx = original_frame_idx
                else:
                    # Calculate how many frames we've moved from the working_frame_idx in SAM2
                    offset = current_frame_idx - working_frame_idx
                    # Apply that same offset to our original_frame_idx
                    mapped_frame_idx = original_frame_idx + offset
                    
                # Skip if the mapped frame is out of bounds
                if mapped_frame_idx < 0 or mapped_frame_idx >= frame_count:
                    continue
                    
                # Check if we have a valid mask
                if len(masks) > 0 and len(object_ids) > 0 and mapped_frame_idx not in all_masks: 
                    mask = masks[0][0].cpu().numpy() > 0.0
                    all_masks[mapped_frame_idx] = mask.astype(np.uint8)
                    
                    # Save mask if output directory is specified
                    if output_dir:
                        mask_filename = os.path.join(output_dir, f"{mapped_frame_idx:04d}.png")
                        cv2.imwrite(mask_filename, mask.astype(np.uint8) * 255)
        
        # Clean up
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
        
        return all_masks
    except Exception as e:
        print(f"Error in extract_masks_from_bbox: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# Simple example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract masks from video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", required=True, help="Output directory for masks")
    parser.add_argument("--points", help="Comma-separated points in format 'x1,y1,x2,y2,...'")
    parser.add_argument("--bbox", help="Bounding box in format 'x1,y1,x2,y2'")
    parser.add_argument("--frame", type=int, default=0, help="Frame to use for seeding")
    
    args = parser.parse_args()
    
    if args.points:
        # Parse points
        point_values = list(map(int, args.points.split(',')))
        points = [(point_values[i], point_values[i+1]) for i in range(0, len(point_values), 2)]
        
        # Extract masks using points
        extract_masks_from_points(
            args.video,
            points,
            args.frame,
            args.output
        )
        
    elif args.bbox:
        # Parse bbox
        x1, y1, x2, y2 = map(int, args.bbox.split(','))
        
        # Extract masks using bbox
        extract_masks_from_bbox(
            args.video,
            (x1, y1, x2, y2),
            args.frame,
            args.output
        )
        
    else:
        print("Error: Either --points or --bbox must be specified")
        exit(1)
    
    print(f"Masks saved to {args.output}/masks/") 