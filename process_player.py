import os
import argparse
import numpy as np
from errors import process_swing_data, detect_swaying, detect_sliding, detect_loss_of_posture_dtl, detect_early_extension_dtl
from extract_phases import identify_swing_phases, save_comparison_frames
from extract_poses import extract_video_keypoints
from visualize_errors import create_error_visualizations

def process_player(dtl_video=None, front_video=None, output_dir="results", 
                 detect_sway=False, detect_slide=False, 
                 detect_posture=False, detect_early_extension=False,
                 create_visualizations=True, comparison= False,is_right_handed=True):
    """
    Detect specified golf swing errors from video inputs.
    
    Args:
        dtl_video (str): Path to down-the-line view video
        front_video (str): Path to front view video
        output_dir (str): Directory to save results
        detect_sway (bool): Whether to detect swaying
        detect_slide (bool): Whether to detect sliding
        detect_posture (bool): Whether to detect loss of posture
        detect_early_extension (bool): Whether to detect early extension
        create_visualizations (bool): Whether to create visualization videos
        is_right_handed (bool): Whether the golfer is right-handed
    
    Returns:
        dict: Dictionary containing detected errors and their metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {}
    masks_front = None
    masks_dtl = None
    
    # Extract keypoints from videos if provided
    dtl_keypoints = None
    front_keypoints = None
    
    if dtl_video:
        print("Extracting keypoints from DTL video...")
        dtl_keypoints = extract_video_keypoints(dtl_video)
        print(f"DTL keypoints shape: {dtl_keypoints.shape}")
    
    if front_video:
        print("Extracting keypoints from front video...")
        front_keypoints = extract_video_keypoints(front_video)
        print(f"Front keypoints shape: {front_keypoints.shape}")
    
    # Identify swing phases (requires front view)
    events = None
    if front_video:
        print("Identifying swing phases...")
        events = identify_swing_phases(front_keypoints, video_path=front_video)
        print(f"Identified phases: {events}")
    if comparison:
        if events is None:
            print("Cannot compare: Could not identify swing phases")
        else:
            if dtl_video is not None:
                save_comparison_frames(dtl_video, output_dir, events)
            if front_video is not None:
                save_comparison_frames(front_video, output_dir, events)
                
    # Check if we have the required data for each requested error detection
    if detect_sway or detect_slide:
        if not front_video:
            print("Cannot detect sway/slide: Front view video is required")
            results["swaying"] = {"error": "Front view video required"}
            results["sliding"] = {"error": "Front view video required"}
        elif not events:
            print("Cannot detect sway/slide: Could not identify swing phases")
            results["swaying"] = {"error": "Could not identify swing phases"}
            results["sliding"] = {"error": "Could not identify swing phases"}
        else:
            # Process front view errors
            if detect_sway:
                print("Detecting swaying...")
                results["swaying"] = detect_swaying(front_keypoints, events, masks_front, is_right_handed)
            
            if detect_slide:
                print("Detecting sliding...")
                results["sliding"] = detect_sliding(front_keypoints, events, masks_front, is_right_handed)
    
    if detect_posture or detect_early_extension:
        if not dtl_video:
            print("Cannot detect posture/early extension: DTL view video is required")
            if detect_posture:
                results["loss_of_posture"] = {"error": "DTL view video required"}
            if detect_early_extension:
                results["early_extension"] = {"error": "DTL view video required"}
        elif not events:
            print("Cannot detect posture/early extension: Could not identify swing phases")
            if detect_posture:
                results["loss_of_posture"] = {"error": "Could not identify swing phases"}
            if detect_early_extension:
                results["early_extension"] = {"error": "Could not identify swing phases"}
        else:
            # Process DTL view errors
            if detect_posture:
                print("Detecting loss of posture...")
                results["loss_of_posture"] = detect_loss_of_posture_dtl(dtl_keypoints, events, is_right_handed)
            
            if detect_early_extension:
                print("Detecting early extension...")
                results["early_extension_dtl"] = detect_early_extension_dtl(dtl_keypoints, events, masks_dtl)
                results["early_extension"] = {
                    'result': results["early_extension_dtl"]['result'] if 'result' in results["early_extension_dtl"] else False
                }
    
    # Create visualizations if requested
    if create_visualizations and (dtl_video or front_video):
        print("Creating visualizations...")
        create_error_visualizations(
            front_keypoints=front_keypoints,
            dtl_keypoints=dtl_keypoints,
            events=events,
            front_video=front_video,
            dtl_video=dtl_video,
            output_folder=os.path.join(output_dir, "visualizations"),
            error_metrics=results,
            masks_front=masks_front,
            masks_dtl=masks_dtl,
            is_right_handed=is_right_handed
        )
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Detect golf swing errors from videos")
    parser.add_argument("--dtl_video", help="Path to down-the-line view video")
    parser.add_argument("--front_video", help="Path to front view video")
    parser.add_argument("--output_dir", default="results", help="Directory for output files")
    parser.add_argument("--detect_sway", action="store_true", help="Detect swaying")
    parser.add_argument("--detect_slide", action="store_true", help="Detect sliding")
    parser.add_argument("--detect_posture", action="store_true", help="Detect loss of posture")
    parser.add_argument("--detect_early_extension", action="store_true", help="Detect early extension")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization creation")
    parser.add_argument("--left_handed", action="store_true", help="Specify if golfer is left-handed")
    
    args = parser.parse_args()
    
    # Check if at least one video is provided
    if not args.dtl_video and not args.front_video:
        print("Error: At least one video (DTL or front view) must be provided")
        return
    
    # Check if at least one error detection is requested
    if not any([args.detect_sway, args.detect_slide, args.detect_posture, args.detect_early_extension]):
        print("Error: At least one error detection must be requested")
        return

    # Run error detection
    results = process_player(
        dtl_video=args.dtl_video,
        front_video=args.front_video,
        output_dir=args.output_dir,
        detect_sway=args.detect_sway,
        detect_slide=args.detect_slide,
        detect_posture=args.detect_posture,
        detect_early_extension=args.detect_early_extension,
        create_visualizations=not args.no_viz,
        is_right_handed=not args.left_handed
    )
    
    # Print results summary
    print("\nError Detection Results:")
    for error_type, metrics in results.items():
        if 'error' in metrics:
            print(f"{error_type}: {metrics['error']}")
        else:
            detected = metrics.get('result', False)
            print(f"{error_type}: {'Detected' if detected else 'Not detected'}")

if __name__ == "__main__":
    main() 