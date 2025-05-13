import os
import glob
from tqdm import tqdm
import argparse
import pandas as pd
import shutil
from errors import process_swing_data
from visualize_errors import create_error_visualizations
from extract_phases import identify_swing_phases
from extract_poses import extract_video_keypoints

def process_player(dtl_video, front_video, output_dir, create_visualizations=True):
    """Process data for a single player directory"""
    # Get player directory name (used as base for finding videos and poses)
    swing_base_name = os.path.basename(dtl_video).replace(" Down the line.mp4", "")

            
    # Create player-specific output directory
    swing_output_dir = os.path.join(output_dir, swing_base_name)
    os.makedirs(swing_output_dir, exist_ok=True)
    
    dtl_keypoints = extract_video_keypoints(dtl_video)
    print(f"dtl_keypoints.shape: {dtl_keypoints.shape}")
    
    front_keypoints = extract_video_keypoints(front_video)
    print(f"front_keypoints.shape: {front_keypoints.shape}")
    
    phases = identify_swing_phases(front_keypoints, video_path=front_video, output_path=os.path.join(swing_output_dir, "phases"))
    print(f"phases: {phases}")
    front_keypoints = front_keypoints[:,:,:2]
    dtl_keypoints = dtl_keypoints[:,:,:2]
    if "address" not in phases or "top_backswing" not in phases or "impact" not in phases:
        print(f"Skipping {swing_base_name} because of missing phases")
        return None
    # Run error detection
    try:
        
        
        # Detect errors
        error_metrics, masks_front, masks_dtl = process_swing_data(
            front_keypoints=front_keypoints,
            dtl_keypoints=dtl_keypoints,
            events=phases,
            front_video_path=front_video,
            dtl_video_path=dtl_video,
            output_dir=swing_output_dir
        )
        
        # Extract 'result' field from each error metric for simplified CSV storage
        simplified_results = {}
        if 'error' not in error_metrics:
            for error_type, metrics in error_metrics.items():
                if isinstance(metrics, dict) and 'result' in metrics:
                    simplified_results[error_type] = metrics['result']
        
        # Save detailed error metrics to JSON file
        import json
        with open(os.path.join(swing_output_dir, "detailed_error_metrics.json"), 'w') as f:
            json.dump(error_metrics, f, default=lambda x: x.tolist() if hasattr(x, 'tolist') else None)
        
        # Save simplified error results to CSV
        results_path = os.path.join(swing_output_dir, "error_results.csv")
        pd.DataFrame([simplified_results]).to_csv(results_path, index=False)
        print(f"Saved error results to {results_path}")
        
        # Create visualizations if requested
        if create_visualizations:
            
            create_error_visualizations(
                front_keypoints=front_keypoints,
                dtl_keypoints=dtl_keypoints,
                events=phases,
                front_video=front_video,
                dtl_video=dtl_video,
                output_folder=os.path.join(swing_output_dir, "visualizations"),
                error_metrics=error_metrics,
                masks_front=masks_front,
                masks_dtl=masks_dtl
            )
            
        # Copy video files to player output directory if they exist
        front_video_output = os.path.join(swing_output_dir, "front_view.mp4")
        shutil.copy2(front_video, front_video_output)
        print(f"Copied front view video to {front_video_output}")
            
        dtl_video_output = os.path.join(swing_output_dir, "dtl_view.mp4")
        shutil.copy2(dtl_video, dtl_video_output)
        print(f"Copied DTL view video to {dtl_video_output}")
        
        return simplified_results
    
    except Exception as e:
        print(f"Error processing player {swing_base_name}: {str(e)}")
        return None

def process_all_players(videos_dir, output_dir, create_visualizations=True):
    """Process all player directories in the phases folder"""
    # Get all player directories in the phases folder
    dtl_videos = [f for f in glob.glob(os.path.join(videos_dir, "*.mp4")) if "Down the line" in f]
    
    if not  dtl_videos:
        print(f"No swings found in {videos_dir}")
        return
    
    print(f"Found {len(dtl_videos)} swings")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each player directory
    all_results = {}
    for dtl_video in tqdm(dtl_videos, desc="Processing swings"):
        swing_name = os.path.basename(dtl_video).replace("Down the line.mp4", "")
        front_video = dtl_video.replace("Down the line", "Face on right")
        if os.path.exists(front_video):
            results = process_player(dtl_video, front_video, output_dir, create_visualizations)
            if results:
                all_results[swing_name] = results
    
    # Compile overall results
    if all_results:
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        summary_path = os.path.join(output_dir, "all_players_results.csv")
        results_df.to_csv(summary_path)
        print(f"\nSaved summary results to {summary_path}")
        
        # Print overall error statistics
        print("\nOverall Error Statistics:")
        for error in ['swaying', 'sliding', 'loss_of_posture', 'early_extension']:
            if error in results_df.columns:
                error_count = results_df[error].sum()
                error_rate = (error_count / len(results_df)) * 100
                print(f"{error.replace('_', ' ').title()}: {error_count} players ({error_rate:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process golf swing data for multiple players")
    parser.add_argument("--videos_dir", default="videos", help="Directory containing video files")
    parser.add_argument("--output_dir", default="results", help="Directory for output files")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization creation")
    
    args = parser.parse_args()
    
    process_all_players(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        create_visualizations=not args.no_viz
    ) 

