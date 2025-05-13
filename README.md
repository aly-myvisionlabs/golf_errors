# Golf Swing Analysis

This repository contains tools for analyzing golf swings using computer vision and machine learning techniques. The project focuses on error detection and analysis in golf swings through various processing stages.

## Features

- Player swing processing and analysis
- Error detection and visualization
- Phase extraction from golf swings
- Pose estimation and analysis
- Mask extraction for detailed swing analysis

## Requirements

The project uses Python 3.11 and requires several dependencies. You can install them using the provided environment file:

```bash
conda env create -f environment.yaml
```

## Project Structure

- `process_player.py`: Main script for processing individual player swings
- `process_all_players.py`: Script for batch processing multiple players
- `errors.py`: Error detection and analysis implementation
- `extract_phases.py`: Golf swing phase extraction
- `extract_poses.py`: Pose estimation for golf swings
- `extract_masks.py`: Mask extraction for detailed analysis
- `visualize_errors.py`: Visualization tools for error analysis
- `golf_parser.py`: Parser for golf swing data
- `utils.py`: Utility functions
- `models/`: Directory containing model files
- `third_party/`: Third-party dependencies

## Usage

1. Set up the environment:
```bash
conda activate golf
```

2. Process individual player:
```bash
python process_player.py --dtl_video /path/to/video --front_video /path/to/video [options]
```

3. Process all players:
```bash
python process_all_players.py
```
