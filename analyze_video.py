#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / 'src').resolve()))

import argparse
import os
import json
from pygpt_net.core.vision.analyzer import Analyzer
from pygpt_net.item.ctx import CtxItem

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze frames from a video file')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--interval', type=int, default=30,
                       help='Number of frames to skip between analyses (default: 30)')
    parser.add_argument('--time-interval', type=float,
                       help='Time in seconds between analyses (overrides --interval if set)')
    parser.add_argument('--output', help='Path to save results (optional)')
    args = parser.parse_args()

    # Verify video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    # Initialize analyzer
    analyzer = Analyzer()
    
    print(f"Analyzing video: {args.video_path}")
    print(f"Frame interval: {args.interval}")
    
    # Analyze video
    results = analyzer.analyze_video_frames(
        video_path=args.video_path,
        frame_interval=args.interval,
        time_interval=args.time_interval
    )
    
    # Print or save results
    if args.output:
        # Create output directory with video name
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        output_dir = os.path.join(args.output, video_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to analysis.json in the video directory
        output_file = os.path.join(output_dir, 'analysis.json')
        print(f"Saving results to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        for result in results:
            print(f"\nFrame {args.video_path}-{result['frame']}:")
            print(result['analysis'])
            print("-" * 80)

if __name__ == "__main__":
    main() 