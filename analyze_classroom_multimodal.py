#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / 'src').resolve()))

import os
import json
import argparse
from src.pygpt_net.core.vision.analyzer import Analyzer
from src.pygpt_net.controller.camera import Camera

def analyze_classroom_multimodel(video_path, time_interval=30, max_frames=3):
    """
    Analyze both audio and video content from a video file using multiple models
    
    :param video_path: path to video file
    :param time_interval: interval length in seconds
    :param max_frames: maximum number of frames to capture
    :return: combined analysis results
    """
    # Initialize analyzer
    analyzer = Analyzer()

    # Analyze video frames and transcription
    analysis_results = analyzer.analyze_video_frames_transcription(
        video_path=video_path,
        time_interval=time_interval,
        max_frames=max_frames
    )

    # Combine results
    combined_results = []
    for i, analysis_result in enumerate(analysis_results):
        interval_data = {
            'interval': i + 1,
            'timestamp': i * time_interval,
            'analysis': analysis_result
        }
        combined_results.append(interval_data)

    return combined_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze classroom video for engagement using multiple models')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--interval', type=int, default=30,
                        help='Length of each interval in seconds (default: 30)')
    parser.add_argument('--max-frames', type=int, default=3,
                        help='Maximum number of frames to capture (default: 3)')
    parser.add_argument('--output', help='Path to save results (optional)')
    args = parser.parse_args()

    # Verify file exists
    if not os.path.exists(args.video_path):
        print(f"Error: File not found: {args.video_path}")
        return

    print(f"Analyzing video: {args.video_path}")
    print(f"Interval length: {args.interval} seconds")

    # Analyze video
    results = analyze_classroom_multimodel(
        video_path=args.video_path,
        time_interval=args.interval,
        max_frames=args.max_frames
    )

    # Print or save results
    if args.output:
        # Create output directory with video name
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        output_dir = os.path.join(args.output, video_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save results to analysis.json
        output_file = os.path.join(output_dir, 'classroom_analysis.json')
        print(f"Saving results to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        for result in results:
            print(f"\nInterval {result['interval']} (Timestamp: {result['timestamp']}s):")
            print("\nTranscription:")
            print(result['transcription'])
            print("\nVideo Analysis:")
            print(json.dumps(result['video_analysis'], indent=2))
            print("-" * 80)

if __name__ == "__main__":
    main() 