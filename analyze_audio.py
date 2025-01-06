#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / 'src').resolve()))

import argparse
import os
import json
from pygpt_net.core.audio.analyzer import AudioAnalyzer
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze audio from video/audio file')
    parser.add_argument('file_path', help='Path to the video/audio file')
    parser.add_argument('--interval', type=int, default=30,
                       help='Length of each interval in seconds (default: 30)')
    parser.add_argument('--output', help='Path to save results (optional)')
    args = parser.parse_args()

    # Verify file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return

    # Initialize analyzer
    analyzer = AudioAnalyzer()
    
    print(f"Analyzing file: {args.file_path}")
    print(f"Interval length: {args.interval} seconds")
    
    # Analyze audio
    results = analyzer.analyze_audio(
        file_path=args.file_path,
        interval_seconds=args.interval
    )
    
    # Print or save results
    if args.output:
        # Create output directory with file name
        file_name = os.path.splitext(os.path.basename(args.file_path))[0]
        output_dir = os.path.join(args.output, file_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to analysis.json in the file directory
        output_file = os.path.join(output_dir, 'audio_analysis.json')
        print(f"Saving results to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        for result in results:
            print(f"\nInterval {result['interval']} ({result['start_time']}s - {result['end_time']}s):")
            print("\nTranscript:")
            print(result['transcript'])
            print("\nAnalysis:")
            print(json.dumps(result['analysis'], indent=2))
            print("-" * 80)

if __name__ == "__main__":
    main() 
