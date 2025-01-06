#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / 'src').resolve()))

import argparse
import os
import json
from pygpt_net.core.audio.analyzer import AudioAnalyzer
from pygpt_net.core.vision.analyzer import Analyzer as VideoAnalyzer

def analyze_classroom(file_path, interval_seconds=30):
    """
    Analyze both audio and video content from a video file
    
    :param file_path: path to video file
    :param interval_seconds: interval length in seconds
    :return: combined analysis results
    """
    # Initialize analyzers
    audio_analyzer = AudioAnalyzer()
    video_analyzer = VideoAnalyzer()
    
    print("Analyzing audio...")
    audio_results = audio_analyzer.analyze_audio(
        file_path=file_path,
        interval_seconds=interval_seconds
    )
    
    print("Analyzing video frames...")
    video_results = video_analyzer.analyze_video_frames(
        video_path=file_path,
        time_interval=interval_seconds
    )
    
    # Combine results
    combined_results = []
    for i in range(max(len(audio_results), len(video_results))):
        interval_data = {
            'interval': i + 1,
            'timestamp': i * interval_seconds,
            'audio_analysis': audio_results[i]['analysis'] if i < len(audio_results) else None,
            'video_analysis': video_results[i]['analysis'] if i < len(video_results) else None,
            'transcript': audio_results[i]['transcript'] if i < len(audio_results) else None,
            'frame_path': video_results[i]['path'] if i < len(video_results) else None,
            'combined_analysis': combine_analyses(
                audio_results[i]['analysis'] if i < len(audio_results) else None,
                video_results[i]['analysis'] if i < len(video_results) else None,
                audio_results[i]['transcript'] if i < len(audio_results) else None
            )
        }
        combined_results.append(interval_data)
    
    return combined_results

def combine_analyses(audio_analysis, video_analysis, transcript):
    """
    Combine audio and video analyses using GPT-4
    
    :param audio_analysis: audio sentiment analysis
    :param video_analysis: video frame analysis
    :param transcript: audio transcript
    :return: combined analysis
    """
    from openai import OpenAI
    import os
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    try:
        audio_analysis_str = json.dumps(audio_analysis, indent=2) if audio_analysis else "No audio analysis available"
    except:
        audio_analysis_str = "No audio analysis available"
        
    try:
        video_analysis_str = json.dumps(video_analysis, indent=2) if video_analysis else "No video analysis available"
    except:
        video_analysis_str = "No video analysis available"
        
    try:
        transcript_str = transcript if transcript else "No transcript available"
    except:
        transcript_str = "No transcript available"
        
    prompt = f"""
Analyze the classroom engagement by combining audio and visual information. Consider both verbal and non-verbal cues.

Audio Analysis:
{audio_analysis_str}

Video Analysis:
{video_analysis_str}

Transcript:
{transcript_str}

Instructions:
1. Combine insights from both audio and visual analyses
2. Consider how verbal and non-verbal cues complement or contrast each other
3. Provide a comprehensive assessment of classroom engagement
4. Return your analysis in the following JSON format:

{{
    "sentiment_score": float,  # Combined score from -5.0 (most negative) to 5.0 (most positive)
    "sentiment_label": string, # Overall sentiment description
    "reasoning": string       # Explanation combining audio and visual cues
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.choices[0].message.content
        
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            else:
                json_str = response_text
            
            analysis = json.loads(json_str)
        except Exception as e:
            print(f"Error parsing JSON from response: {str(e)}")
            analysis = response_text
            
        return analysis
        
    except Exception as e:
        print(f"Error in combined analysis: {str(e)}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze classroom video for engagement')
    parser.add_argument('file_path', help='Path to the video file')
    parser.add_argument('--interval', type=int, default=30,
                       help='Length of each interval in seconds (default: 30)')
    parser.add_argument('--output', help='Path to save results (optional)')
    args = parser.parse_args()

    # Verify file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return

    print(f"Analyzing video: {args.file_path}")
    print(f"Interval length: {args.interval} seconds")
    
    # Analyze video
    results = analyze_classroom(
        file_path=args.file_path,
        interval_seconds=args.interval
    )
    
    # Print or save results
    if args.output:
        # Create output directory with video name
        video_name = os.path.splitext(os.path.basename(args.file_path))[0]
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
            print("\nTranscript:")
            print(result['transcript'])
            print("\nCombined Analysis:")
            print(json.dumps(result['combined_analysis'], indent=2))
            print("-" * 80)

if __name__ == "__main__":
    main() 
    