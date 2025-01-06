#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================== #
# This file is a part of PYGPT package               #
# Website: https://pygpt.net                         #
# GitHub:  https://github.com/szczyglis-dev/py-gpt   #
# MIT License                                        #
# Created By  : Marcin Szczygliński                  #
# Updated Date: 2024.12.14 18:00:00                  #
# ================================================== #

import json
import os
from openai import OpenAI
from moviepy.editor import VideoFileClip

from pydub import AudioSegment
from pygpt_net.item.ctx import CtxItem

class AudioAnalyzer:
    def __init__(self, window=None):
        """
        Audio analyzer

        :param window: Window instance
        """
        self.window = window
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def analyze_audio(self, file_path, interval_seconds=30):
        """
        Analyze audio from file with intervals
        
        :param file_path: path to audio/video file
        :param interval_seconds: length of each interval in seconds
        """
        results = []
        
        # Check if it's a video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        is_video = any(file_path.lower().endswith(ext) for ext in video_extensions)
        
        if is_video:
            # Extract audio from video
            audio_path = self.extract_audio(file_path)
            if not audio_path:
                return results
        else:
            audio_path = file_path

        # Split audio into intervals and analyze each
        intervals = self.split_audio(audio_path, interval_seconds)
        
        for idx, interval in enumerate(intervals):
            start_time = idx * interval_seconds
            end_time = (idx + 1) * interval_seconds
            
            # Transcribe interval
            transcript = self.transcribe_audio(interval['path'])
            
            if transcript:
                # Analyze sentiment
                analysis = self.analyze_sentiment(transcript)
                results.append({
                    'interval': idx + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'path': interval['path'],
                    'transcript': transcript,
                    'analysis': analysis
                })
            
            # Clean up interval file
            try:
                os.remove(interval['path'])
            except Exception as e:
                print(f"Error removing interval file: {str(e)}")
        
        # Clean up extracted audio file if it was from video
        if is_video and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                print(f"Error removing temporary audio file: {str(e)}")
        
        return results

    def split_audio(self, audio_path, interval_seconds):
        """
        Split audio file into intervals
        
        :param audio_path: path to audio file
        :param interval_seconds: length of each interval in seconds
        :return: list of interval paths
        """
        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Create temp directory for intervals
            temp_dir = os.path.join('/tmp', 'audio_intervals')
            os.makedirs(temp_dir, exist_ok=True)
            
            intervals = []
            interval_ms = interval_seconds * 1000  # Convert to milliseconds
            
            # Split audio into intervals
            for i, start in enumerate(range(0, len(audio), interval_ms)):
                end = start + interval_ms
                interval = audio[start:end]
                
                # Generate interval file path
                interval_path = os.path.join(temp_dir, f"interval_{i+1}.mp3")
                
                # Export interval
                interval.export(interval_path, format="mp3")
                
                intervals.append({
                    'path': interval_path,
                    'start': start / 1000,  # Convert back to seconds
                    'end': end / 1000
                })
            
            return intervals
            
        except Exception as e:
            print(f"Error splitting audio: {str(e)}")
            return []

    def extract_audio(self, video_path):
        """
        Extract audio from video file
        
        :param video_path: path to video file
        :return: path to extracted audio file
        """
        try:
            temp_dir = os.path.join('/tmp', 'audio_analysis')
            os.makedirs(temp_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(temp_dir, f"{base_name}_audio.mp3")
            
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)
            video.close()
            
            return audio_path
            
        except Exception as e:
            print(f"Error extracting audio from video: {str(e)}")
            return None

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file using Whisper
        
        :param audio_path: path to audio file
        :return: transcription text
        """
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return None

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of transcribed text
        
        :param text: transcribed text to analyze
        :return: sentiment analysis
        """
        prompt = """
Analyze the sentiment in this classroom discussion transcript. Consider tone, engagement level, and participation.

Instructions:
1. Analyze sentiment based on the spoken content and tone
2. Consider factors like participation level, question asking, and responses
3. Provide a clear explanation of your analysis
4. Return your analysis in the following JSON format:

{
    "sentiment_score": float,  // Score from -5.0 (most negative) to 5.0 (most positive)
    "sentiment_label": string, // Brief label describing the sentiment
    "reasoning": string,       // Brief explanation of your analysis
    "audio_cues": [           // List of observed audio indicators
        string,
        ...
    ],
    "recommendations": [      // List of suggested improvements
        string,
        ...
    ]
}

Note: sentiment_score must be between -5.0 and 5.0, where:
* 5.0: Extremely engaged and enthusiastic
* 3.0: Actively engaged
* 1.0: Slightly engaged
* 0.0: Neutral
* -1.0: Slightly disengaged
* -3.0: Notably disengaged
* -5.0: Completely disengaged or disruptive

Transcript to analyze: """ + text

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.choices[0].message.content
            print(response_text)
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
            print(f"Error in sentiment analysis: {str(e)}")
            raise e
            return None 
