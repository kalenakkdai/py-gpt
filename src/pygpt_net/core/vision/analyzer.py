#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================== #
# This file is a part of PYGPT package               #
# Website: https://pygpt.net                         #
# GitHub:  https://github.com/szczyglis-dev/py-gpt   #
# MIT License                                        #
# Created By  : Marcin Szczygliński                  #
# Updated Date: 2024.12.14 00:00:00                  #
# ================================================== #

import os
import cv2
import datetime
from openai import OpenAI
import dotenv
import json

dotenv.load_dotenv()  # Load environment variables from .env file

from pygpt_net.core.bridge.context import BridgeContext
from pygpt_net.item.attachment import AttachmentItem
from pygpt_net.item.ctx import CtxItem
from pygpt_net.controller.camera import Camera


class Analyzer:
    def __init__(self, window=None):
        """
        Image analyzer

        :param window: Window instance
        """
        self.window = window if window else self.initialize_minimal_window()

    def initialize_minimal_window(self):
        """
        Initialize a minimal window or necessary components
        """
        class MinimalWindow:
            def __init__(self):
                self.core = self.Core()
                self.controller = self.Controller()

            class Core:
                def __init__(self):
                    self.models = {'gpt-4o': "gpt-4o"}
                    self.gpt = self.GPT()
                    self.config = self.Config()

                class GPT:
                    def __init__(self):
                        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                        self.vision = self.Vision(self)  # Pass self reference to Vision
                    
                    class Vision:
                        def __init__(self, gpt):
                            self.gpt = gpt  # Store reference to parent GPT instance
                        
                        def send(self, context, extra):
                            try:
                                if extra:
                                    messages = extra
                                else:
                                    import base64
                                    messages = [{"role": "system", "content": context.system_prompt}]
                                    
                                    image_contents = []
                                    for attachment in context.attachments.values():
                                        with open(attachment.path, "rb") as image_file:
                                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                                            image_contents.append({
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                                }
                                            })
                                    
                                    messages.append({
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": context.prompt},
                                            *image_contents
                                        ]
                                    })

                                response = self.gpt.client.chat.completions.create(
                                    model=context.model,
                                    messages=messages,
                                    max_tokens=1000
                                )
                                return response
                                
                            except Exception as e:
                                print(f"Error in vision analysis: {str(e)}")
                                raise e

                class Config:
                    def get(self, key):
                        if key == 'vision.capture.quality':
                            return 95
                        if key == 'user_dir':
                            return './data/images'
                        return None

                    def get_user_dir(self, subdir):
                        path = os.path.join('./data/images', subdir)
                        os.makedirs(path, exist_ok=True)
                        return path

            class Controller:
                def __init__(self):
                    self.camera = self.Camera()
                    self.attachment = self.Attachment()

                class Camera:
                    def __init__(self):
                        self.capture_dir = None

                    def get_default_capture_dir(self, video_path):
                        """
                        Get default capture directory based on video filename
                        
                        :param video_path: path to video file
                        :return: path to capture directory
                        """
                        video_name = os.path.splitext(os.path.basename(video_path))[0]
                        return os.path.join('./data', 'analysis', video_name, 'frames')

                    def capture_from_video(self, video_path, frame_number):
                        try:
                            cap = cv2.VideoCapture(video_path)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                            ret, frame = cap.read()
                            
                            if not ret:
                                print("Failed to capture frame")
                                return ""
                            
                            now = datetime.datetime.now()
                            dt = now.strftime("%Y-%m-%d_%H-%M-%S")
                            name = f'frame_{frame_number:06d}_{dt}'
                            
                            # Use provided capture directory or get default based on video name
                            capture_dir = self.capture_dir if self.capture_dir else self.get_default_capture_dir(video_path)
                            os.makedirs(capture_dir, exist_ok=True)
                            
                            path = os.path.join(capture_dir, name + '.jpg')
                            
                            # Save with high quality
                            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            cap.release()
                            
                            return path
                            
                        except Exception as e:
                            print("Video frame capture exception", e)
                            return ""

                class Attachment:
                    def unlock(self):
                        pass

                    def is_capture_clear(self):
                        return False

                    def clear(self, force=False, auto=False):
                        pass

        return MinimalWindow()

    def send(
            self,
            ctx: CtxItem,
            prompt: str,
            files: dict
    ) -> str:
        """
        Send text from user input (called from UI)

        :param ctx: context
        :param prompt: analyze prompt
        :param files: files
        :return: response
        """
        model = self.window.core.models.get("gpt-4o")
        context = BridgeContext()
        context.prompt = prompt
        context.attachments = files
        context.history = []
        context.stream = False
        context.model = model
        context.system_prompt = ("You are an expert in image recognition. "
                                 "You are analyzing the image and providing a detailed description of the image.")

        extra = {}
        output = ""
        response = self.window.core.gpt.vision.send(context, extra)
        if response.choices[0] and response.choices[0].message.content:
            output = response.choices[0].message.content.strip()
        for id in files:
            ctx.images_before.append(files[id].path)
            files[id].consumed = True  # allow for deletion

        # re-allow clearing attachments
        self.window.controller.attachment.unlock()
        return output

    def from_screenshot(
            self,
            ctx: CtxItem,
            prompt: str
    ) -> str:
        """
        Image analysis from screenshot

        :param ctx: context
        :param prompt: analyze prompt
        :return: response
        """
        path = self.window.controller.painter.capture.screenshot(
            attach_cursor=True,
            silent=True,
        )
        attachment = AttachmentItem()
        attachment.path = path
        files = {
            "screenshot": attachment,
        }
        return self.send(ctx, prompt, files)

    def from_camera(
            self,
            ctx: CtxItem,
            prompt: str
    ) -> str:
        """
        Image analysis from camera

        :param ctx: context
        :param prompt: analyze prompt
        :return: response
        """
        path = self.window.controller.camera.capture_frame_save()
        attachment = AttachmentItem()
        attachment.path = path
        files = {
            "camera": attachment,
        }
        if path:
            return self.send(ctx, prompt, files)
        else:
            return "FAILED: There was a problem with capturing the image."

    def from_path(
            self,
            ctx: CtxItem,
            prompt: str,
            path: str
    ) -> str:
        """
        Image analysis from path

        :param ctx: context item
        :param prompt: analyze prompt
        :param path: path to file
        :return: response
        """
        if not path:
            return self.from_current_attachments(ctx, prompt)  # try current if no path provided

        if not os.path.exists(path):
            return "FAILED: File not found"

        attachment = AttachmentItem()
        attachment.path = path
        files = {
            "img": attachment,
        }
        return self.send(ctx, prompt, files)

    def from_current_attachments(
            self,
            ctx: CtxItem,
            prompt: str
    ) -> str:
        """
        Image analysis from current attachments

        :param ctx: context item
        :param prompt: analyze prompt
        :return: response
        """
        mode = self.window.core.config.get("mode")
        files = self.window.core.attachments.get_all(mode)  # clear is locked here
        result = self.send(ctx, prompt, files)  # unlocks clear

        # clear if capture clear
        if self.window.controller.attachment.is_capture_clear():
            self.window.controller.attachment.clear(True, auto=True)

        return result

    def from_video(
            self,
            ctx: CtxItem,
            prompt: str,
            video_path: str,
            frame_number: int = 0
    ) -> str:
        """
        Image analysis from video frame

        :param ctx: context item
        :param prompt: analyze prompt
        :param video_path: path to video file
        :param frame_number: frame number to capture
        :return: response
        """
        path = self.window.controller.camera.capture_from_video(
            video_path,
            frame_number
        )
        if not path:
            return "FAILED: There was a problem capturing the frame from video."
        
        attachment = AttachmentItem()
        attachment.path = path
        files = {
            "video_frame": attachment,
        }
        return (self.send(ctx, prompt, files), path)

    def analyze_video_frames(self, video_path, frame_interval=30, time_interval=None):
        """
        Analyze frames from video at regular intervals
        
        :param video_path: path to video file
        :param frame_interval: number of frames to skip between analyses
        :param time_interval: time in seconds between analyses
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()
        
        ctx = CtxItem()
        prompt = """
You are an AI assistant tasked with analyzing classroom sentiment. Based on the combination of visual cues (e.g., facial expressions, body language) and conversational context (e.g., tone, participation), detect the overall classroom sentiment and provide reasoning for your analysis.

Instructions:
1. Analyze sentiment based on descriptive inputs of what students look like and say
2. Identify emotions such as engagement, confusion, boredom, or frustration
3. Provide a clear and concise explanation of how the visual and conversational data lead to the detected sentiment
4. Return your analysis in the following JSON format:

{
    "sentiment_score": float,  // Score from -5.0 (most negative) to 5.0 (most positive)
    "sentiment_label": string, // Brief label describing the sentiment (e.g., "highly engaged", "moderately bored")
    "reasoning": string,       // Brief explanation of your analysis
    "visual_cues": [          // List of observed visual indicators
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
"""
        
        results = []
        if time_interval:
            # Convert time interval to frame interval
            frame_interval = int(time_interval * fps)
        for frame_num in range(0, total_frames, frame_interval):
            response, path = self.from_video(
                ctx, 
                prompt, 
                video_path, 
                frame_num
            )
            try:
                # Try to extract JSON from the response
                # Look for JSON content between triple backticks if present
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                else:
                    json_str = response
                
                analysis = json.loads(json_str)
            except Exception as e:
                print(f"Error parsing JSON from response at image {video_path}: {str(e)}")
                analysis = response  # Keep original response if JSON parsing fails
                
            results.append({
                'frame': frame_num,
                'path': path,
                'analysis': analysis
            })
        
        return results
    
    def analyze_video_frames_transcription(self, video_path, time_interval=30, max_frames=3):
        """
        Analyze video frames and transcription for sentiment analysis

        :param video_path: path to video file
        :param time_interval: time in seconds between analyses
        :return: list of analysis results
        """
        # Initialize camera
        camera = Camera()

        # Capture frames and transcribe audio
        segments = camera.capture_frames_and_transcription(
            video_path=video_path,
            duration=time_interval,
            max_frames=max_frames
        )

        results = []

        # Prepare context for OpenAI API
        system_prompt = """
        You are an AI assistant tasked with analyzing classroom sentiment. 
        Based on the combination of visual cues (e.g., facial expressions, body language) and conversational context (e.g., tone, participation), 
        detect the overall classroom sentiment and provide reasoning for your analysis.

        Instructions:
        1. Analyze sentiment based on descriptive inputs of what students look like and say
        2. Identify emotions such as engagement, confusion, boredom, or frustration
        3. Provide a clear and concise explanation of how the visual and conversational data lead to the detected sentiment
        4. Return your analysis in the following JSON format:

        {
            "sentiment_score": float,  // Score from -5.0 (most negative) to 5.0 (most positive)
            "sentiment_label": string, // Brief label describing the sentiment (e.g., "highly engaged", "moderately bored")
            "reasoning": string,       // Brief explanation of your analysis
            "transcript_cues": string,  // Transcript cues
            "visual_cues": [          // List of observed visual indicators
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
        """

        for base64_frames, transcription in segments:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": transcription},
                    *[
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                        for frame in base64_frames
                    ]
                ]}
            ]

            context = BridgeContext()
            context.history = []
            context.stream = False
            context.model = self.window.core.models.get("gpt-4o")
            context.system_prompt = ("You are an expert in sentiment analysis. "
                                     "Analyze the provided transcript and images to determine the overall sentiment.")

            # Send data to OpenAI for analysis
            try:
                response = self.window.core.gpt.vision.send(context, messages)
                if response.choices[0] and response.choices[0].message.content:
                    analysis_result = response.choices[0].message.content.strip()
                    if "```json" in analysis_result:
                        json_str = analysis_result.split("```json")[1].split("```")[0]
                        results.append({
                            "transcription": transcription,
                            "analysis": json.loads(json_str)
                        })
                    else:
                        results.append({
                            "transcription": transcription,
                            "analysis": analysis_result
                        })
            except Exception as e:
                print(f"Error in sentiment analysis: {str(e)}")
                print(f"Response: {response}")
                results.append({"error": analysis_result})

        return results