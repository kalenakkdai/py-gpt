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

import datetime
import os
import time
from typing import Any

import cv2
import base64
import tempfile
import shutil
from PySide6.QtCore import Slot, QObject, QTimer, QRunnable, Signal
from PySide6.QtGui import QImage, QPixmap, Qt
from PySide6.QtWidgets import QMessageBox

from pygpt_net.core.events import AppEvent, KernelEvent
from pygpt_net.core.camera import CaptureWorker
from pygpt_net.utils import trans
from openai import OpenAI
from pygpt_net.item.ctx import CtxItem
from pygpt_net.ui.dialog.result_dialog import ResultDialog  # Ensure this path is correct
import wave
import pyaudio

class Camera(QObject):
    def __init__(self, window=None):
        """
        Camera controller

        :param window: Window instance
        """
        super(Camera, self).__init__()
        self.window = window
        self.frame = None
        self.thread_started = False
        self.is_capture = False
        self.stop = False
        self.auto = False
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.auto_capture_timer = QTimer(self)
        self.result_dialog = ResultDialog(self.window)  # Initialize the dialog

    def setup(self):
        """Setup camera"""
        if self.is_capture and not self.thread_started:
            self.start()

    def setup_ui(self):
        """Update layout checkboxes"""
        if self.window.core.config.get('vision.capture.enabled'):
            self.is_capture = True
            self.window.ui.menu['video.capture'].setChecked(True)
            self.window.ui.nodes['icon.video.capture'].set_icon(":/icons/webcam.svg")
            # self.window.ui.nodes['vision.capture.enable'].setChecked(True)
        else:
            self.is_capture = False
            self.window.ui.menu['video.capture'].setChecked(False)
            self.window.ui.nodes['icon.video.capture'].set_icon(":/icons/webcam_off.svg")
            # self.window.ui.nodes['vision.capture.enable'].setChecked(False)

        if self.window.core.config.get('vision.capture.auto'):
            self.auto = True
            self.window.ui.menu['video.capture.auto'].setChecked(True)
            # self.window.ui.nodes['vision.capture.auto'].setChecked(True)
        else:
            self.auto = False
            self.window.ui.menu['video.capture.auto'].setChecked(False)
            # self.window.ui.nodes['vision.capture.auto'].setChecked(False)

        # update label
        if not self.window.core.config.get('vision.capture.auto'):
            self.window.ui.nodes['video.preview'].label.setText(trans("vision.capture.label"))
        else:
            self.window.ui.nodes['video.preview'].label.setText(trans("vision.capture.auto.label"))

    def update(self):
        """Update camera frame"""
        if not self.thread_started \
                or self.frame is None \
                or not self.is_capture:
            return

        # scale and update frame
        width = self.window.ui.nodes['video.preview'].video.width()
        image = QImage(
            self.frame,
            self.frame.shape[1],
            self.frame.shape[0],
            self.frame.strides[0],
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            width,
            pixmap.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.window.ui.nodes['video.preview'].video.setPixmap(scaled_pixmap)

    def manual_capture(self, force: bool = False):
        """
        Capture frame via click on video output

        :param force: force capture even if auto is enabled
        """
        if not self.is_auto() or force:
            if not self.capture_frame(True):
                event = KernelEvent(KernelEvent.STATE_ERROR, {
                    'msg': trans("vision.capture.manual.captured.error"),
                })
                self.window.dispatch(event)
        else:
            event = KernelEvent(KernelEvent.STATUS, {
                'status': trans('vision.capture.auto.click'),
            })
            self.window.dispatch(event)
        self.window.dispatch(AppEvent(AppEvent.CAMERA_CAPTURED))  # app event

    def internal_capture(self) -> bool:
        """
        Capture frame internally

        :return: True if success
        """
        before_enabled = self.is_enabled()
        if not self.thread_started:
            self.is_capture = True
            self.window.ui.menu['video.capture'].setChecked(True)
            self.window.ui.nodes['icon.video.capture'].set_icon(":/icons/webcam.svg")
            print("Starting camera thread...")
            self.start()
            time.sleep(3)

        if not self.capture_frame(False):
            event = KernelEvent(KernelEvent.STATE_ERROR, {
                    'msg': trans("vision.capture.manual.captured.error"),
                })
            self.window.dispatch(event)
            result = False
        else:
            self.window.dispatch(AppEvent(AppEvent.CAMERA_CAPTURED))  # app event
            result = True

        # stop capture if not enabled before
        if not before_enabled:
            self.disable_capture_internal()

        return result

    def handle_auto_capture(self):
        """Handle auto capture"""
        if self.is_enabled():
            if self.is_auto():
                self.capture_frame(switch=False)
                self.window.controller.chat.log("Captured frame from camera.")  # log

    def get_current_frame(self, flip_colors: bool = True):
        """
        Get current frame

        :param flip_colors: True if flip colors
        """
        if self.frame is None:
            return None
        if flip_colors:
            return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        else:
            return self.frame

    def capture_frame(self, switch: bool = True) -> bool:
        """
        Capture frame and save it as attachment

        :param switch: true if switch to attachments tab (tmp: disabled)
        :return: True if success
        """
        # clear attachments before capture if needed
        if self.window.controller.attachment.is_capture_clear():
            self.window.controller.attachment.clear(True, auto=True)

        # capture frame
        try:
            # prepare filename
            now = datetime.datetime.now()
            dt = now.strftime("%Y-%m-%d_%H-%M-%S")
            name = 'cap-' + dt
            path = os.path.join(
                self.window.core.config.get_user_dir('capture'),
                name + '.jpg'
            )

            # capture frame
            compression_params = [
                cv2.IMWRITE_JPEG_QUALITY,
                int(self.window.core.config.get('vision.capture.quality'))
            ]
            frame = self.get_current_frame()
            self.window.controller.painter.capture.camera()  # capture to draw

            cv2.imwrite(path, frame, compression_params)
            mode = self.window.core.config.get('mode')

            # make attachment
            dt_info = now.strftime("%Y-%m-%d %H:%M:%S")
            title = trans('vision.capture.name.prefix') + ' ' + name
            title = title.replace('cap-', '').replace('_', ' ')
            self.window.core.attachments.new(mode, title, path, False)
            self.window.core.attachments.save()
            self.window.controller.attachment.update()

            # show last capture time in status
            event = KernelEvent(KernelEvent.STATUS, {
                'status': trans("vision.capture.manual.captured.success") + ' ' + dt_info,
            })
            self.window.dispatch(event)

            return True

        except Exception as e:
            print("Frame capture exception", e)
            self.window.core.debug.log(e)
            event = KernelEvent(KernelEvent.STATUS, {
                'status': trans('vision.capture.error'),
            })
            self.window.dispatch(event)
        return False

    def capture_frame_save(self) -> str:
        """
        Capture frame and save 

        :return: Path to saved frame
        """
        # capture frame
        before_enabled = self.is_enabled()
        if not self.thread_started:
            self.is_capture = True
            self.window.ui.menu['video.capture'].setChecked(True)
            self.window.ui.nodes['icon.video.capture'].set_icon(":/icons/webcam.svg")
            print("Starting camera thread...")
            self.start()
            time.sleep(3)

        path = ""
        try:
            # prepare filename
            now = datetime.datetime.now()
            dt = now.strftime("%Y-%m-%d_%H-%M-%S")
            name = 'cap-' + dt
            path = os.path.join(
                self.window.core.config.get_user_dir('capture'),
                name + '.jpg'
            )
            # capture frame
            compression_params = [
                cv2.IMWRITE_JPEG_QUALITY,
                int(self.window.core.config.get('vision.capture.quality'))
            ]
            frame = self.get_current_frame()
            cv2.imwrite(path, frame, compression_params)
            # stop capture if not enabled before
            if not before_enabled:
                self.disable_capture_internal()

            return path

        except Exception as e:
            print("Frame capture exception", e)
            self.window.core.debug.log(e)

        # stop capture if not enabled before
        if not before_enabled:
            self.disable_capture_internal()
        return path

    def disable_capture_internal(self):
        """Disable camera capture"""
        self.window.ui.menu['video.capture'].setChecked(False)
        self.window.ui.nodes['icon.video.capture'].set_icon(":/icons/webcam_off.svg")
        self.disable_capture(force=True)

    def show_camera(self):
        """Show camera"""
        if self.is_capture:
            self.window.ui.nodes['video.preview'].setVisible(True)

    def hide_camera(self, stop: bool = True):
        """
        Hide camera

        :param stop: True if stop capture thread
        """
        self.window.ui.nodes['video.preview'].setVisible(False)

        if stop:
            self.stop_capture()

    def enable_capture(self):
        """Enable capture"""
        if not self.capture_allowed():
            return

        self.is_capture = True
        self.window.core.config.set('vision.capture.enabled', True)
        """
        self.window.controller.config.checkbox.apply(
            'config',
            'vision.capture.enabled',
            {'value': True}
        )
        """
        self.window.ui.nodes['video.preview'].setVisible(True)
        if not self.thread_started:
            self.start()

    def disable_capture(self, force: bool = False):
        """
        Disable capture

        :param force: force disable
        """
        if not self.capture_allowed() and not force:
            return

        self.is_capture = False
        self.window.core.config.set('vision.capture.enabled', False)
        """
        self.window.controller.config.checkbox.apply(
            'config',
            'vision.capture.enabled',
            {'value': False}
        )
        """
        # self.window.ui.nodes['vision.capture.enable'].setChecked(False)
        self.window.ui.nodes['video.preview'].setVisible(False)
        self.stop_capture()
        self.blank_screen()

    def toggle(self, state: bool):
        """
        Toggle camera

        :param state: state
        """
        if state:
            self.enable_capture()
        else:
            self.disable_capture()
        self.setup_ui()

    def toggle_capture(self):
        """Toggle camera"""
        if not self.is_capture:
            self.enable_capture()
        else:
            self.disable_capture()
        self.setup_ui()

    def enable_auto(self):
        """Enable capture"""
        if not self.capture_allowed():
            return

        self.auto = True
        self.window.core.config.set('vision.capture.auto', True)
        self.window.ui.menu['video.capture.auto'].setChecked(True)
        """
        self.window.controller.config.checkbox.apply(
            'config',
            'vision.capture.auto',
            {'value': True}
        )
        """
        self.window.ui.nodes['video.preview'].label.setText(trans("vision.capture.auto.label"))

        if not self.window.core.config.get('vision.capture.enabled'):
            self.enable_capture()
            self.window.ui.menu['video.capture'].setChecked(True)

        # Start the timer to capture and send every 10 seconds
        interval = 10000  # 5,000 ms = 5 seconds
        duration = 5
        self.auto_capture_timer.timeout.connect(lambda: self.capture_and_send(duration))
        self.auto_capture_timer.start(interval)

    def disable_auto(self):
        """Disable capture"""
        if not self.capture_allowed():
            return

        self.auto = False
        self.window.core.config.set('vision.capture.auto', False)
        self.window.ui.menu['video.capture.auto'].setChecked(False)
        """
        self.window.controller.config.checkbox.apply(
            'config',
            'vision.capture.auto',
            {'value': False}
        )
        """
        self.window.ui.nodes['video.preview'].label.setText(trans("vision.capture.label"))

        # Stop the timer
        self.auto_capture_timer.stop()

    def toggle_auto(self, state: bool):
        """
        Toggle camera

        :param state: state (True/False)
        """
        if state:
            self.enable_auto()
        else:
            self.disable_auto()

        self.window.update_status('')

    def is_enabled(self) -> bool:
        """
        Check if camera is enabled

        :return: True if enabled, false otherwise
        """
        return self.is_capture

    def is_auto(self) -> bool:
        """
        Check if camera is enabled

        :return: True if enabled, false otherwise
        """
        return self.auto

    def blank_screen(self):
        """Make and set blank screen"""
        self.window.ui.nodes['video.preview'].video.setPixmap(QPixmap.fromImage(QImage()))

    def start(self):
        """Start camera thread"""
        if self.thread_started:
            return

        # prepare thread
        self.stop = False

        # worker
        worker = CaptureWorker()
        worker.window = self.window

        # signals
        worker.signals.capture.connect(self.handle_capture)
        worker.signals.finished.connect(self.handle_stop)
        worker.signals.unfinished.connect(self.handle_unfinished)
        worker.signals.stopped.connect(self.handle_stop)
        worker.signals.error.connect(self.handle_error)

        # start
        self.window.threadpool.start(worker)
        self.thread_started = True
        self.window.dispatch(AppEvent(AppEvent.CAMERA_ENABLED))  # app event

    def stop_capture(self):
        """Stop camera capture thread"""
        if not self.thread_started:
            return

        self.stop = True
        self.window.dispatch(AppEvent(AppEvent.CAMERA_DISABLED))  # app event

    @Slot(object)
    def handle_error(self, err: Any):
        """
        Handle thread error signal

        :param err: error message
        """
        self.window.core.debug.log(err)
        self.window.ui.dialogs.alert(err)

    @Slot(object)
    def handle_capture(self, frame):
        """
        Handle capture frame signal

        :param frame: frame
        """
        self.frame = frame
        self.update()

    @Slot()
    def handle_stop(self):
        """On capture stopped signal"""
        self.thread_started = False
        self.hide_camera(False)

    @Slot()
    def handle_unfinished(self):
        """On capture unfinished (never started) signal"""
        if self.window.core.platforms.is_snap():
            self.window.ui.dialogs.open(
                'snap_camera',
                width=400,
                height=200
            )
        self.thread_started = False
        self.disable_capture()

    def capture_allowed(self) -> bool:
        """
        Check if capture is allowed

        :return: True if capture is allowed
        """
        if self.window.controller.painter.is_active():
            return True
        if self.window.controller.ui.vision.has_vision():
            return True
        return False

    def capture_from_video(
        self,
        video_path: str,
        frame_number: int = 0
    ) -> str:
        """
        Capture frame from video file

        :param video_path: path to video file
        :param frame_number: frame number to capture (default: 0 for first frame)
        :return: path to saved frame
        """
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read the frame
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to capture frame")
                return ""
            
            # Prepare filename
            now = datetime.datetime.now()
            dt = now.strftime("%Y-%m-%d_%H-%M-%S")
            name = 'video-cap-' + dt
            path = os.path.join(
                self.window.core.config.get_user_dir('capture'),
                name + '.jpg'
            )
            
            # Save the frame
            compression_params = [
                cv2.IMWRITE_JPEG_QUALITY,
                int(self.window.core.config.get('vision.capture.quality'))
            ]
            cv2.imwrite(path, frame, compression_params)
            
            # Release the video capture
            cap.release()
            
            return path
            
        except Exception as e:
            print("Video frame capture exception", e)
            self.window.core.debug.log(e)
            return ""

    def capture_frames_and_transcription(self, video_path: str, duration: int = 30, max_frames: int = 3) -> list:
        """
        Capture frames from video segments, save them, and return a list of transcriptions with frames.

        :param video_path: path to video file
        :param duration: duration of each video segment to capture in seconds
        :param max_frames: maximum number of frames to capture per segment
        :return: list of tuples (list of base64 encoded frames, transcription text) for each segment
        """
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            segment_frames = int(fps * duration)  # Frames per segment

            results = []

            # Get the parent directory of the video_path
            parent_dir = os.path.dirname(video_path)
            
            # Extract the video file name without the extension
            video_file_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Define the base directory for analysis using the parent directory
            analysis_base_dir = os.path.join(parent_dir, '..', 'analysis')
            
            # Create directories for saving frames and segments
            frames_dir = os.path.join(analysis_base_dir, video_file_name, 'frames')
            segments_dir = os.path.join(analysis_base_dir, video_file_name, 'segments')
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(segments_dir, exist_ok=True)

            segment_index = 0
            while cap.isOpened():
                # Set the start frame for the current segment
                cap.set(cv2.CAP_PROP_POS_FRAMES, segment_index * segment_frames)

                # Initialize storage for encoded frames
                base64_frames = []
                frame_count = 0
                captured_frames = 0

                while frame_count < segment_frames and captured_frames < max_frames:
                    success, frame = cap.read()
                    if not success:
                        break

                    if frame_count % (segment_frames // max_frames) == 0:
                        # Save frame as image
                        frame_filename = os.path.join(frames_dir, f"segment_{segment_index}_frame_{captured_frames}.jpg")
                        cv2.imwrite(frame_filename, frame)

                        # Encode frame to base64
                        _, buffer = cv2.imencode(".jpg", frame)
                        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

                        captured_frames += 1

                    frame_count += 1

                if not base64_frames:
                    break

                # Extract audio segment for transcription
                audio_filename = os.path.join(segments_dir, f"segment_{segment_index}_audio.wav")
                os.system(f"ffmpeg -i {video_path} -ss {segment_index * duration} -t {duration} -q:a 0 -map a {audio_filename} -y -loglevel quiet")

                try:
                    # Use OpenAI client for transcription
                    with open(audio_filename, "rb") as audio_file:
                        transcription = self.client.audio.transcriptions.create(
                            model="whisper-1",
                                file=audio_file,
                                response_format="text"
                        )
                        results.append((base64_frames, transcription))

                        segment_index += 1
                except Exception as e:
                    print(f"Error in transcription: {e}")
                    transcription = ""
                    break
            # Release the video capture
            cap.release()

            return results

        except Exception as e:
            print("Video frame capture and transcription exception", e)
            raise e

    def capture_and_send(self, interval=2):
        """Capture an image and send it for analysis in a separate thread"""
        # Show loading indicator
        event = KernelEvent(KernelEvent.STATUS, {
            'status': 'Capturing and analyzing...',
        })
        self.window.dispatch(event)
        
        worker = CaptureAndSendWorker(self, interval)
        worker.signals.finished.connect(self.show_analysis_result)
        worker.signals.error.connect(self.handle_capture_error)
        self.window.threadpool.start(worker)

    def handle_capture_error(self, error_msg):
        """Handle errors from the capture worker"""
        print(f"Capture and analysis error: {error_msg}")
        # Optionally show error in UI
        event = KernelEvent(KernelEvent.STATE_ERROR, {
            'msg': f"Capture error: {error_msg}",
        })
        self.window.dispatch(event)

    def show_analysis_result(self, result: str, image_path: str):
        """Show analysis result in the dialog with image"""
        self.result_dialog.add_result(result, image_path)
        if not self.result_dialog.isVisible():
            self.result_dialog.show()

    def capture_audio(self, duration=10, filename="audio_capture.wav"):
        """Capture audio from the microphone"""
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1
        fs = 44100  # Record at 44100 samples per second
        p = pyaudio.PyAudio()

        print("Recording audio...")
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []  # Initialize array to store frames

        # Store data in chunks for the specified duration
        for _ in range(0, int(fs / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        print("Audio recording complete.")
        return filename

    def transcribe_audio(self, audio_path):
        """Transcribe audio using OpenAI's Whisper model"""
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                return transcription
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""

class CaptureWorkerSignals(QObject):
    finished = Signal(str, str)  # Signals for result and image_path
    error = Signal(str)

class CaptureAndSendWorker(QRunnable):
    def __init__(self, camera, interval):
        super().__init__()
        self.camera = camera
        self.interval = interval
        self.signals = CaptureWorkerSignals()

    def run(self):
        try:
            # Capture frame
            path = self.camera.capture_frame_save()
            if not path:
                self.signals.error.emit("Failed to capture frame")
                return

            # Capture audio
            audio_path = self.camera.capture_audio(self.interval)
            
            # Transcribe audio
            transcription = self.camera.transcribe_audio(audio_path)
            
            if transcription:
                from pygpt_net.core.vision.analyzer import Analyzer
                ctx = CtxItem()
                prompt = f"""
                You are an AI assistant tasked with analyzing classroom sentiment. 
                Based on the combination of visual cues (e.g., facial expressions, body language) and conversational context (e.g., tone, participation), 
                detect the overall classroom sentiment and provide reasoning for your analysis.

                Instructions:
                1. Analyze sentiment based on descriptive inputs of what students look like and say
                2. Identify emotions such as engagement, confusion, boredom, or frustration
                3. Provide a clear and concise explanation of how the visual and conversational data lead to the detected sentiment
                4. Return your analysis in the following JSON format:

                {{
                    "sentiment_score": float,  // Score from -5.0 (most negative) to 5.0 (most positive)
                    "sentiment_label": string, // Brief label describing the sentiment (e.g., "highly engaged", "moderately bored")
                    "reasoning": string,       // Brief explanation of your analysis
                    "transcript_cues": "{transcription}",  // Transcript cues
                    "visual_cues": [          // List of observed visual indicators
                        string,
                        ...
                    ],
                    "recommendations": [      // List of suggested improvements
                        string,
                        ...
                    ]
                }}

                Note: sentiment_score must be between -5.0 and 5.0, where:
                * 5.0: Extremely engaged and enthusiastic
                * 3.0: Actively engaged
                * 1.0: Slightly engaged
                * 0.0: Neutral
                * -1.0: Slightly disengaged
                * -3.0: Notably disengaged
                * -5.0: Completely disengaged or disruptive
                """

                analyzer = Analyzer(self.camera.window)
                response = analyzer.from_camera(ctx, prompt)
                
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                else:
                    json_str = response
                
                # Emit the result
                self.signals.finished.emit(json_str, path)
            
        except Exception as e:
            self.signals.error.emit(str(e))

