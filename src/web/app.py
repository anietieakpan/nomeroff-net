# src/web/app.py
from flask import Flask, render_template, Response, jsonify
import cv2
import threading
from queue import Queue
import logging

from ..detector.plate_detector import LicensePlateDetector

logger = logging.getLogger(__name__)

app = Flask(__name__)

class VideoCamera:
    def __init__(self, detector: LicensePlateDetector):
        self.detector = detector
        self.video = cv2.VideoCapture(0)
        self.frame_queue = Queue(maxsize=10)
        self.is_running = True
        
        # Start frame capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.start()

    def _capture_frames(self):
        """Continuously capture frames in a separate thread"""
        while self.is_running:
            success, frame = self.video.read()
            if success:
                if not self.frame_queue.full():
                    visualization, _ = self.detector.process_frame(frame)
                    self.frame_queue.put(visualization)

    def get_frame(self):
        """Get the latest processed frame"""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        return None

    def __del__(self):
        self.is_running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join()
        self.video.release()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

def gen(camera):
    """Video streaming generator function"""
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen(app.camera),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/statistics')
def statistics():
    """Get current detection statistics"""
    try:
        stats = app.detector.performance.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500