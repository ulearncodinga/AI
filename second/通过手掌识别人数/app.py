import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# 初始化MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Flask应用初始化
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局变量
hand_count = 0


class HandDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_hands(self, frame):
        """检测手部并返回举手数量"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        raised_hands = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 获取手腕和指尖的关键点
                wrist = hand_landmarks.landmark[0]
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]

                # 计算指尖相对于手腕的y坐标
                tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
                avg_tip_y = sum(tip.y for tip in tips) / len(tips)

                # 如果指尖在手腕上方，认为是举手
                if avg_tip_y < wrist.y - 0.1:
                    raised_hands += 1

        return raised_hands, results


detector = HandDetector()


@app.route('/')
def index():
    """主页路由"""
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """处理来自前端的视频帧"""
    try:
        data = request.json
        image_data = data['image']

        # 解码base64图像
        image_data = image_data.split(',')[1]  # 移除data:image/jpeg;base64,前缀
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # 转换为OpenCV格式
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 检测举手
        raised_hands, results = detector.detect_hands(frame)
        global hand_count
        hand_count = raised_hands

        # 在帧上绘制检测结果
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

        # 添加举手数量显示（使用PIL支持中文）
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # 使用支持中文的字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 40)
        except:
            font = ImageFont.load_default()

        # 绘制中文文本
        text = f'举手人数: {raised_hands}'
        draw.text((10, 10), text, font=font, fill=(0, 255, 0))

        # 转换回OpenCV格式
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 编码回base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'count': raised_hands,
            'processed_image': f'data:image/jpeg;base64,{processed_image}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@socketio.on('get_hand_count')
def handle_get_hand_count():
    """获取举手数量"""
    global hand_count
    emit('hand_count_update', {'count': hand_count})


if __name__ == '__main__':
    # 启动Flask应用，添加允许不安全Werkzeug服务器的参数
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)