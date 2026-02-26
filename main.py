from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QComboBox
import sys
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import json
import os

# 导入你的UI文件
from pyui2 import Ui_MainWindow


# 初始化MediaPipe手部检测
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 配置文件路径
CONFIG_FILE = "gesture_config.json"

# 预设可选按键（下拉框选项）
KEY_OPTIONS = [
    "无操作", "上一首", "下一首", "播放/暂停", "音量+", "音量-", "静音", "切换窗口",
    "快进", "快退", "最小化窗口", "截屏"  # 新增的4个功能选项
]

# 按键对应的操作函数
KEY_ACTIONS = {
    "无操作": lambda: None,
    "上一首": lambda: pyautogui.press("prevtrack"),
    "下一首": lambda: pyautogui.press("nexttrack"),
    "播放/暂停": lambda: pyautogui.press("playpause"),
    "音量+": lambda: pyautogui.press("volumeup"),
    "音量-": lambda: pyautogui.press("volumedown"),
    "静音": lambda: pyautogui.press("volumemute"),
    "切换窗口": lambda: pyautogui.hotkey("alt", "tab"),
    # 新增的4个功能逻辑
    "快进": lambda: pyautogui.press("end"),          # 快进（按右方向键）
    "快退": lambda: pyautogui.press("home"),           # 快退（按左方向键）
    "最小化窗口": lambda: pyautogui.hotkey("win", "d"), # 最小化所有窗口（Win+D）
    "截屏": lambda: pyautogui.hotkey("win", "shift", "s") # 系统截屏（Win+Shift+S）
}

# 手势识别与UI下拉框的对应关系（匹配你的objectName）
GESTURE_UI_MAP = {
    "THUMB_UP": "combo_thumb_up",  # 点赞手势
    "FIST": "combo_fist",  # 拳头手势
    "OK": "combo_ok",  # OK手势
    "PEACE": "combo_peace",  # 剪刀手（双指）手势
    "ONE": "combo_one"  # 数字1手势
}


# 摄像头检测线程（移除画面相关逻辑）
def get_x_distance(p1, p2):
    return abs(p1.x - p2.x)


class CameraThread(QThread):
    gesture_signal = pyqtSignal(str)  # 仅传递识别的手势（可选）

    def __init__(self, gesture_key_map):
        super().__init__()
        self.is_running = False
        self.last_action_time = 0
        self.ACTION_COOLDOWN = 1.5  # 操作冷却时间（秒）
        self.gesture_key_map = gesture_key_map  # 接收自定义手势-按键映射

        # 初始化MediaPipe手部检测
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def run(self):
        self.is_running = True
        cap = cv2.VideoCapture(0)  # 打开默认摄像头
        # 仅保留摄像头采集，不处理画面显示
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.is_running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 镜像翻转、转换颜色空间（仅用于识别，不显示）
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            # 检测手部关键点
            results = self.hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            current_gesture = "NONE"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 仅识别手势，不绘制关键点（也可保留绘制，不影响）
                    current_gesture = self.recognize_gesture(hand_landmarks)

                    # 执行自定义映射的操作（带冷却）
                    current_time = time.time()
                    if (current_gesture in self.gesture_key_map and
                            current_gesture != "NONE" and
                            (current_time - self.last_action_time) > self.ACTION_COOLDOWN):
                        key_action = self.gesture_key_map[current_gesture]
                        KEY_ACTIONS[key_action]()
                        self.last_action_time = current_time
                        print(f"执行手势[{current_gesture}] → 按键[{key_action}]")

            # 可选：传递手势信息到UI（如需显示当前手势可保留）
            self.gesture_signal.emit(current_gesture)

            # 控制帧率
            time.sleep(0.01)

        # 释放资源
        cap.release()
        self.hands.close()

    def stop(self):
        self.is_running = False
        self.wait()

    # 手势识别核心逻辑（匹配你的5个手势）
    def recognize_gesture(self, hand_landmarks):
        landmarks = hand_landmarks.landmark

        # 1. OK手势：拇指指尖和食指指尖距离很近
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        ok_distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
        if ok_distance < 0.05:
            return "OK"

        # 2. 点赞（THUMB_UP）：拇指竖起，其他手指弯曲
        thumb_up = (landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y)
        fingers_closed = all([
            landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y,
            landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y
        ])
        if thumb_up and fingers_closed:
            return "THUMB_UP"

        # 3. 拳头（FIST）：所有手指都弯曲
        fist = all([
            landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
            landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
            landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y,
            landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_MCP].y,
            landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_MCP].y,

        ])
        if fist:
            return "FIST"

        # 4. 剪刀手（PEACE）：食指+中指竖起，其他弯曲
        peace = (
            # 核心：食指和中指指尖横向距离<0.03（交叉重叠）
                get_x_distance(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                               landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) < 0.03 and
                # 食指、中指伸直（交叉时需要伸直）
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[
                    mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                # 无名指、小指弯曲（排除剪刀手干扰）
                landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[
                    mp_hands.HandLandmark.RING_FINGER_PIP].y and
                landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y
        )
        if peace:
            return "PEACE"
        # 5. 数字1（ONE）：仅食指竖起
        one = (
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[
            mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[
                    mp_hands.HandLandmark.RING_FINGER_PIP].y and
                landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y
        )
        if one:
            return "ONE"

        return "NONE"


# 主UI窗口类（移除所有画面显示相关代码）
class GestureControlWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("手势控制界面")
        self.setFixedSize(500, 700)  # 可根据你的UI调整

        # 1. 初始化下拉框
        self.init_comboboxes()

        # 2. 加载历史配置
        self.gesture_key_map = self.load_config()
        self.set_comboboxes_from_config()

        # 3. 初始化摄像头线程
        self.camera_thread = None
        self.is_camera_running = False

        # 4. 绑定按钮事件（确保按钮objectName匹配）
        self.pushButton_save.clicked.connect(self.save_config)  # SAVE按钮
        self.pushButton_open_close.clicked.connect(self.toggle_camera)  # OPEN/CLOSE按钮

    # 初始化所有下拉框，添加预设按键选项
    def init_comboboxes(self):
        for combo_name in GESTURE_UI_MAP.values():
            combo = getattr(self, combo_name, None)
            if combo and isinstance(combo, QComboBox):
                combo.clear()
                combo.addItems(KEY_OPTIONS)

    # 从配置文件加载手势-按键映射
    def load_config(self):
        default_map = {
            "THUMB_UP": "上一首",
            "FIST": "下一首",
            "OK": "播放",
            "PEACE": "音量加",
            "ONE": "音量减"
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return default_map
        return default_map

    # 根据配置设置下拉框选中项
    def set_comboboxes_from_config(self):
        for gesture, combo_name in GESTURE_UI_MAP.items():
            combo = getattr(self, combo_name, None)
            if combo and isinstance(combo, QComboBox):
                key = self.gesture_key_map.get(gesture, "无操作")
                index = combo.findText(key)
                if index >= 0:
                    combo.setCurrentIndex(index)

    # 保存当前下拉框的配置到文件
    def save_config(self):
        # 从下拉框读取当前映射
        for gesture, combo_name in GESTURE_UI_MAP.items():
            combo = getattr(self, combo_name, None)
            if combo and isinstance(combo, QComboBox):
                self.gesture_key_map[gesture] = combo.currentText()

        # 写入配置文件
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.gesture_key_map, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "保存成功", "手势-按键配置已保存！")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"配置保存出错：{str(e)}")

    # 启动/停止摄像头（仅控制识别，不显示画面）
    def toggle_camera(self):
        if not self.is_camera_running:
            # 启动摄像头线程（仅识别，不显示画面）
            self.camera_thread = CameraThread(self.gesture_key_map)
            # 可选：绑定手势信号（如需在UI显示当前手势则保留，否则可删除）
            self.camera_thread.gesture_signal.connect(self.update_gesture_label)
            self.camera_thread.start()
            self.pushButton_open_close.setText("CLOSE")
            self.is_camera_running = True
        else:
            # 停止摄像头线程
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread = None
            self.pushButton_open_close.setText("OPEN")
            self.is_camera_running = False

    # 可选：更新当前手势显示（如需在UI显示则保留，否则可删除）
    def update_gesture_label(self, gesture):
        if hasattr(self, "label_gesture"):
            gesture_name_map = {
                "THUMB_UP": "点赞",
                "FIST": "拳头",
                "OK": "OK",
                "PEACE": "剪刀手",
                "ONE": "数字1",
                "NONE": "无"
            }
            self.label_gesture.setText(f"当前手势：{gesture_name_map.get(gesture, '无')}")

    # 窗口关闭时清理资源
    def closeEvent(self, event):
        if self.is_camera_running and self.camera_thread:
            self.camera_thread.stop()
        event.accept()


# 程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureControlWindow()
    window.show()
    sys.exit(app.exec())