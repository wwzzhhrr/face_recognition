import cv2
import numpy as np
import dlib
from typing import Tuple, Optional

class FaceAligner:
    def __init__(self, predictor_path: str = None, desired_face_width: int = 256, desired_face_height: int = 256):
        """
        人脸对齐器，用于检测人脸并将其居中裁切到指定位置
        
        Args:
            predictor_path: dlib人脸关键点预测器模型路径 (可选，如果不提供则使用简单的人脸检测)
            desired_face_width: 输出人脸图像的宽度
            desired_face_height: 输出人脸图像的高度
        """
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height
        
        # 初始化人脸检测器
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 如果提供了预测器路径，则使用dlib进行更精确的人脸对齐
        self.use_landmarks = predictor_path is not None
        if self.use_landmarks:
            try:
                self.detector = dlib.get_frontal_face_detector()
                self.predictor = dlib.shape_predictor(predictor_path)
            except Exception as e:
                print(f"Warning: Could not load dlib predictor: {e}")
                print("Falling back to basic face detection")
                self.use_landmarks = False
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        检测图像中的人脸
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            人脸边界框 (x, y, w, h) 或 None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # 返回最大的人脸，并验证其有效性
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # 验证人脸框是否在图像范围内且有效
        if (x >= 0 and y >= 0 and 
            x + w <= image.shape[1] and y + h <= image.shape[0] and 
            w > 0 and h > 0):
            return tuple(largest_face)
        else:
            print(f"Warning: Invalid face detection result: ({x}, {y}, {w}, {h})")
            return None
    
    def get_face_landmarks(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        获取人脸关键点
        
        Args:
            image: 输入图像
            face_rect: 人脸边界框 (x, y, w, h)
            
        Returns:
            关键点坐标数组 [(x1, y1), (x2, y2), ...] 或 None
        """
        if not self.use_landmarks:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x, y, w, h = face_rect
        
        # 转换为dlib格式的矩形
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        
        # 获取关键点
        landmarks = self.predictor(gray, dlib_rect)
        
        # 转换为numpy数组
        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(landmarks.num_parts)])
        return points
    
    def align_face_with_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        使用关键点对齐人脸
        
        Args:
            image: 输入图像
            landmarks: 人脸关键点
            
        Returns:
            对齐后的人脸图像
        """
        # 使用眼睛关键点进行对齐
        left_eye_pts = landmarks[36:42]  # 左眼关键点
        right_eye_pts = landmarks[42:48]  # 右眼关键点
        
        # 计算眼睛中心点
        left_eye_center = np.mean(left_eye_pts, axis=0)
        right_eye_center = np.mean(right_eye_pts, axis=0)
        
        # 计算眼睛之间的角度
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 计算眼睛之间的距离
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        
        # 期望的眼睛距离（基于输出图像大小）
        desired_eye_distance = self.desired_face_width * 0.35
        scale = desired_eye_distance / eye_distance
        
        # 计算眼睛中点
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                      (left_eye_center[1] + right_eye_center[1]) // 2)
        
        # 创建变换矩阵
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # 调整变换矩阵以将眼睛中心移动到图像中心
        tx = self.desired_face_width * 0.5
        ty = self.desired_face_height * 0.4  # 眼睛稍微偏上
        M[0, 2] += (tx - eyes_center[0])
        M[1, 2] += (ty - eyes_center[1])
        
        # 应用变换
        aligned_face = cv2.warpAffine(image, M, (self.desired_face_width, self.desired_face_height))
        return aligned_face
    
    def align_face_simple(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        简单的人脸对齐（仅基于人脸边界框）
        
        Args:
            image: 输入图像
            face_rect: 人脸边界框 (x, y, w, h)
            
        Returns:
            对齐后的人脸图像
        """
        x, y, w, h = face_rect
        
        # 扩展边界框以包含更多面部区域
        padding = int(max(w, h) * 0.3)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # 安全检查：确保裁切区域有效
        if x1 >= x2 or y1 >= y2:
            print(f"Warning: Invalid crop region ({x1}, {y1}, {x2}, {y2}), using original face rect")
            # 回退到原始人脸区域
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
        
        # 再次检查裁切区域
        if x1 >= x2 or y1 >= y2:
            print(f"Error: Still invalid crop region, creating default image")
            # 创建一个默认的灰色图像
            return np.full((self.desired_face_height, self.desired_face_width, 3), 128, dtype=np.uint8)
        
        # 裁切人脸区域
        face_crop = image[y1:y2, x1:x2]
        
        # 检查裁切结果是否为空
        if face_crop.size == 0:
            print(f"Warning: Empty face crop, creating default image")
            return np.full((self.desired_face_height, self.desired_face_width, 3), 128, dtype=np.uint8)
        
        # 调整大小到期望尺寸
        aligned_face = cv2.resize(face_crop, (self.desired_face_width, self.desired_face_height))
        return aligned_face
    
    def process_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        处理单张图像，检测并对齐人脸
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            对齐后的人脸图像或None（如果未检测到人脸）
        """
        # 检测人脸
        face_rect = self.detect_face(image)
        if face_rect is None:
            return None
        
        # 如果使用关键点对齐
        if self.use_landmarks:
            landmarks = self.get_face_landmarks(image, face_rect)
            if landmarks is not None:
                return self.align_face_with_landmarks(image, landmarks)
        
        # 使用简单对齐
        return self.align_face_simple(image, face_rect)
    
    def process_image_file(self, input_path: str, output_path: str = None) -> Optional[np.ndarray]:
        """
        处理图像文件
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径（可选）
            
        Returns:
            对齐后的人脸图像或None
        """
        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not load image from {input_path}")
            return None
        
        # 处理图像
        aligned_face = self.process_image(image)
        
        # 保存结果
        if aligned_face is not None and output_path is not None:
            cv2.imwrite(output_path, aligned_face)
            print(f"Aligned face saved to {output_path}")
        
        return aligned_face
    
    def batch_process(self, input_dir: str, output_dir: str, file_extension: str = ".jpg") -> int:
        """
        批量处理图像文件夹
        
        Args:
            input_dir: 输入文件夹路径
            output_dir: 输出文件夹路径
            file_extension: 文件扩展名
            
        Returns:
            成功处理的图像数量
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        processed_count = 0
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(file_extension.lower()):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"aligned_{filename}")
                
                aligned_face = self.process_image_file(input_path, output_path)
                if aligned_face is not None:
                    processed_count += 1
                    print(f"Processed: {filename}")
                else:
                    print(f"Failed to process: {filename}")
        
        print(f"Successfully processed {processed_count} images")
        return processed_count


# 使用示例
if __name__ == "__main__":
    # 创建人脸对齐器实例
    # 如果有dlib的shape_predictor_68_face_landmarks.dat文件，可以传入路径获得更好的对齐效果
    aligner = FaceAligner(desired_face_width=128, desired_face_height=128)
    
    # 处理单张图像
    # aligned_face = aligner.process_image_file("input.jpg", "output_aligned.jpg")
    
    # 批量处理
    # aligner.batch_process("input_folder", "output_folder")