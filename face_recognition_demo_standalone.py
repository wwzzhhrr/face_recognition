import os
import sys
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from collections import Counter, defaultdict

# 导入自定义模块
from PCA import PCA
from SVD_decomposer import SVDDecomposer
from face_aligner import FaceAligner


class FaceRecognitionUtils:
    """人脸识别工具类，包含图像处理和评估功能"""
    
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size
        self.face_aligner = FaceAligner(desired_face_width=image_size[0], desired_face_height=image_size[1])
        
    def load_image_simple(self, path):
        """简单图像加载"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, self.image_size)
        return img.flatten()
    
    def load_image_with_alignment(self, path, save_processed=False, save_path=None):
        """带人脸对齐的图像加载"""
        img = cv2.imread(path)
        if img is None:
            return None
        
        # 使用FaceAligner进行人脸检测和对齐
        aligned_face = self.face_aligner.process_image(img)
        
        if aligned_face is None:
            # 如果没有检测到人脸，使用简单的灰度转换和缩放
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, self.image_size)
            processed_image = img_resized
        else:
            # 转换为灰度图像（如果还不是）
            if len(aligned_face.shape) == 3:
                aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            processed_image = aligned_face
        
        # 保存处理后的图像（如果需要）
        if save_processed and save_path:
            cv2.imwrite(save_path, processed_image)
        
        return processed_image.flatten()
    
    def load_dataset(self, data_dir, use_alignment=True, save_processed=False, processed_dir=None, prefix="processed_"):
        """通用数据集加载方法"""
        if save_processed and processed_dir and not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        images, labels, filenames = [], [], []
        failed_count = 0
        
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith(".jpg"):
                file_path = os.path.join(data_dir, filename)
                
                # 根据文件名提取标签（如果可能）
                label = ''.join(filter(str.isalpha, filename)) if any(c.isalpha() for c in filename) else filename
                
                # 加载图像
                if use_alignment:
                    save_path = os.path.join(processed_dir, f"{prefix}{filename}") if save_processed and processed_dir else None
                    img_vector = self.load_image_with_alignment(file_path, save_processed, save_path)
                else:
                    img_vector = self.load_image_simple(file_path)
                
                if img_vector is not None:
                    images.append(img_vector)
                    labels.append(label)
                    filenames.append(filename)
                else:
                    failed_count += 1
        
        return np.array(images), labels, filenames, failed_count
    
    def split_training_data(self, data_dir, train_count=3, use_alignment=True):
        """分割训练数据为训练集和测试集"""
        person_images = defaultdict(list)
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith(".jpg"):
                person = ''.join(filter(str.isalpha, filename))
                person_images[person].append(os.path.join(data_dir, filename))
        
        X_train, y_train = [], []
        X_test, y_test = [], []
        failed_count = 0
        
        for person, files in person_images.items():
            files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            train_files = files[:train_count]
            test_files = files[train_count:]
            
            # 处理训练图像
            for path in train_files:
                if use_alignment:
                    img_vector = self.load_image_with_alignment(path)
                else:
                    img_vector = self.load_image_simple(path)
                
                if img_vector is not None:
                    X_train.append(img_vector)
                    y_train.append(person)
                else:
                    failed_count += 1
            
            # 处理测试图像
            for path in test_files:
                if use_alignment:
                    img_vector = self.load_image_with_alignment(path)
                else:
                    img_vector = self.load_image_simple(path)
                
                if img_vector is not None:
                    X_test.append(img_vector)
                    y_test.append(person)
                else:
                    failed_count += 1
        
        return np.array(X_train), y_train, np.array(X_test), y_test, failed_count
    
    def calculate_anomaly_threshold(self, X_train_proj, y_train, percentile=95):
        """计算异常检测阈值"""
        # 计算每个训练样本到其他同类样本的最小距离
        intra_class_distances = []
        
        unique_labels = list(set(y_train))
        for label in unique_labels:
            same_class_indices = [i for i, y in enumerate(y_train) if y == label]
            
            if len(same_class_indices) > 1:
                for i in same_class_indices:
                    other_same_class = [j for j in same_class_indices if j != i]
                    distances_to_same_class = [np.linalg.norm(X_train_proj[i] - X_train_proj[j]) 
                                             for j in other_same_class]
                    intra_class_distances.extend(distances_to_same_class)
        
        # 计算每个训练样本到所有训练样本的最小距离
        min_distances = []
        for i in range(len(X_train_proj)):
            distances = [np.linalg.norm(X_train_proj[i] - X_train_proj[j]) 
                        for j in range(len(X_train_proj)) if i != j]
            min_distances.append(min(distances))
        
        all_distances = intra_class_distances + min_distances
        threshold = np.percentile(all_distances, percentile)
        
        return threshold, {
            'intra_class_mean': np.mean(intra_class_distances) if intra_class_distances else 0,
            'intra_class_std': np.std(intra_class_distances) if intra_class_distances else 0,
            'min_distance_mean': np.mean(min_distances),
            'min_distance_std': np.std(min_distances)
        }
    
    def predict_with_anomaly_detection(self, test_sample, X_train_proj, y_train, threshold):
        """带异常检测的预测方法"""
        distances = np.linalg.norm(X_train_proj - test_sample, axis=1)
        min_distance = np.min(distances)
        min_index = np.argmin(distances)
        
        is_anomaly = min_distance > threshold
        predicted_label = y_train[min_index] if not is_anomaly else "UNKNOWN"
        
        return predicted_label, min_distance, is_anomaly
    
    def evaluate_with_anomaly_detection(self, X_test_proj, y_test, X_train_proj, y_train, threshold, 
                                       dataset_name="Test", expected_anomalies=False, verbose=True):
        """带异常检测的评估方法"""
        correct = 0
        anomaly_correct = 0
        total_anomalies = 0
        results = []
        
        for i in range(len(X_test_proj)):
            pred, distance, is_anomaly = self.predict_with_anomaly_detection(
                X_test_proj[i], X_train_proj, y_train, threshold
            )
            actual = y_test[i] if not expected_anomalies else "UNKNOWN"
            
            if not expected_anomalies:
                # 正常测试集评估
                is_correct = pred == actual
                if is_correct:
                    correct += 1
                
                result = {
                    'index': i + 1,
                    'predicted': pred,
                    'actual': actual,
                    'distance': distance,
                    'is_anomaly': is_anomaly,
                    'correct': is_correct
                }
                results.append(result)
                
                if verbose:
                    status = "✓" if is_correct else "✗"
                    anomaly_flag = "[异常]" if is_anomaly else ""
                    print(f"[{dataset_name} {i+1:2d}] {status} 预测: {pred:8s} | 实际: {actual:8s} | 距离: {distance:.4f} {anomaly_flag}")
            else:
                # 误导项测试评估
                total_anomalies += 1
                is_correct = is_anomaly
                if is_correct:
                    anomaly_correct += 1
                
                result = {
                    'index': i + 1,
                    'filename': y_test[i],
                    'predicted': pred,
                    'distance': distance,
                    'is_anomaly': is_anomaly,
                    'anomaly_detected': is_correct
                }
                results.append(result)
                
                if verbose:
                    status = "✓" if is_correct else "✗"
                    print(f"[{dataset_name} {i+1:2d}] {status} 文件: {y_test[i]:12s} | 预测: {pred:8s} | 距离: {distance:.4f} | 异常检测: {'成功' if is_correct else '失败'}")
        
        if not expected_anomalies:
            accuracy = correct / len(X_test_proj) if len(X_test_proj) > 0 else 0
            if verbose:
                print(f"\n{dataset_name} 准确率: {accuracy:.2%} ({correct}/{len(X_test_proj)})")
            return accuracy, 0, results
        else:
            anomaly_accuracy = anomaly_correct / total_anomalies if total_anomalies > 0 else 0
            if verbose:
                print(f"\n{dataset_name} 异常检测准确率: {anomaly_accuracy:.2%} ({anomaly_correct}/{total_anomalies})")
            return 0, anomaly_accuracy, results

class DynamicFaceRecognitionSystem:
    """
    动态人脸识别系统
    支持实时添加新人员、重新训练模型和人脸识别
    整合了main_v3.py的数据处理逻辑
    """
    
    def __init__(self, image_size=(64, 64), n_components=20):
        self.image_size = image_size
        self.n_components = n_components
        self.training_data = []
        self.training_labels = []
        self.pca = None
        self.mean_face = None
        self.projected_data = None
        self.persons = []
        self.recognition_threshold = 0.8
        self.anomaly_threshold = None
        
        # 初始化工具类
        self.utils = FaceRecognitionUtils(image_size=image_size)
        
        # 创建必要的目录
        os.makedirs("live_captured_images", exist_ok=True)
        os.makedirs("processed_training_images", exist_ok=True)
        os.makedirs("processed_challenge_images", exist_ok=True)
        os.makedirs("processed_misleading_images", exist_ok=True)
        
        print("\n" + "="*60)
        print("🎭 动态人脸识别系统 - 演示版本")
        print("="*60)
        print("基于PCA的人脸识别，支持动态学习新人员")
        print("整合了完整的数据处理和异常检测功能")
        print("适合观众展示和互动演示")
        print("="*60 + "\n")
        
    def load_initial_data_advanced(self, train_dir="training_images", train_count=3):
        """
        使用main_v3.py的逻辑加载初始训练数据
        支持数据分割和人脸对齐处理
        """
        print("📂 正在使用高级方法加载初始训练数据...")
        
        if not os.path.exists(train_dir):
            print(f"❌ 训练数据目录不存在: {train_dir}")
            return False
        
        try:
            # 使用工具类分割训练数据
            X_train, y_train, X_test_train, y_test_train, failed_train = self.utils.split_training_data(
                train_dir, train_count=train_count, use_alignment=True
            )
            
            # 保存处理后的训练图像
            self.utils.load_dataset(
                train_dir, use_alignment=True, save_processed=True, 
                processed_dir="processed_training_images", prefix="processed_"
            )
            
            if len(X_train) == 0:
                print("❌ 没有成功加载任何训练数据")
                return False
            
            # 设置训练数据
            self.training_data = X_train
            self.training_labels = y_train
            self.persons = sorted(set(y_train))
            
            print(f"✅ 成功加载训练数据:")
            print(f"   训练图像: {len(X_train)} 张")
            print(f"   测试图像: {len(X_test_train)} 张")
            print(f"   处理失败: {failed_train} 张")
            print(f"   检测到人员: {self.persons}")
            
            # 训练模型
            if self.train_model_advanced():
                # 保存测试数据用于后续评估
                self.test_data = X_test_train
                self.test_labels = y_test_train
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ 加载数据时出错: {e}")
            return False
    
    def train_model_advanced(self):
        """
        使用main_v3.py的逻辑训练模型
        包括异常检测阈值计算
        """
        print("🔧 正在使用高级方法训练PCA模型...")
        
        if len(self.training_data) == 0:
            print("❌ 没有训练数据！")
            return False
            
        try:
            # 训练PCA
            actual_components = min(self.n_components, len(self.training_data)-1)
            self.pca = PCA(n_components=actual_components)
            self.pca.fit(self.training_data)
            
            # 计算投影数据
            self.projected_data = self.pca.transform(self.training_data)
            
            # 计算异常检测阈值
            self.anomaly_threshold, stats = self.utils.calculate_anomaly_threshold(
                self.projected_data, self.training_labels, percentile=95
            )
            
            print(f"✅ 模型训练完成！")
            print(f"   使用主成分数量: {self.pca.n_components}")
            print(f"   异常检测阈值: {self.anomaly_threshold:.4f}")
            print(f"   类内距离统计: 均值={stats['intra_class_mean']:.4f}, 标准差={stats['intra_class_std']:.4f}")
            print(f"   最小距离统计: 均值={stats['min_distance_mean']:.4f}, 标准差={stats['min_distance_std']:.4f}")
            print(f"📊 当前训练集: {len(set(self.training_labels))} 个人，共 {len(self.training_data)} 张图像")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            return False
    
    def comprehensive_evaluation(self):
        """
        使用main_v3.py的逻辑进行全面评估
        包括训练集测试、挑战集和误导项检测
        """
        print("\n" + "="*60)
        print("🧪 全面模型评估")
        print("="*60)
        
        if self.pca is None or self.anomaly_threshold is None:
            print("❌ 模型尚未训练或缺少异常检测阈值！")
            return
        
        # 评估训练集测试部分
        if hasattr(self, 'test_data') and len(self.test_data) > 0:
            print("\n📊 训练集测试结果:")
            X_test_proj = self.pca.transform(self.test_data)
            acc_train, _, _ = self.utils.evaluate_with_anomaly_detection(
                X_test_proj, self.test_labels, self.projected_data, self.training_labels, 
                self.anomaly_threshold, "训练集测试", expected_anomalies=False
            )
        else:
            acc_train = 0
            print("\n⚠️ 跳过训练集测试（无数据）")
        
        # 评估挑战集
        challenge_dir = "challenge_images"
        if os.path.exists(challenge_dir):
            print("\n🎯 挑战集测试结果:")
            X_challenge, y_challenge, _, failed_challenge = self.utils.load_dataset(
                challenge_dir, use_alignment=True, save_processed=True,
                processed_dir="processed_challenge_images", prefix="processed_challenge_"
            )
            
            if len(X_challenge) > 0:
                X_challenge_proj = self.pca.transform(X_challenge)
                acc_challenge, _, _ = self.utils.evaluate_with_anomaly_detection(
                    X_challenge_proj, y_challenge, self.projected_data, self.training_labels,
                    self.anomaly_threshold, "挑战集", expected_anomalies=False
                )
            else:
                acc_challenge = 0
                print("⚠️ 挑战集为空")
        else:
            acc_challenge = 0
            print("\n⚠️ 跳过挑战集测试（目录不存在）")
        
        # 评估误导项检测
        misleading_dir = "misleading_images"
        if os.path.exists(misleading_dir):
            print("\n🚫 误导项检测结果:")
            X_misleading, _, misleading_filenames, failed_misleading = self.utils.load_dataset(
                misleading_dir, use_alignment=True, save_processed=True,
                processed_dir="processed_misleading_images", prefix="processed_misleading_"
            )
            
            if len(X_misleading) > 0:
                X_misleading_proj = self.pca.transform(X_misleading)
                _, anomaly_acc, _ = self.utils.evaluate_with_anomaly_detection(
                    X_misleading_proj, misleading_filenames, self.projected_data, self.training_labels,
                    self.anomaly_threshold, "误导项检测", expected_anomalies=True
                )
            else:
                anomaly_acc = 0
                print("⚠️ 误导项为空")
        else:
            anomaly_acc = 0
            print("\n⚠️ 跳过误导项测试（目录不存在）")
        
        # 最终总结
        print("\n" + "="*60)
        print("📋 最终评估结果")
        print("="*60)
        print(f"训练集测试准确率: {acc_train:.2%}")
        print(f"挑战集准确率: {acc_challenge:.2%}")
        print(f"误导项检测准确率: {anomaly_acc:.2%}")
        print(f"图像尺寸: {self.image_size}")
        print(f"PCA主成分: {self.pca.n_components}")
        print(f"异常检测阈值: {self.anomaly_threshold:.4f}")
        print(f"\n💾 处理后的图像已保存:")
        print(f"   训练图像: processed_training_images/")
        print(f"   挑战图像: processed_challenge_images/")
        print(f"   误导项图像: processed_misleading_images/")
        print(f"\n🔧 功能特性:")
        print(f"   • 人脸检测和对齐处理")
        print(f"   • 基于PCA的降维")
        print(f"   • 异常检测和误导项识别")
        print(f"   • 详细的测试结果输出")
        print(f"   • 动态添加新人员")
        print("="*60)

    def add_new_person_data(self, person_name, num_photos=5):
        """为新人员添加训练数据"""
        print(f"\n开始为 '{person_name}' 拍摄训练照片...")
        print(f"将拍摄 {num_photos} 张照片用于训练")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            print("请检查摄像头连接或权限设置")
            return False
            
        photos_taken = 0
        new_data = []
        
        print("\n拍照说明:")
        print("• 请面向摄像头，保持良好光线")
        print("• 按 [空格键] 拍照")
        print("• 按 [ESC键] 退出")
        print("• 建议在不同角度拍摄以提高识别准确率")
        
        input("\n按 [Enter] 键开始拍照...")
        
        while photos_taken < num_photos:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
                
            # 显示当前帧
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Photos: {photos_taken}/{num_photos}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Adding: {person_name}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display_frame, "SPACE: Capture, ESC: Exit", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f'Adding Training Data for {person_name}', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # 空格键
                # 使用人脸对齐处理
                aligned_face = self.utils.face_aligner.process_image(frame)
                
                if aligned_face is not None:
                    # 转换为灰度图像
                    if len(aligned_face.shape) == 3:
                        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
                    processed_image = aligned_face
                else:
                    # 如果没有检测到人脸，使用简单处理
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    processed_image = cv2.resize(gray, self.image_size)
                
                # 保存图像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{person_name}_{timestamp}_{photos_taken+1}.jpg"
                save_path = os.path.join("live_captured_images", filename)
                
                cv2.imwrite(save_path, processed_image)
                
                # 添加到训练数据
                flattened = processed_image.flatten()
                new_data.append(flattened)
                
                photos_taken += 1
                print(f"拍摄第 {photos_taken} 张照片: {filename}")
                
                # 短暂暂停显示拍摄效果
                cv2.putText(display_frame, "CAPTURED!", (200, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow(f'Adding Training Data for {person_name}', display_frame)
                cv2.waitKey(500)
                
            elif key == 27:  # ESC键
                print("\n用户取消拍照")
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        if new_data:
            # 添加到训练集
            if len(self.training_data) == 0:
                self.training_data = np.array(new_data)
            else:
                self.training_data = np.vstack([self.training_data, new_data])
            
            self.training_labels.extend([person_name] * len(new_data))
            
            # 更新人员列表
            if person_name not in self.persons:
                self.persons.append(person_name)
                
            print(f"\n成功添加 {len(new_data)} 张 '{person_name}' 的照片")
            
            # 重新训练模型
            if self.train_model_advanced():
                print(f"'{person_name}' 已加入识别系统！")
                return True
            else:
                print("重新训练失败")
                return False
        else:
            print("没有添加任何照片")
            return False
    
    def recognize_face(self, show_confidence=True):
        """
        人脸识别（默认使用异常检测）
        """
        if self.pca is None or self.anomaly_threshold is None:
            print("错误: 模型尚未训练或缺少异常检测阈值！请先加载数据")
            return None
            
        print("\n启动人脸识别模式...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return None
            
        print("\n识别说明:")
        print("• 请面向摄像头")
        print("• 按 [空格键] 进行识别")
        print("• 按 [ESC键] 退出识别模式")
        print("• 支持异常检测，可识别未知人员")
        
        input("\n按 [Enter] 键开始识别...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 显示界面
            display_frame = frame.copy()
            cv2.putText(display_frame, "Face Recognition System", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, "SPACE: Recognize, ESC: Exit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Threshold: {self.anomaly_threshold:.3f}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 显示已知人员列表
            y_offset = 140
            cv2.putText(display_frame, "Known Persons:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            for i, person in enumerate(self.persons[:5]):  # 只显示前5个
                y_offset += 25
                cv2.putText(display_frame, f"• {person}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Face Recognition System', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # 空格键
                # 使用人脸对齐处理图像
                aligned_face = self.utils.face_aligner.process_image(frame)
                
                if aligned_face is not None:
                    if len(aligned_face.shape) == 3:
                        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
                    processed_image = aligned_face
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    processed_image = cv2.resize(gray, self.image_size)
                
                test_image = processed_image.flatten()
                
                # 进行识别
                result = self._perform_recognition(test_image, show_confidence)
                
                # 显示结果
                if result:
                    self._display_recognition_result(processed_image, result)
                    
                # 询问是否继续
                print("\n继续识别？(y/n): ", end="")
                continue_recognition = input().lower().strip()
                if continue_recognition != 'y' and continue_recognition != 'yes':
                    break
                    
            elif key == 27:  # ESC键
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print("\n识别模式已退出")
    
    def _perform_recognition(self, test_image, show_confidence=True):
        """
        执行带异常检测的人脸识别
        """
        try:
            # 投影到PCA空间
            test_projected = self.pca.transform(test_image.reshape(1, -1))
            
            # 使用异常检测进行预测
            predicted_person, min_distance, is_anomaly = self.utils.predict_with_anomaly_detection(
                test_projected[0], self.projected_data, self.training_labels, self.anomaly_threshold
            )
            
            # 计算置信度
            confidence = 1 / (1 + min_distance)
            
            result = {
                'person': predicted_person,
                'confidence': confidence,
                'distance': min_distance,
                'is_anomaly': is_anomaly,
                'recognized': not is_anomaly
            }
            
            if not is_anomaly:
                print(f"\n识别结果: {predicted_person}")
            else:
                print(f"\n识别结果: 未知人员（异常检测）")
            
            if show_confidence:
                print(f"置信度: {result['confidence']:.3f}")
                print(f"距离: {result['distance']:.3f}")
                print(f"异常检测: {'是' if is_anomaly else '否'}")
            
            return result
            
        except Exception as e:
            print(f"识别过程出错: {e}")
            return None
    
    def _display_recognition_result(self, image, result):
        """显示识别结果"""
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap='gray')
        
        if result['recognized']:
            title = f"识别结果: {result['person']}\n置信度: {result['confidence']:.3f}\n距离: {result['distance']:.3f}"
            plt.title(title, fontsize=14, color='green')
        else:
            title = f"识别结果: 未知人员\n置信度: {result['confidence']:.3f}\n距离: {result['distance']:.3f}\n异常检测: 是"
            plt.title(title, fontsize=14, color='red')
            
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def show_training_distribution(self):
        """显示训练数据分布"""
        if len(self.training_labels) == 0:
            print("没有训练数据")
            return
            
        distribution = Counter(self.training_labels)
        
        print("\n" + "="*50)
        print("训练数据分布统计")
        print("="*50)
        
        for person, count in distribution.items():
            print(f"{person}: {count} 张图像")
        print(f"\n总计: {len(self.training_labels)} 张图像，{len(distribution)} 个人")
        print("="*50)
        
        # 可视化分布
        plt.figure(figsize=(12, 6))
        persons = list(distribution.keys())
        counts = list(distribution.values())
        
        bars = plt.bar(persons, counts, color='skyblue', alpha=0.8)
        plt.title("训练数据分布", fontsize=16)
        plt.xlabel("人员", fontsize=12)
        plt.ylabel("图像数量", fontsize=12)
        plt.xticks(rotation=45)
        
        # 在柱状图上显示数值
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def visualize_pca_projection(self):
        """可视化PCA投影"""
        if self.projected_data is None:
            print("模型尚未训练！")
            return
            
        print("\n生成PCA投影可视化...")
        
        plt.figure(figsize=(14, 10))
        
        # 使用不同颜色表示不同的人
        unique_persons = list(set(self.training_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_persons)))
        
        for i, person in enumerate(unique_persons):
            idx = [j for j, label in enumerate(self.training_labels) if label == person]
            plt.scatter(self.projected_data[idx, 0], self.projected_data[idx, 1], 
                       c=[colors[i]], label=person, alpha=0.7, s=60)
        
        plt.title("PCA投影可视化（前两个主成分）", fontsize=16)
        plt.xlabel("第一主成分", fontsize=12)
        plt.ylabel("第二主成分", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def display_menu():
    """显示主菜单"""
    print("\n" + "="*60)
    print("动态人脸识别系统 - 主菜单")
    print("="*60)
    print("1. 显示训练数据分布")
    print("2. 可视化PCA投影")
    print("3. 添加新人员")
    print("4. 人脸识别")
    print("5. 全面模型评估")
    print("6. 退出系统")
    print("="*60)
    
def main():
    """主程序"""
    # 创建系统实例
    face_system = DynamicFaceRecognitionSystem()
    
    # 使用高级方法加载初始数据
    if not face_system.load_initial_data_advanced():
        print("\n初始化失败，程序退出")
        return
    
    # 显示初始状态
    face_system.show_training_distribution()
    
    # 主循环
    while True:
        try:
            display_menu()
            choice = input("\n请选择功能 (1-6): ").strip()
            
            if choice == '1':
                face_system.show_training_distribution()
                
            elif choice == '2':
                face_system.visualize_pca_projection()
                
            elif choice == '3':
                person_name = input("\n请输入新人员姓名: ").strip()
                if person_name:
                    num_photos = input("请输入拍照数量 (3-10, 默认5): ").strip()
                    try:
                        num_photos = int(num_photos) if num_photos else 5
                        num_photos = max(3, min(10, num_photos))  # 限制范围
                    except ValueError:
                        num_photos = 5
                    
                    face_system.add_new_person_data(person_name, num_photos)
                else:
                    print("请输入有效的人员姓名")
                    
            elif choice == '4':
                face_system.recognize_face()
                
            elif choice == '5':
                face_system.comprehensive_evaluation()
                
            elif choice == '6':
                print("\n感谢使用动态人脸识别系统！")
                print("演示结束，再见！")
                break
                
            else:
                print("无效选择，请输入 1-6")
                
        except KeyboardInterrupt:
            print("\n\n用户中断程序")
            print("感谢使用动态人脸识别系统！")
            break
        except Exception as e:
            print(f"\n程序运行出错: {e}")
            print("请重试或联系技术支持")

if __name__ == "__main__":
    # 设置matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行主程序
    main()