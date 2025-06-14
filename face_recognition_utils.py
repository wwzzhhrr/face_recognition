import os
import cv2
import numpy as np
from face_aligner import FaceAligner
from collections import defaultdict


class FaceRecognitionUtils:
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size
        self.face_aligner = FaceAligner(desired_face_width=image_size[0], desired_face_height=image_size[1])
        
    def load_image_simple(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, self.image_size)
        return img.flatten()
    
    def load_image_with_alignment(self, path, save_processed=False, save_path=None):
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
    
    def predict_simple(self, test_sample, X_train_proj, y_train):
        """简单预测方法"""
        distances = np.linalg.norm(X_train_proj - test_sample, axis=1)
        min_index = np.argmin(distances)
        return y_train[min_index], distances[min_index]
    
    def predict_with_anomaly_detection(self, test_sample, X_train_proj, y_train, threshold):
        """带异常检测的预测方法"""
        distances = np.linalg.norm(X_train_proj - test_sample, axis=1)
        min_distance = np.min(distances)
        min_index = np.argmin(distances)
        
        is_anomaly = min_distance > threshold
        predicted_label = y_train[min_index] if not is_anomaly else "UNKNOWN"
        
        return predicted_label, min_distance, is_anomaly
    
    def evaluate_simple(self, X_test_proj, y_test, X_train_proj, y_train, dataset_name="Test", verbose=True):
        """简单评估方法"""
        correct = 0
        results = []
        
        for i in range(len(X_test_proj)):
            pred, distance = self.predict_simple(X_test_proj[i], X_train_proj, y_train)
            actual = y_test[i]
            is_correct = pred == actual
            
            if is_correct:
                correct += 1
            
            result = {
                'index': i + 1,
                'predicted': pred,
                'actual': actual,
                'distance': distance,
                'correct': is_correct
            }
            results.append(result)
            
            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"[{dataset_name} {i+1:2d}] {status} 预测: {pred:8s} | 实际: {actual:8s} | 距离: {distance:.4f}")
        
        accuracy = correct / len(X_test_proj) if len(X_test_proj) > 0 else 0
        
        if verbose:
            print(f"\n{dataset_name} 准确率: {accuracy:.2%} ({correct}/{len(X_test_proj)})")
        
        return accuracy, results
    
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