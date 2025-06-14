from face_recognition_utils import FaceRecognitionUtils
from PCA import PCA


def main():
    
    utils = FaceRecognitionUtils(image_size=(64, 64))
    
    train_dir = "training_images"
    challenge_dir = "challenge_images"
    misleading_dir = "misleading_images"
    processed_train_dir = "processed_training_images"
    processed_challenge_dir = "processed_challenge_images"
    processed_misleading_dir = "processed_misleading_images"
    
    print("\n处理并保存训练数据...")
    # 分割训练数据（带人脸对齐）
    X_train, y_train, X_test_train, y_test_train, failed_train = utils.split_training_data(
        train_dir, train_count=3, use_alignment=True
    )
    
    # 保存处理后的训练图像
    utils.load_dataset(
        train_dir, use_alignment=True, save_processed=True, 
        processed_dir=processed_train_dir, prefix="processed_"
    )
    
    print("\n处理并保存挑战集数据...")
    # 加载挑战集（带人脸对齐）
    X_challenge, y_challenge, _, failed_challenge = utils.load_dataset(
        challenge_dir, use_alignment=True, save_processed=True,
        processed_dir=processed_challenge_dir, prefix="processed_challenge_"
    )
    
    print("\n处理并保存误导项数据...")
    # 加载误导项（带人脸对齐）
    X_misleading, _, misleading_filenames, failed_misleading = utils.load_dataset(
        misleading_dir, use_alignment=True, save_processed=True,
        processed_dir=processed_misleading_dir, prefix="processed_misleading_"
    )
    
    # 数据统计
    print(f"\n数据统计:")
    print(f"   训练图像: {len(X_train)} 张")
    print(f"   训练集测试: {len(X_test_train)} 张")
    print(f"   挑战集: {len(X_challenge)} 张")
    print(f"   误导项: {len(X_misleading)} 张")
    print(f"   处理失败: {failed_train + failed_challenge + failed_misleading} 张")
    
    # 训练PCA
    print("\n训练PCA模型...")
    pca = PCA(n_components=20)
    X_train_proj = pca.fit_transform(X_train)
    X_test_train_proj = pca.transform(X_test_train)
    X_challenge_proj = pca.transform(X_challenge)
    X_misleading_proj = pca.transform(X_misleading)
    
    print(f"   使用主成分数量: {pca.n_components}")
    
    # 计算异常检测阈值
    print("\n计算异常检测阈值...")
    threshold, stats = utils.calculate_anomaly_threshold(X_train_proj, y_train, percentile=95)
    print(f"   异常检测阈值: {threshold:.4f}")
    print(f"   类内距离统计: 均值={stats['intra_class_mean']:.4f}, 标准差={stats['intra_class_std']:.4f}")
    print(f"   最小距离统计: 均值={stats['min_distance_mean']:.4f}, 标准差={stats['min_distance_std']:.4f}")
    
    # 评估模型
    print("\n" + "=" * 40)
    print("模型评估")
    print("=" * 40)
    
    # 评估训练集测试部分
    if len(X_test_train) > 0:
        print("\n训练集测试结果:")
        acc_train, _, _ = utils.evaluate_with_anomaly_detection(
            X_test_train_proj, y_test_train, X_train_proj, y_train, threshold, 
            "训练集测试", expected_anomalies=False
        )
    else:
        acc_train = 0
        print("\n跳过训练集测试（无数据）")
    
    # 评估挑战集
    if len(X_challenge) > 0:
        print("\n挑战集测试结果:")
        acc_challenge, _, _ = utils.evaluate_with_anomaly_detection(
            X_challenge_proj, y_challenge, X_train_proj, y_train, threshold,
            "挑战集", expected_anomalies=False
        )
    else:
        acc_challenge = 0
        print("\n跳过挑战集测试（无数据）")
    
    # 评估误导项检测
    if len(X_misleading) > 0:
        print("\n误导项检测结果:")
        _, anomaly_acc, _ = utils.evaluate_with_anomaly_detection(
            X_misleading_proj, misleading_filenames, X_train_proj, y_train, threshold,
            "误导项检测", expected_anomalies=True
        )
    else:
        anomaly_acc = 0
        print("\n跳过误导项测试（无数据）")
    
    # 最终总结
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"训练集测试准确率: {acc_train:.2%}")
    print(f"挑战集准确率: {acc_challenge:.2%}")
    print(f"误导项检测准确率: {anomaly_acc:.2%}")
    print(f"图像尺寸: {utils.image_size}")
    print(f"PCA主成分: {pca.n_components}")
    print(f"异常检测阈值: {threshold:.4f}")
    print(f"\n处理后的图像已保存:")
    print(f"   训练图像: {processed_train_dir}/")
    print(f"   挑战图像: {processed_challenge_dir}/")
    print(f"   误导项图像: {processed_misleading_dir}/")
    print(f"\n功能特性:")
    print(f"   • 人脸检测和对齐处理")
    print(f"   • 基于PCA的降维")
    print(f"   • 异常检测和误导项识别")
    print(f"   • 详细的测试结果输出")

if __name__ == "__main__":
    main()