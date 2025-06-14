from face_recognition_utils import FaceRecognitionUtils
from PCA import PCA
import numpy as np

def main():
    print("=" * 60)
    print("人脸识别系统 v1.0 - 基础版本")
    print("=" * 60)
    
    # 初始化工具
    utils = FaceRecognitionUtils(image_size=(64, 64))
    
    # 数据目录
    train_dir = "training_images"
    challenge_dir = "challenge_images"
    
    print("\n加载训练数据...")
    # 分割训练数据
    X_train, y_train, X_test_train, y_test_train, failed_train = utils.split_training_data(
        train_dir, train_count=3, use_alignment=False
    )
    
    print("\n加载挑战集数据...")
    # 加载挑战集
    X_challenge, y_challenge, _, failed_challenge = utils.load_dataset(
        challenge_dir, use_alignment=False
    )
    
    # 数据统计
    print(f"\n数据统计:")
    print(f"   训练图像: {len(X_train)} 张")
    print(f"   训练集测试: {len(X_test_train)} 张")
    print(f"   挑战集: {len(X_challenge)} 张")
    print(f"   处理失败: {failed_train + failed_challenge} 张")
    
    # 训练PCA
    print("\n训练PCA模型...")
    pca = PCA(n_components=20)
    X_train_proj = pca.fit_transform(X_train)
    X_test_train_proj = pca.transform(X_test_train)
    X_challenge_proj = pca.transform(X_challenge)
    
    print(f"   使用主成分数量: {pca.n_components}")
    
    # 评估模型
    print("\n" + "=" * 40)
    print("模型评估")
    print("=" * 40)
    
    # 评估训练集测试部分
    if len(X_test_train) > 0:
        print("\n训练集测试结果:")
        acc_train, _ = utils.evaluate_simple(
            X_test_train_proj, y_test_train, X_train_proj, y_train, "训练集测试"
        )
    else:
        acc_train = 0
        print("\n跳过训练集测试（无数据）")
    
    # 评估挑战集
    if len(X_challenge) > 0:
        print("\n挑战集测试结果:")
        acc_challenge, _ = (utils.evaluate_simple(
            X_challenge_proj, y_challenge, X_train_proj, y_train, "挑战集"
        ))
    else:
        acc_challenge = 0
        print("\n跳过挑战集测试（无数据）")
    
    # 最终总结
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"训练集测试准确率: {acc_train:.2%}")
    print(f"挑战集准确率: {acc_challenge:.2%}")
    print(f"图像尺寸: {utils.image_size}")
    print(f"PCA主成分: {pca.n_components}")

if __name__ == "__main__":
    main()