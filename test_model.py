import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd

# 从model_gan导入必要的组件
from model_gan import UNetPlusPlus, PolypDataset, dice_coefficient, iou_coefficient, precision_score, recall_score, accuracy_score
from visualization import generate_curves, plot_segmentation_examples, create_comparison_plots, plot_metrics_summary

def evaluate_model_test(model, dataloader, device, criterion):
    """
    在测试集上评估模型性能
    
    参数:
        model: 要评估的模型
        dataloader: 测试数据加载器
        device: 计算设备
        criterion: 损失函数
        
    返回:
        包含评估结果的字典
    """
    model.eval()
    metric_functions = {
        'dice': dice_coefficient,
        'iou': iou_coefficient,
        'precision': precision_score,
        'recall': recall_score,
        'accuracy': accuracy_score
    }
    
    metric_values = {name: 0 for name in metric_functions}
    num_samples = 0
    total_loss = 0
    all_preds = []
    all_targets = []
    
    print(f"开始评估测试集...")
    
    with torch.no_grad():
        for batch in dataloader:
            # 读取batch数据
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
            else:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 二值化输出和掩码，用于指标计算
            binary_outputs = (outputs > 0.5).float()
            binary_masks = (masks > 0.5).float()
            
            # 计算指标
            for name, metric_fn in metric_functions.items():
                metric_values[name] += metric_fn(binary_outputs, binary_masks).item() * images.size(0)
            
            # 累计批次大小和损失
            num_samples += images.size(0)
            total_loss += loss.item() * images.size(0)
            
            # 收集预测和目标，用于ROC和PR曲线
            all_preds.extend(outputs.view(-1).cpu().numpy())
            all_targets.extend(binary_masks.view(-1).cpu().numpy())
    
    # 计算平均损失和指标
    avg_loss = total_loss / num_samples
    avg_metrics = {name: value / num_samples for name, value in metric_values.items()}
    
    # 计算ROC和PR曲线
    curves_data = generate_curves(
        predictions=all_preds,
        targets=all_targets
    )
    
    # 返回评估结果
    return {
        'loss': avg_loss,
        'metrics': avg_metrics,
        'curves': curves_data,
        'all_preds': all_preds,
        'all_targets': all_targets
    }

def test_model_on_datasets(args):
    """
    在所有数据集的测试集上测试模型
    
    参数:
        args: 命令行参数
    """
    # 设置路径
    data_root = args.data_root
    models_dir = args.models_dir
    results_dir = args.results_dir
    
    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 定义数据集名称
    dataset_names = ['Kvasir-SEG', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'CVC-ColonDB']
    
    # 收集所有数据集的结果，用于后续绘制对比图
    all_datasets_results = {}
    
    # 收集结果数据
    results_data = []
    
    # 循环处理每个数据集
    for dataset_name in dataset_names:
        print(f"\n处理数据集: {dataset_name}")
        
        # 创建测试数据集和数据加载器
        test_dataset = PolypDataset(data_root, dataset_name, split="test", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        print(f"测试集大小: {len(test_dataset)}")
        
        # 创建模型
        generator = UNetPlusPlus(in_channels=3, out_channels=1).to(device)
        
        # 加载最佳模型参数
        checkpoint_path = os.path.join(models_dir, dataset_name, 'best_model.pth')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            best_epoch = checkpoint.get('epoch', 'unknown')
            best_dice = checkpoint.get('best_dice', 'unknown')
            print(f"已加载模型从 {checkpoint_path}")
            print(f"最佳模型来自Epoch {best_epoch}，验证集Dice系数: {best_dice}")
        else:
            print(f"错误: 未找到模型 {checkpoint_path}")
            continue
        
        # 数据集测试结果目录
        dataset_result_dir = os.path.join(results_dir, dataset_name)
        os.makedirs(dataset_result_dir, exist_ok=True)
        
        # 在测试集上评估
        criterion_bce = nn.BCELoss().to(device)
        test_metrics = evaluate_model_test(generator, test_loader, device, criterion_bce)
        
        # 输出测试结果
        print(f"\n{dataset_name} 测试结果:")
        print(f"Loss: {test_metrics['loss']:.4f}")
        for metric_name, metric_value in test_metrics['metrics'].items():
            print(f"{metric_name}: {metric_value:.4f}")
        print(f"ROC AUC: {test_metrics['curves']['roc_auc']:.4f}")
        print(f"PR AUC: {test_metrics['curves']['pr_auc']:.4f}")
        
        # 保存数据集测试结果
        all_datasets_results[dataset_name] = {
            'test_metrics': test_metrics['metrics'],
            'curves': test_metrics['curves']
        }
        
        # 保存ROC和PR曲线
        if len(test_metrics['all_preds']) > 0 and len(test_metrics['all_targets']) > 0:
            curve_path = os.path.join(dataset_result_dir, "test_roc_pr_curves.png")
            generate_curves(
                predictions=test_metrics['all_preds'], 
                targets=test_metrics['all_targets'],
                save_path=curve_path,
                title=f"Test Results - {dataset_name}"
            )
            print(f"曲线已保存到: {curve_path}")
        
        # 保存分割示例图
        plot_segmentation_examples(generator, test_loader, dataset_result_dir, device, num_examples=5)
        
        # 收集结果数据
        results_data.append({
            'Dataset': dataset_name,
            'Dice': test_metrics['metrics']['dice'],
            'IoU': test_metrics['metrics']['iou'],
            'Precision': test_metrics['metrics']['precision'],
            'Recall': test_metrics['metrics']['recall'],
            'Accuracy': test_metrics['metrics']['accuracy'],
            'ROC_AUC': test_metrics['curves']['roc_auc']
        })
    
    # 如果至少有一个数据集有结果
    if results_data:
        # 计算宏平均
        dice_mean = np.mean([row['Dice'] for row in results_data])
        iou_mean = np.mean([row['IoU'] for row in results_data])
        precision_mean = np.mean([row['Precision'] for row in results_data])
        recall_mean = np.mean([row['Recall'] for row in results_data])
        accuracy_mean = np.mean([row['Accuracy'] for row in results_data])
        roc_auc_mean = np.mean([row['ROC_AUC'] for row in results_data])
        
        # 添加宏平均行
        results_data.append({
            'Dataset': '宏平均(Macro Average)',
            'Dice': dice_mean,
            'IoU': iou_mean,
            'Precision': precision_mean,
            'Recall': recall_mean,
            'Accuracy': accuracy_mean,
            'ROC_AUC': roc_auc_mean
        })
        
        # 创建完整的DataFrame
        results_df = pd.DataFrame(results_data)
        
        # 保存结果表格
        results_df.to_csv(os.path.join(results_dir, 'test_results.csv'), index=False)
        
        # 生成所有数据集的对比图
        comparison_dir = os.path.join(results_dir, 'test_comparison')
        create_comparison_plots(all_datasets_results, comparison_dir)
        
        # 生成度量指标汇总图
        plot_metrics_summary(results_df, results_dir)
        
        # 打印结果摘要
        print("\n测试结果摘要:")
        print(results_df.to_string(index=False))
        print(f"\n详细结果已保存到: {results_dir}/test_results.csv")
    else:
        print("错误: 没有任何数据集的测试结果")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试GAN模型在多个息肉数据集上的表现")
    parser.add_argument('--data_root', type=str, default='data', help='数据集根目录')
    parser.add_argument('--results_dir', type=str, default='test_results', help='测试结果保存目录')
    parser.add_argument('--models_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    
    args = parser.parse_args()
    
    test_model_on_datasets(args) 