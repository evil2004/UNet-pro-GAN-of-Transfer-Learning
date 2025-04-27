import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def generate_curves(predictions, targets, save_path=None, title=None):
    """
    生成ROC曲线和PR曲线，计算相关指标。
    
    参数:
        predictions: 模型预测的概率值，形状为 [N]
        targets: 真实标签，形状为 [N]，值为0或1
        save_path: 可选，保存曲线图的路径
        title: 可选，曲线图的标题
    
    返回:
        包含曲线数据和指标的字典
    """
    # 初始化默认返回结果
    result = {
        'roc_auc': 0.0,
        'pr_auc': 0.0,
        'fpr': [],
        'tpr': [],
        'precision': [],
        'recall': []
    }
    
    try:
        # 类型检查和转换
        import torch
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy().flatten()
        elif isinstance(predictions, list):
            predictions = np.array(predictions).flatten()
        
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy().flatten()
        elif isinstance(targets, list):
            targets = np.array(targets).flatten()
        
        # 检查目标值是否为二值，如果不是，则转换
        if not np.array_equal(np.unique(targets), np.array([0, 1])) and not np.array_equal(np.unique(targets), np.array([0])) and not np.array_equal(np.unique(targets), np.array([1])):
            warnings.warn("Target values are not binary (0 and 1), thresholding will be applied.")
            targets = (targets > 0.5).astype(float)
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)
        
        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(targets, predictions)
        pr_auc = auc(recall, precision)
        
        # 绘制曲线
        if save_path is not None:
            # 确保保存路径的目录存在
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 5))
            
            # ROC曲线
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve' if title is None else f'{title} - ROC')
            plt.legend(loc="lower right")
            
            # PR曲线
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve' if title is None else f'{title} - PR')
            plt.legend(loc="lower left")
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        # 更新结果
        result['roc_auc'] = float(roc_auc)
        result['pr_auc'] = float(pr_auc)
        result['fpr'] = fpr.tolist()
        result['tpr'] = tpr.tolist()
        result['precision'] = precision.tolist()
        result['recall'] = recall.tolist()
        
    except Exception as e:
        warnings.warn(f"Error generating curves: {str(e)}. Returning default values.")
    
    return result


def plot_training_history(train_losses, val_metrics_history, save_dir, dataset_name):
    """
    绘制训练历史曲线
    
    参数:
        train_losses: 训练损失列表，每个元素为包含epoch、损失等信息的字典
        val_metrics_history: 验证指标历史记录字典
        save_dir: 保存目录
        dataset_name: 数据集名称
    """
    if len(train_losses) == 0:
        return
    
    # 创建保存目录
    plots_dir = os.path.join(save_dir, dataset_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 绘制训练损失曲线
    plt.figure(figsize=(12, 8))
    epochs = [item['epoch'] for item in train_losses]
    g_losses = [item['generator_loss'] for item in train_losses]
    d_losses = [item['discriminator_loss'] for item in train_losses]
    pixel_losses = [item['pixel_loss'] for item in train_losses]
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, g_losses, 'b-', label='Generator Loss')
    plt.plot(epochs, d_losses, 'r-', label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Losses - {dataset_name}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, pixel_losses, 'g-', label='Pixel Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pixel Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_losses.png'))
    plt.close()
    
    # 绘制验证指标曲线
    plt.figure(figsize=(15, 10))
    for i, (metric_name, metric_values) in enumerate(val_metrics_history.items()):
        if len(metric_values) > 0:
            plt.subplot(3, 2, i+1)
            plt.plot(epochs, metric_values, 'b-', label=metric_name)
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name.capitalize()} - {dataset_name}')
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'validation_metrics.png'))
    plt.close()
    
    # 保存训练指标数据
    train_history = {
        'epoch': epochs,
        'generator_loss': g_losses,
        'discriminator_loss': d_losses,
        'pixel_loss': pixel_losses
    }
    for metric_name, metric_values in val_metrics_history.items():
        if len(metric_values) > 0:
            train_history[f'val_{metric_name}'] = metric_values
    
    pd.DataFrame(train_history).to_csv(os.path.join(plots_dir, 'training_history.csv'), index=False)


def plot_segmentation_examples(generator, test_loader, dataset_result_dir, device, num_examples=5):
    """
    生成分割示例图像
    
    参数:
        generator: 生成器模型
        test_loader: 测试数据加载器
        dataset_result_dir: 结果保存目录
        device: 计算设备
        num_examples: 示例数量
    """
    import torch
    
    generator.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_examples:  # 只保存指定数量的示例
                break
            
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
            else:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)
            
            outputs = generator(images)
            
            # 将预测转为二值
            pred_masks = (outputs > 0.5).float()
            
            # 保存可视化结果
            for j in range(images.size(0)):
                img = images[j].cpu().numpy().transpose(1, 2, 0)
                true_mask = masks[j].cpu().numpy().squeeze()
                pred_mask = pred_masks[j].cpu().numpy().squeeze()
                
                # 归一化图像
                img = (img - img.min()) / (img.max() - img.min())
                
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title('Input Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(true_mask, cmap='gray')
                plt.title('Ground Truth Mask')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred_mask, cmap='gray')
                plt.title('Predicted Mask')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(dataset_result_dir, f'sample_{i*images.size(0)+j}.png'))
                plt.close()


def create_comparison_plots(dataset_results, save_dir):
    """
    为多个数据集创建对比曲线图
    
    参数:
        dataset_results: 包含各数据集结果的字典
        save_dir: 保存图像的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 如果结果集为空，直接返回
    if not dataset_results:
        warnings.warn("没有数据集结果可以用于创建对比图")
        return
    
    # 1. 创建ROC曲线对比图
    plt.figure(figsize=(10, 8))
    has_roc_data = False
    for dataset_name, result in dataset_results.items():
        if 'curves' in result and 'fpr' in result['curves'] and 'tpr' in result['curves']:
            fpr = result['curves']['fpr']
            tpr = result['curves']['tpr']
            roc_auc = result['curves']['roc_auc']
            plt.plot(fpr, tpr, lw=2, label=f'{dataset_name} (AUC = {roc_auc:.3f})')
            has_roc_data = True
    
    if has_roc_data:
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'roc_comparison.png'))
        plt.close()
    else:
        plt.close()
        warnings.warn("没有足够的数据生成ROC对比曲线")
    
    # 2. 创建PR曲线对比图
    plt.figure(figsize=(10, 8))
    has_pr_data = False
    for dataset_name, result in dataset_results.items():
        if 'curves' in result and 'precision' in result['curves'] and 'recall' in result['curves']:
            precision = result['curves']['precision']
            recall = result['curves']['recall']
            pr_auc = result['curves']['pr_auc']
            plt.plot(recall, precision, lw=2, label=f'{dataset_name} (AUC = {pr_auc:.3f})')
            has_pr_data = True
    
    if has_pr_data:
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'pr_comparison.png'))
        plt.close()
    else:
        plt.close()
        warnings.warn("没有足够的数据生成PR对比曲线")
    
    # 3. 创建训练损失对比图 - 跳过，我们只关注测试指标
    
    # 4. 创建验证指标对比图 - 跳过，我们只关注测试指标
    
    # 5. 创建最终性能对比条形图
    metrics_names = ['Dice', 'IoU', 'Precision', 'Recall', 'Accuracy', 'ROC_AUC']
    metrics_values = {}
    valid_datasets = []
    
    # 首先检查哪些数据集有完整的指标数据
    for dataset_name, result in dataset_results.items():
        has_complete_metrics = (
            'test_metrics' in result and 
            all(metric in result['test_metrics'] for metric in ['dice', 'iou', 'precision', 'recall', 'accuracy']) and
            'curves' in result and 'roc_auc' in result['curves']
        )
        
        if has_complete_metrics:
            metrics_values[dataset_name] = [
                result['test_metrics'].get('dice', 0),
                result['test_metrics'].get('iou', 0),
                result['test_metrics'].get('precision', 0),
                result['test_metrics'].get('recall', 0),
                result['test_metrics'].get('accuracy', 0),
                result['curves'].get('roc_auc', 0)
            ]
            valid_datasets.append(dataset_name)
    
    # 只有当有有效数据集时才创建条形图
    if valid_datasets:
        x = np.arange(len(metrics_names))
        width = 0.8 / len(valid_datasets) if len(valid_datasets) > 0 else 0.4  # 条形宽度
        
        fig, ax = plt.subplots(figsize=(15, 10))
        for i, dataset_name in enumerate(valid_datasets):
            ax.bar(x + i * width, metrics_values[dataset_name], width, label=dataset_name)
        
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x + width * (len(valid_datasets) - 1) / 2 if len(valid_datasets) > 0 else x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        plt.ylim(0, 1)
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_comparison_bar.png'))
        plt.close()
        
        print(f"Comparison plots saved to {save_dir}")
    else:
        warnings.warn("没有足够的数据生成性能指标对比条形图")


def plot_metrics_summary(results_df, save_dir):
    """
    绘制所有数据集的度量指标汇总图
    
    参数:
        results_df: 包含所有数据集结果的DataFrame
        save_dir: 保存目录
    """
    # 生成结果条形图
    metrics = ['Dice', 'IoU', 'Precision', 'Recall', 'Accuracy', 'ROC_AUC']
    
    plt.figure(figsize=(15, 12))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        bars = plt.bar(results_df['Dataset'][:-1], results_df[metric][:-1])
        plt.axhline(y=results_df[metric].iloc[-1], color='r', linestyle='--', label='宏平均(Macro Average)')
        plt.title(metric)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'))
    plt.close()


def plot_combined_training_history(train_histories, save_dir):
    """
    将所有数据集的训练历史绘制在同一张图上
    
    参数:
        train_histories: 字典，键为数据集名称，值为训练历史
        save_dir: 保存目录
    """
    plots_dir = os.path.join(save_dir, "combined_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 绘制损失对比图
    plt.figure(figsize=(15, 10))
    
    # 1. 生成器损失曲线
    plt.subplot(2, 2, 1)
    for dataset_name, history in train_histories.items():
        epochs = [item['epoch'] for item in history['train_losses']]
        g_losses = [item['generator_loss'] for item in history['train_losses']]
        plt.plot(epochs, g_losses, '-', label=f'{dataset_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. 判别器损失曲线
    plt.subplot(2, 2, 2)
    for dataset_name, history in train_histories.items():
        epochs = [item['epoch'] for item in history['train_losses']]
        d_losses = [item['discriminator_loss'] for item in history['train_losses']]
        plt.plot(epochs, d_losses, '-', label=f'{dataset_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.grid(True)
    
    # 3. 像素损失曲线
    plt.subplot(2, 2, 3)
    for dataset_name, history in train_histories.items():
        epochs = [item['epoch'] for item in history['train_losses']]
        pixel_losses = [item['pixel_loss'] for item in history['train_losses']]
        plt.plot(epochs, pixel_losses, '-', label=f'{dataset_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pixel Loss')
    plt.legend()
    plt.grid(True)
    
    # 4. 验证Dice系数曲线
    plt.subplot(2, 2, 4)
    for dataset_name, history in train_histories.items():
        if 'val_metrics_history' in history and 'dice' in history['val_metrics_history']:
            epochs = [i+1 for i in range(len(history['val_metrics_history']['dice']))]
            dice_values = history['val_metrics_history']['dice']
            plt.plot(epochs, dice_values, '-', label=f'{dataset_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Validation Dice')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_datasets_training_history.png'))
    plt.close()
    
    print(f"所有数据集的训练历史曲线已保存至: {os.path.join(plots_dir, 'all_datasets_training_history.png')}")
    
    # 绘制所有验证指标曲线
    metrics = ['dice', 'iou', 'precision', 'recall', 'accuracy']
    plt.figure(figsize=(15, 12))
    
    for i, metric_name in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        for dataset_name, history in train_histories.items():
            if 'val_metrics_history' in history and metric_name in history['val_metrics_history']:
                epochs = [j+1 for j in range(len(history['val_metrics_history'][metric_name]))]
                metric_values = history['val_metrics_history'][metric_name]
                plt.plot(epochs, metric_values, '-', label=f'{dataset_name}')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'Validation {metric_name.capitalize()}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_datasets_validation_metrics.png'))
    plt.close()
    
    print(f"所有数据集的验证指标曲线已保存至: {os.path.join(plots_dir, 'all_datasets_validation_metrics.png')}") 