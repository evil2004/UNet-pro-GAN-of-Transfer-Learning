import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 从model_gan导入必要的组件
from model_gan import UNetPlusPlus, PolypDataset, dice_coefficient, iou_coefficient, precision_score, recall_score, accuracy_score
from visualization import generate_curves, plot_segmentation_examples, create_comparison_plots, plot_metrics_summary

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

# 新增函数：绘制所有数据集合并的ROC曲线和PR曲线
def plot_combined_curves(dataset_results, save_dir, plot_type='source'):
    """
    将所有数据集的ROC曲线和PR曲线绘制在同一个图表上
    
    参数:
        dataset_results: 包含各数据集结果的字典
        save_dir: 保存图像的目录
        plot_type: 'source'表示源模型, 'finetune'表示微调模型, 'comparison'表示对比
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建ROC曲线图
    plt.figure(figsize=(12, 10))
    for dataset_name, result in dataset_results.items():
        if 'curves' in result and 'fpr' in result['curves'] and 'tpr' in result['curves']:
            fpr = result['curves']['fpr']
            tpr = result['curves']['tpr']
            roc_auc = result['curves']['roc_auc']
            plt.plot(fpr, tpr, lw=2, label=f'{dataset_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真阳性率 (True Positive Rate)')
    plt.title(f'所有数据集的ROC曲线比较 - {plot_type}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{plot_type}_combined_roc_curves.png'), dpi=300)
    plt.close()
    
    # 创建PR曲线图
    plt.figure(figsize=(12, 10))
    for dataset_name, result in dataset_results.items():
        if 'curves' in result and 'precision' in result['curves'] and 'recall' in result['curves']:
            precision = result['curves']['precision']
            recall = result['curves']['recall']
            pr_auc = result['curves']['pr_auc']
            plt.plot(recall, precision, lw=2, label=f'{dataset_name} (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title(f'所有数据集的PR曲线比较 - {plot_type}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{plot_type}_combined_pr_curves.png'), dpi=300)
    plt.close()
    
    print(f"已生成所有数据集的合并ROC和PR曲线图: {save_dir}")

# 新增函数：绘制所有数据集的评估指标对比图
def plot_combined_metrics(dataset_results, save_dir, plot_type='source'):
    """
    将所有数据集的评估指标绘制在同一个图表上进行对比
    
    参数:
        dataset_results: 包含各数据集结果的字典
        save_dir: 保存图像的目录
        plot_type: 'source'表示源模型, 'finetune'表示微调模型
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取指标
    metrics = ['dice', 'iou', 'precision', 'recall', 'accuracy']
    metric_names = ['Dice系数', 'IoU系数', '精确率', '召回率', '准确率']
    
    datasets = list(dataset_results.keys())
    if not datasets:
        print("警告: 没有可用的数据集结果")
        return
    
    # 创建大图
    plt.figure(figsize=(15, 10))
    
    # 设置柱状图的位置和宽度
    x = np.arange(len(metrics))
    width = 0.8 / len(datasets) if len(datasets) > 0 else 0.4  # 柱状图宽度
    
    # 为每个数据集绘制所有指标的柱状图
    for i, dataset_name in enumerate(datasets):
        values = []
        for metric in metrics:
            if 'test_metrics' in dataset_results[dataset_name] and metric in dataset_results[dataset_name]['test_metrics']:
                values.append(dataset_results[dataset_name]['test_metrics'][metric])
            else:
                values.append(0)
        
        # 添加ROC AUC
        if 'curves' in dataset_results[dataset_name] and 'roc_auc' in dataset_results[dataset_name]['curves']:
            values.append(dataset_results[dataset_name]['curves']['roc_auc'])
        else:
            values.append(0)
    
        # 绘制柱状图，偏移以避免重叠
        plt.bar(x + i * width - width * (len(datasets) - 1) / 2, values[:len(metrics)], width, label=dataset_name)
    
    plt.xlabel('评估指标')
    plt.ylabel('分数')
    plt.title(f'所有数据集的评估指标对比 - {plot_type}')
    plt.xticks(x, metric_names)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(save_dir, f'{plot_type}_combined_metrics.png'), dpi=300)
    plt.close()
    
    print(f"已生成所有数据集的评估指标对比图: {os.path.join(save_dir, f'{plot_type}_combined_metrics.png')}")
    
    # 单独创建ROC AUC的对比图
    plt.figure(figsize=(10, 6))
    roc_auc_values = []
    for dataset_name in datasets:
        if 'curves' in dataset_results[dataset_name] and 'roc_auc' in dataset_results[dataset_name]['curves']:
            roc_auc_values.append(dataset_results[dataset_name]['curves']['roc_auc'])
        else:
            roc_auc_values.append(0)
    
    plt.bar(datasets, roc_auc_values)
    plt.xlabel('数据集')
    plt.ylabel('ROC AUC')
    plt.title(f'所有数据集的ROC AUC对比 - {plot_type}')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # 保存ROC AUC对比图
    plt.savefig(os.path.join(save_dir, f'{plot_type}_roc_auc_comparison.png'), dpi=300)
    plt.close()
    
    print(f"已生成所有数据集的ROC AUC对比图: {os.path.join(save_dir, f'{plot_type}_roc_auc_comparison.png')}")

def test_finetune_model(args):
    """
    测试迁移学习模型在所有数据集上的表现
    
    参数:
        args: 命令行参数
    """
    # 设置路径
    data_root = args.data_root
    source_models_dir = args.source_models_dir
    finetune_models_dir = args.finetune_models_dir
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
    source_dataset = 'Kvasir-SEG'  # 源数据集
    target_datasets = ['CVC-ClinicDB', 'ETIS-LaribPolypDB', 'CVC-ColonDB']  # 目标数据集
    all_datasets = [source_dataset] + target_datasets
    
    # 收集所有数据集的结果，用于后续绘制对比图
    source_model_results = {}
    finetune_model_results = {}
    
    # 收集结果数据
    results_data = []
    
    print("\n1. 首先测试源模型在所有数据集上的表现")
    # 加载源模型
    source_generator = UNetPlusPlus(in_channels=3, out_channels=1).to(device)
    source_checkpoint_path = os.path.join(source_models_dir, source_dataset, 'best_model.pth')
    
    if os.path.exists(source_checkpoint_path):
        source_checkpoint = torch.load(source_checkpoint_path, map_location=device)
        source_generator.load_state_dict(source_checkpoint['generator_state_dict'])
        best_epoch = source_checkpoint.get('epoch', 'unknown')
        best_dice = source_checkpoint.get('best_dice', 'unknown')
        print(f"已加载源模型从 {source_checkpoint_path}")
        print(f"源模型来自Epoch {best_epoch}，验证集Dice系数: {best_dice}")
    else:
        print(f"错误: 未找到源模型 {source_checkpoint_path}")
        return
    
    # 在所有数据集上测试源模型
    for dataset_name in all_datasets:
        print(f"\n测试源模型在 {dataset_name} 上的表现:")
        
        # 创建测试数据集和数据加载器
        test_dataset = PolypDataset(data_root, dataset_name, split="test", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        print(f"测试集大小: {len(test_dataset)}")
        
        # 源模型测试结果目录
        source_result_dir = os.path.join(results_dir, f"source_model_{dataset_name}")
        os.makedirs(source_result_dir, exist_ok=True)
        
        # 在测试集上评估源模型
        criterion_bce = nn.BCELoss().to(device)
        test_metrics = evaluate_model_test(source_generator, test_loader, device, criterion_bce)
        
        # 输出测试结果
        print(f"\n源模型在 {dataset_name} 上的测试结果:")
        print(f"Loss: {test_metrics['loss']:.4f}")
        for metric_name, metric_value in test_metrics['metrics'].items():
            print(f"{metric_name}: {metric_value:.4f}")
        print(f"ROC AUC: {test_metrics['curves']['roc_auc']:.4f}")
        print(f"PR AUC: {test_metrics['curves']['pr_auc']:.4f}")
        
        # 保存源模型测试结果
        source_model_results[dataset_name] = {
            'test_metrics': test_metrics['metrics'],
            'curves': test_metrics['curves']
        }
        
        # 保存ROC和PR曲线
        if len(test_metrics['all_preds']) > 0 and len(test_metrics['all_targets']) > 0:
            curve_path = os.path.join(source_result_dir, "test_roc_pr_curves.png")
            generate_curves(
                predictions=test_metrics['all_preds'], 
                targets=test_metrics['all_targets'],
                save_path=curve_path,
                title=f"Source Model Test Results - {dataset_name}"
            )
            print(f"源模型曲线已保存到: {curve_path}")
        
        # 保存分割示例图
        plot_segmentation_examples(source_generator, test_loader, source_result_dir, device, num_examples=5)
        
        # 收集源模型结果数据
        results_data.append({
            'Dataset': dataset_name,
            'Model': 'Source Model',
            'Dice': test_metrics['metrics']['dice'],
            'IoU': test_metrics['metrics']['iou'],
            'Precision': test_metrics['metrics']['precision'],
            'Recall': test_metrics['metrics']['recall'],
            'Accuracy': test_metrics['metrics']['accuracy'],
            'ROC_AUC': test_metrics['curves']['roc_auc']
        })
    
    # 生成源模型所有数据集的合并ROC和PR曲线
    plot_combined_curves(source_model_results, results_dir, plot_type='source')
    
    # 生成源模型所有数据集的评估指标对比图
    plot_combined_metrics(source_model_results, results_dir, plot_type='source')
    
    print("\n2. 测试微调模型在所有数据集上的表现")
    
    for target_dataset in target_datasets:
        print(f"\n测试微调模型在 {target_dataset} 上的表现:")
        
        # 创建测试数据集和数据加载器
        test_dataset = PolypDataset(data_root, target_dataset, split="test", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        print(f"测试集大小: {len(test_dataset)}")
        
        # 加载微调模型
        finetune_generator = UNetPlusPlus(in_channels=3, out_channels=1).to(device)
        finetune_checkpoint_path = os.path.join(finetune_models_dir, target_dataset, 'best_model.pth')
        
        if os.path.exists(finetune_checkpoint_path):
            finetune_checkpoint = torch.load(finetune_checkpoint_path, map_location=device)
            finetune_generator.load_state_dict(finetune_checkpoint['generator_state_dict'])
            best_epoch = finetune_checkpoint.get('epoch', 'unknown')
            best_dice = finetune_checkpoint.get('best_dice', 'unknown')
            print(f"已加载微调模型从 {finetune_checkpoint_path}")
            print(f"微调模型来自Epoch {best_epoch}，验证集Dice系数: {best_dice}")
        else:
            print(f"错误: 未找到微调模型 {finetune_checkpoint_path}")
            continue
        
        # 微调模型测试结果目录
        finetune_result_dir = os.path.join(results_dir, f"finetune_model_{target_dataset}")
        os.makedirs(finetune_result_dir, exist_ok=True)
        
        # 在测试集上评估微调模型
        criterion_bce = nn.BCELoss().to(device)
        test_metrics = evaluate_model_test(finetune_generator, test_loader, device, criterion_bce)
        
        # 输出测试结果
        print(f"\n微调模型在 {target_dataset} 上的测试结果:")
        print(f"Loss: {test_metrics['loss']:.4f}")
        for metric_name, metric_value in test_metrics['metrics'].items():
            print(f"{metric_name}: {metric_value:.4f}")
        print(f"ROC AUC: {test_metrics['curves']['roc_auc']:.4f}")
        print(f"PR AUC: {test_metrics['curves']['pr_auc']:.4f}")
        
        # 保存微调模型测试结果
        finetune_model_results[target_dataset] = {
            'test_metrics': test_metrics['metrics'],
            'curves': test_metrics['curves']
        }
        
        # 保存ROC和PR曲线
        if len(test_metrics['all_preds']) > 0 and len(test_metrics['all_targets']) > 0:
            curve_path = os.path.join(finetune_result_dir, "test_roc_pr_curves.png")
            generate_curves(
                predictions=test_metrics['all_preds'], 
                targets=test_metrics['all_targets'],
                save_path=curve_path,
                title=f"Finetune Model Test Results - {target_dataset}"
            )
            print(f"微调模型曲线已保存到: {curve_path}")
        
        # 保存分割示例图
        plot_segmentation_examples(finetune_generator, test_loader, finetune_result_dir, device, num_examples=5)
        
        # 收集微调模型结果数据
        results_data.append({
            'Dataset': target_dataset,
            'Model': 'Finetune Model',
            'Dice': test_metrics['metrics']['dice'],
            'IoU': test_metrics['metrics']['iou'],
            'Precision': test_metrics['metrics']['precision'],
            'Recall': test_metrics['metrics']['recall'],
            'Accuracy': test_metrics['metrics']['accuracy'],
            'ROC_AUC': test_metrics['curves']['roc_auc']
        })
    
    # 微调模型加入源数据集结果(使用源模型)以组成完整的微调结果集
    finetune_model_results[source_dataset] = source_model_results[source_dataset]
    
    # 生成微调模型所有数据集的合并ROC和PR曲线
    plot_combined_curves(finetune_model_results, results_dir, plot_type='finetune')
    
    # 生成微调模型所有数据集的评估指标对比图
    plot_combined_metrics(finetune_model_results, results_dir, plot_type='finetune')
    
    # 生成对比报告
    if results_data:
        # 创建完整的DataFrame
        results_df = pd.DataFrame(results_data)
        
        # 保存结果表格
        results_df.to_csv(os.path.join(results_dir, 'finetune_test_results.csv'), index=False)
        
        # 新增：生成总体评分表格
        # 分别为源模型和微调模型创建总体评分
        print("\n总体评分表格:")
        
        # 从结果中提取所有源模型数据
        source_results = results_df[results_df['Model'] == 'Source Model']
        source_results = source_results.drop(columns=['Model'])
        
        # 为微调模型创建一个单独的总体评分
        finetune_only_results = []
        # 首先添加源数据集的结果（源模型在源数据集上的表现）
        source_dataset_results = results_df[(results_df['Dataset'] == source_dataset) & (results_df['Model'] == 'Source Model')]
        if not source_dataset_results.empty:
            source_result_copy = source_dataset_results.iloc[0].copy()
            source_result_copy['Model'] = 'Finetune Model'  # 为保持一致而修改标签
            finetune_only_results.append(source_result_copy)
        
        # 然后添加目标数据集的微调模型结果
        for dataset in target_datasets:
            dataset_results = results_df[(results_df['Dataset'] == dataset) & (results_df['Model'] == 'Finetune Model')]
            if not dataset_results.empty:
                finetune_only_results.append(dataset_results.iloc[0])
        
        # 如果有微调结果，创建微调模型的DataFrame
        if finetune_only_results:
            finetune_results = pd.DataFrame(finetune_only_results)
            finetune_results = finetune_results.drop(columns=['Model'])
            
            # 计算微调模型的宏平均
            finetune_means = pd.DataFrame([{
                'Dataset': '宏平均(Macro Average)',
                'Dice': finetune_results['Dice'].mean(),
                'IoU': finetune_results['IoU'].mean(),
                'Precision': finetune_results['Precision'].mean(),
                'Recall': finetune_results['Recall'].mean(),
                'Accuracy': finetune_results['Accuracy'].mean(),
                'ROC_AUC': finetune_results['ROC_AUC'].mean()
            }])
            finetune_results = pd.concat([finetune_results, finetune_means], ignore_index=True)
            
            # 打印微调模型总体评分，格式化输出
            print("\n微调模型总体评分:")
            pd.set_option('display.float_format', '{:.6f}'.format)
            print(finetune_results[['Dataset', 'Dice', 'IoU', 'Precision', 'Recall', 'Accuracy', 'ROC_AUC']].to_string(index=False))
            
            # 将评分表保存到CSV
            finetune_results.to_csv(os.path.join(results_dir, 'finetune_overall_scores.csv'), index=False)
        
        # 生成源模型和微调模型的对比图
        comparison_dir = os.path.join(results_dir, 'finetune_comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 绘制对比图
        for target_dataset in target_datasets:
            if target_dataset in source_model_results and target_dataset in finetune_model_results:
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(f"源模型 vs 微调模型在 {target_dataset} 上的性能对比", fontsize=16)
                
                # 绘制Dice对比
                axes[0, 0].bar(['源模型', '微调模型'], 
                               [source_model_results[target_dataset]['test_metrics']['dice'], 
                                finetune_model_results[target_dataset]['test_metrics']['dice']])
                axes[0, 0].set_title('Dice系数对比')
                axes[0, 0].set_ylim(0, 1)
                
                # 绘制IoU对比
                axes[0, 1].bar(['源模型', '微调模型'], 
                               [source_model_results[target_dataset]['test_metrics']['iou'], 
                                finetune_model_results[target_dataset]['test_metrics']['iou']])
                axes[0, 1].set_title('IoU系数对比')
                axes[0, 1].set_ylim(0, 1)
                
                # 绘制精确率对比
                axes[0, 2].bar(['源模型', '微调模型'], 
                               [source_model_results[target_dataset]['test_metrics']['precision'], 
                                finetune_model_results[target_dataset]['test_metrics']['precision']])
                axes[0, 2].set_title('精确率对比')
                axes[0, 2].set_ylim(0, 1)
                
                # 绘制召回率对比
                axes[1, 0].bar(['源模型', '微调模型'], 
                               [source_model_results[target_dataset]['test_metrics']['recall'], 
                                finetune_model_results[target_dataset]['test_metrics']['recall']])
                axes[1, 0].set_title('召回率对比')
                axes[1, 0].set_ylim(0, 1)
                
                # 绘制准确率对比
                axes[1, 1].bar(['源模型', '微调模型'], 
                               [source_model_results[target_dataset]['test_metrics']['accuracy'], 
                                finetune_model_results[target_dataset]['test_metrics']['accuracy']])
                axes[1, 1].set_title('准确率对比')
                axes[1, 1].set_ylim(0, 1)
                
                # 绘制ROC AUC对比
                axes[1, 2].bar(['源模型', '微调模型'], 
                               [source_model_results[target_dataset]['curves']['roc_auc'], 
                                finetune_model_results[target_dataset]['curves']['roc_auc']])
                axes[1, 2].set_title('ROC AUC对比')
                axes[1, 2].set_ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(os.path.join(comparison_dir, f"{target_dataset}_comparison.png"))
                plt.close()
        
        # 创建汇总对比表格
        summary_data = []
        for dataset in target_datasets:
            if dataset in source_model_results and dataset in finetune_model_results:
                source_metrics = source_model_results[dataset]['test_metrics']
                finetune_metrics = finetune_model_results[dataset]['test_metrics']
                
                # 计算性能提升百分比
                improvement = {
                    'Dice': (finetune_metrics['dice'] - source_metrics['dice']) / source_metrics['dice'] * 100,
                    'IoU': (finetune_metrics['iou'] - source_metrics['iou']) / source_metrics['iou'] * 100,
                    'Precision': (finetune_metrics['precision'] - source_metrics['precision']) / source_metrics['precision'] * 100,
                    'Recall': (finetune_metrics['recall'] - source_metrics['recall']) / source_metrics['recall'] * 100,
                    'Accuracy': (finetune_metrics['accuracy'] - source_metrics['accuracy']) / source_metrics['accuracy'] * 100,
                    'ROC_AUC': (finetune_model_results[dataset]['curves']['roc_auc'] - source_model_results[dataset]['curves']['roc_auc']) / source_model_results[dataset]['curves']['roc_auc'] * 100
                }
                
                summary_data.append({
                    'Dataset': dataset,
                    'Source_Dice': source_metrics['dice'],
                    'Finetune_Dice': finetune_metrics['dice'],
                    'Dice_Improvement(%)': improvement['Dice'],
                    'Source_IoU': source_metrics['iou'],
                    'Finetune_IoU': finetune_metrics['iou'],
                    'IoU_Improvement(%)': improvement['IoU'],
                    'Source_Precision': source_metrics['precision'],
                    'Finetune_Precision': finetune_metrics['precision'],
                    'Precision_Improvement(%)': improvement['Precision'],
                    'Source_Recall': source_metrics['recall'],
                    'Finetune_Recall': finetune_metrics['recall'],
                    'Recall_Improvement(%)': improvement['Recall'],
                })
        
        # 创建并保存汇总表格
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(results_dir, 'finetune_improvement_summary.csv'), index=False)
        
        # 打印结果摘要
        print("\n迁移学习测试结果摘要:")
        
        # 显示只有Dataset, Model, Dice, IoU列的简化表格
        simplified_df = results_df[['Dataset', 'Model', 'Dice', 'IoU']]
        print(simplified_df.to_string(index=False))
        
        print("\n性能提升摘要:")
        simplified_summary = summary_df[['Dataset', 'Dice_Improvement(%)', 'IoU_Improvement(%)']]
        print(simplified_summary.to_string(index=False))
        
        print(f"\n详细结果已保存到: {results_dir}")
    else:
        print("错误: 没有任何测试结果数据")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试迁移学习模型在息肉分割任务上的表现")
    parser.add_argument('--data_root', type=str, default='data', help='数据集根目录')
    parser.add_argument('--results_dir', type=str, default='finetune_test_results', help='测试结果保存目录')
    parser.add_argument('--source_models_dir', type=str, default='models/finetune', help='源模型目录')
    parser.add_argument('--finetune_models_dir', type=str, default='models/finetune', help='微调模型目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    
    args = parser.parse_args()
    
    test_finetune_model(args) 