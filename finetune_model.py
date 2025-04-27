import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tqdm import tqdm
import pandas as pd
import time
import datetime
import cv2
from torch.amp import autocast, GradScaler
import warnings
import argparse

# 导入可视化模块
from visualization import (
    generate_curves, 
    plot_training_history, 
    plot_segmentation_examples, 
    create_comparison_plots,
    plot_metrics_summary,
    plot_combined_training_history
)

# 从model_gan.py导入必要的类和函数
from model_gan import (
    UNetPlusPlus, 
    Discriminator, 
    PolypDataset,
    GANLoss, 
    DiceLoss, 
    CombinedLoss,
    set_seed,
    dice_coefficient, 
    iou_coefficient, 
    precision_score, 
    recall_score, 
    accuracy_score,
    evaluate_model
)

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 定义迁移学习训练函数
def train_with_transfer(generator, discriminator, train_loader, val_loader, device, save_dir, dataset_name, num_epochs=100, lr=0.0002, is_finetune=False):
    """
    训练或微调模型
    
    参数:
        generator: 生成器模型
        discriminator: 判别器模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 使用的设备
        save_dir: 保存目录
        dataset_name: 数据集名称
        num_epochs: 训练轮次
        lr: 学习率
        is_finetune: 是否是微调阶段
    """
    criterion_gan = GANLoss(gan_mode='lsgan').to(device)
    criterion_pixelwise = CombinedLoss(alpha=0.3).to(device)
    
    # 修改像素损失权重
    lambda_pixel = 100
    
    # 微调阶段使用较小的学习率
    if is_finetune:
        lr = lr * 0.2
        print(f"微调模式: 学习率减小到 {lr}")
    
    # 设置优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr*0.5, betas=(0.5, 0.999))
    
    # 修改学习率调度策略，基础训练使用StepLR，与model_gan一致
    if is_finetune:
        # 微调阶段使用余弦退火
        lr_scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=lr*0.1)
        lr_scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs, eta_min=lr*0.1)
    else:
        # 基础训练阶段使用StepLR，与model_gan.py保持一致
        lr_scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.9)
        lr_scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.9)
    
    # 设置梯度裁剪阈值
    max_grad_norm = 1.0
    
    scaler = GradScaler(enabled=True)  # 用于混合精度训练
    
    # 创建保存目录
    model_save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    best_dice = 0.0
    
    # 添加早停参数
    patience = 10  # 如果10个epoch内验证指标没有提升，则停止训练
    patience_counter = 0
    
    # 微调阶段减少早停耐心值
    if is_finetune:
        patience = 5
    
    # 用于记录训练过程中的指标
    train_losses = []
    val_metrics_history = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'accuracy': []
    }
    best_val_metrics = None
    best_val_predictions = None
    best_val_targets = None
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_pixel_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            real_images = batch["image"].to(device)
            real_masks = batch["mask"].to(device)
            
            # 添加噪声到真实标签，帮助稳定训练
            real_label_noise = 0.1 * torch.randn(real_masks.size()).to(device)
            real_masks_noisy = real_masks + real_label_noise
            real_masks_noisy = torch.clamp(real_masks_noisy, 0.0, 1.0)
            
            batch_size = real_images.size(0)
            
            # ---------------------
            #  训练判别器
            # ---------------------
            optimizer_D.zero_grad()
            
            # 使用自动混合精度训练
            with autocast(device_type='cuda', enabled=True):
                # 生成假掩码
                fake_masks = generator(real_images)
                
                # 真实样本判别
                pred_real = discriminator(real_images, real_masks_noisy)
                loss_d_real = criterion_gan(pred_real, True)
                
                # 假样本判别
                pred_fake = discriminator(real_images, fake_masks.detach())
                loss_d_fake = criterion_gan(pred_fake, False)
                
                # 总判别器损失
                loss_d = (loss_d_real + loss_d_fake) * 0.5
            
            scaler.scale(loss_d).backward()
            
            # 应用梯度裁剪
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
            
            scaler.step(optimizer_D)
            
            epoch_d_loss += loss_d.item()
            
            # ---------------------
            #  训练生成器
            # ---------------------
            optimizer_G.zero_grad()
            
            with autocast(device_type='cuda', enabled=True):
                # 再次生成假掩码
                fake_masks = generator(real_images)
                
                # GAN损失（生成器试图欺骗判别器）
                pred_fake = discriminator(real_images, fake_masks)
                loss_g_gan = criterion_gan(pred_fake, True)
                
                # 只计算Dice损失部分
                dice_loss = criterion_pixelwise.dice_loss(fake_masks, real_masks)
            
            # 将BCE损失计算移到autocast上下文外，避免不安全的警告
            fake_masks_float = fake_masks.float()
            real_masks_float = real_masks.float()
            bce_loss = criterion_pixelwise.bce_loss(fake_masks_float, real_masks_float)
            
            # 手动组合损失
            loss_g_pixel = criterion_pixelwise.alpha * bce_loss + (1 - criterion_pixelwise.alpha) * dice_loss
            
            # 总生成器损失
            loss_g = loss_g_gan + lambda_pixel * loss_g_pixel
            
            scaler.scale(loss_g).backward()
            
            # 应用梯度裁剪
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
            
            scaler.step(optimizer_G)
            
            scaler.update()
            
            epoch_g_loss += loss_g.item()
            epoch_pixel_loss += loss_g_pixel.item()
        
        epoch_g_loss /= len(train_loader)
        epoch_d_loss /= len(train_loader)
        epoch_pixel_loss /= len(train_loader)
        
        # 记录训练损失
        train_losses.append({
            'epoch': epoch + 1,
            'generator_loss': epoch_g_loss,
            'discriminator_loss': epoch_d_loss,
            'pixel_loss': epoch_pixel_loss
        })
        
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Generator Loss: {epoch_g_loss:.4f}, Discriminator Loss: {epoch_d_loss:.4f}, Pixel Loss: {epoch_pixel_loss:.4f}")
        print(f"LR - Generator: {optimizer_G.param_groups[0]['lr']:.6f}, Discriminator: {optimizer_D.param_groups[0]['lr']:.6f}")
        
        # 验证
        val_metrics = evaluate_model(
            model=generator,
            dataloader=val_loader,
            device=device,
            criterion=criterion_pixelwise,
            metrics={
                'dice': dice_coefficient,
                'iou': iou_coefficient,
                'precision': precision_score,
                'recall': recall_score,
                'accuracy': accuracy_score
            },
            epoch=epoch,
            prefix="val",
            gen_curves=True
        )
        
        # 记录验证指标
        for metric_name, metric_value in val_metrics['metrics'].items():
            val_metrics_history[metric_name].append(metric_value)
        
        print(f"Validation - Dice: {val_metrics['metrics']['dice']:.4f}, IoU: {val_metrics['metrics']['iou']:.4f}, "
              f"Precision: {val_metrics['metrics']['precision']:.4f}, Recall: {val_metrics['metrics']['recall']:.4f}, "
              f"Accuracy: {val_metrics['metrics']['accuracy']:.4f}")
        
        # 保存最佳模型
        if val_metrics['metrics']['dice'] > best_dice:
            best_dice = val_metrics['metrics']['dice']
            save_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_dice': best_dice,
            }, save_path)
            
            # 保存最佳模型的验证指标和预测结果
            best_val_metrics = val_metrics['metrics'].copy()
            best_val_predictions = val_metrics.get('all_preds', []).copy() if 'all_preds' in val_metrics else []
            best_val_targets = val_metrics.get('all_targets', []).copy() if 'all_targets' in val_metrics else []
            
            print(f"保存最佳模型到 {save_path}")
            
            # 重置早停计数器
            patience_counter = 0
        else:
            # 如果性能没有提高，增加计数器
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
        
        # 早停机制
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # 每10个epoch保存一次
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'dice': val_metrics['metrics']['dice'],
            }, os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth'))
    
    # 绘制训练曲线
    plot_training_history(train_losses, val_metrics_history, save_dir, dataset_name)
    
    # 保存最佳模型的ROC和PR曲线
    if best_val_predictions and best_val_targets:
        curve_path = os.path.join(model_save_dir, 'best_roc_pr_curves.png')
        generate_curves(
            predictions=best_val_predictions,
            targets=best_val_targets,
            save_path=curve_path,
            title=f"Best Model ROC & PR Curves - {dataset_name}"
        )
        print(f"Best model curves saved to: {curve_path}")
    
    # 训练结束时输出最佳验证指标
    print("\n" + "=" * 50)
    print(f"训练完成! 最佳模型性能 ({dataset_name}):")
    print(f"最佳Dice系数: {best_dice:.4f}")
    if best_val_metrics:
        print(f"IoU: {best_val_metrics['iou']:.4f}")
        print(f"Precision: {best_val_metrics['precision']:.4f}")
        print(f"Recall: {best_val_metrics['recall']:.4f}")
        print(f"Accuracy: {best_val_metrics['accuracy']:.4f}")
    print("=" * 50)
    
    # 返回训练历史和最佳指标
    return generator, discriminator, {
        'train_losses': train_losses,
        'val_metrics_history': val_metrics_history,
        'best_val_metrics': best_val_metrics,
        'best_val_predictions': best_val_predictions,
        'best_val_targets': best_val_targets
    }

def finetune_on_dataset(source_model_path, data_root, dataset_name, save_dir, device, args):
    """
    在特定数据集上微调模型
    
    参数:
        source_model_path: 源模型路径
        data_root: 数据根目录
        dataset_name: 目标数据集名称
        save_dir: 保存目录
        device: 使用的设备
        args: 命令行参数
    """
    print(f"\n在 {dataset_name} 上微调模型...")
    
    # 加载预训练的模型
    generator = UNetPlusPlus(in_channels=3, out_channels=1).to(device)
    discriminator = Discriminator(in_channels=4).to(device)
    
    checkpoint = torch.load(source_model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    print(f"已加载源模型: {source_model_path}")
    
    # 移除微调时的数据增强，只使用基本数据转换
    basic_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 加载目标数据集，使用基本数据转换
    train_dataset = PolypDataset(data_root, dataset_name, split="train", transform=basic_transform)
    val_dataset = PolypDataset(data_root, dataset_name, split="validation", transform=basic_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    print(f"已移除数据增强，仅使用基本数据转换（调整大小和转换为张量）")
    
    # 微调模型
    generator, discriminator, finetune_history = train_with_transfer(
        generator,
        discriminator,
        train_loader,
        val_loader,
        device=device,
        save_dir=os.path.join(save_dir, "finetune"),
        dataset_name=dataset_name,
        num_epochs=args.finetune_epochs,
        lr=args.lr,
        is_finetune=True
    )
    
    print(f"在 {dataset_name} 上微调完成！")
    
    return finetune_history

def main(args):
    # 设置路径
    data_root = args.data_root
    results_dir = args.results_dir
    models_dir = args.models_dir
    
    # 确保目录存在
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    finetune_dir = os.path.join(models_dir, "finetune")
    os.makedirs(finetune_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 基本数据转换
    basic_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 定义数据集名称
    source_dataset = 'Kvasir-SEG'  # 源数据集
    target_datasets = ['CVC-ClinicDB', 'ETIS-LaribPolypDB', 'CVC-ColonDB']  # 目标数据集
    
    # 第一步：在源数据集上训练
    print(f"\n第一步：在源数据集 {source_dataset} 上训练基础模型...")
    
    # 加载源数据集，使用基础数据转换而非增强数据转换
    train_dataset = PolypDataset(data_root, source_dataset, split="train", transform=basic_transform)
    val_dataset = PolypDataset(data_root, source_dataset, split="validation", transform=basic_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"源数据集 - 训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    print(f"已移除数据增强，仅使用基本数据转换（调整大小和转换为张量）")
    
    # 创建模型
    generator = UNetPlusPlus(in_channels=3, out_channels=1).to(device)
    discriminator = Discriminator(in_channels=4).to(device)
    
    # 训练源模型
    source_generator, source_discriminator, source_history = train_with_transfer(
        generator,
        discriminator,
        train_loader,
        val_loader,
        device=device,
        save_dir=finetune_dir,
        dataset_name=source_dataset,
        num_epochs=args.epochs,
        lr=args.lr,
        is_finetune=False
    )
    
    # 获取源模型的最佳模型路径
    source_model_path = os.path.join(finetune_dir, source_dataset, 'best_model.pth')
    print(f"源模型训练完成，最佳模型保存到: {source_model_path}")
    
    # 第二步：在目标数据集上微调模型
    print("\n第二步：在目标数据集上微调模型...")
    
    # 收集所有微调历史
    all_finetune_histories = {source_dataset: source_history}
    
    for target_dataset in target_datasets:
        finetune_history = finetune_on_dataset(
            source_model_path=source_model_path,
            data_root=data_root,
            dataset_name=target_dataset,
            save_dir=models_dir,
            device=device,
            args=args
        )
        all_finetune_histories[target_dataset] = finetune_history
    
    # 绘制所有数据集的训练历史对比图
    plot_combined_training_history(all_finetune_histories, finetune_dir)
    
    print("\n迁移学习训练完成！请使用test_model.py评估微调后的模型。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="迁移学习策略：先在Kvasir-SEG上训练，再微调到其他数据集")
    parser.add_argument('--data_root', type=str, default='data', help='数据集根目录')
    parser.add_argument('--results_dir', type=str, default='results_finetune', help='结果保存目录')
    parser.add_argument('--models_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='源模型训练轮次')
    parser.add_argument('--finetune_epochs', type=int, default=50, help='微调轮次')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    main(args) 