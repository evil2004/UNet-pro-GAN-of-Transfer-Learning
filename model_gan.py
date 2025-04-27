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

# 导入可视化模块
from visualization import (
    generate_curves, 
    plot_training_history, 
    plot_segmentation_examples, 
    create_comparison_plots,
    plot_metrics_summary,
    plot_combined_training_history
)

# 设置随机种子确保实验可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 使用PyTorch原生实现评估指标，利用GPU加速
def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    计算Dice系数 = 2*|X∩Y|/(|X|+|Y|)
    
    参数:
        y_pred: 预测的二值化掩码
        y_true: 真实的二值化掩码
        smooth: 平滑项，防止分母为0
        
    返回:
        Dice系数值（PyTorch张量）
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    
    return dice

def iou_coefficient(y_pred, y_true, smooth=1e-6):
    """
    计算IoU系数(交并比) = |X∩Y|/|X∪Y|
    
    参数:
        y_pred: 预测的二值化掩码
        y_true: 真实的二值化掩码
        smooth: 平滑项，防止分母为0
        
    返回:
        IoU系数值（PyTorch张量）
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def precision_score(y_pred, y_true, smooth=1e-6):
    """
    计算精确率 = TP/(TP+FP)
    
    参数:
        y_pred: 预测的二值化掩码
        y_true: 真实的二值化掩码
        smooth: 平滑项，防止分母为0
        
    返回:
        精确率值（PyTorch张量）
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    true_positives = (y_pred * y_true).sum()
    predicted_positives = y_pred.sum()
    
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    
    return precision

def recall_score(y_pred, y_true, smooth=1e-6):
    """
    计算召回率 = TP/(TP+FN)
    
    参数:
        y_pred: 预测的二值化掩码
        y_true: 真实的二值化掩码
        smooth: 平滑项，防止分母为0
        
    返回:
        召回率值（PyTorch张量）
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    true_positives = (y_pred * y_true).sum()
    actual_positives = y_true.sum()
    
    recall = (true_positives + smooth) / (actual_positives + smooth)
    
    return recall

def accuracy_score(y_pred, y_true):
    """
    计算准确率 = (TP+TN)/(TP+TN+FP+FN)
    
    参数:
        y_pred: 预测的二值化掩码
        y_true: 真实的二值化掩码
        
    返回:
        准确率值（PyTorch张量）
    """
    y_pred = (y_pred > 0.5).float().view(-1)
    y_true = (y_true > 0.5).float().view(-1)
    
    correct = (y_pred == y_true).float().sum()
    total = y_true.numel()
    
    accuracy = correct / total
    
    return accuracy

# 设置matplotlib字体，使用英文避免中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置英文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 定义UNet++生成器
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetPlusPlus, self).__init__()
        
        # 编码器
        self.enc1_1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2_1 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3_1 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4_1 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # 瓶颈
        self.bottleneck = self.conv_block(512, 1024)
        
        # 解码器和嵌套连接
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4_1 = self.conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3_2 = self.conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_3 = self.conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_4 = self.conv_block(128, 64)
        
        # UNet++的嵌套跳跃连接
        self.up3_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3_1 = self.conv_block(512, 256)
        
        self.up2_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_1 = self.conv_block(256, 128)
        
        self.up2_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_2 = self.conv_block(384, 128)
        
        self.up1_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_1 = self.conv_block(128, 64)
        
        self.up1_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_2 = self.conv_block(192, 64)
        
        self.up1_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_3 = self.conv_block(256, 64)
        
        # 最终输出层
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码器路径
        x1_1 = self.enc1_1(x)
        x2_0 = self.pool1(x1_1)
        x2_1 = self.enc2_1(x2_0)
        x3_0 = self.pool2(x2_1)
        x3_1 = self.enc3_1(x3_0)
        x4_0 = self.pool3(x3_1)
        x4_1 = self.enc4_1(x4_0)
        x5_0 = self.pool4(x4_1)
        # 瓶颈
        x5_1 = self.bottleneck(x5_0)
        
        # 解码器路径 - 密集嵌套结构
        x4_2 = self.up4(x5_1)
        x4_2 = torch.cat([x4_1, x4_2], dim=1)
        x4_2 = self.dec4_1(x4_2)
        x3_2 = self.up3(x4_2)
        x3_2 = torch.cat([x3_1, x3_2], dim=1)
        x3_2 = self.dec3_2(x3_2)
        x2_3 = self.up2(x3_2)
        x2_3 = torch.cat([x2_1, x2_3], dim=1)
        x2_3 = self.dec2_3(x2_3)
        x1_4 = self.up1(x2_3)
        x1_4 = torch.cat([x1_1, x1_4], dim=1)
        x1_4 = self.dec1_4(x1_4)
        
        output = self.output(x1_4)
        return torch.sigmoid(output)

# 定义PatchGAN判别器
class Discriminator(nn.Module):
    def __init__(self, in_channels=4):  # 输入是原图+掩码或生成的掩码，所以是3+1
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            discriminator_block(in_channels, 64, normalization=False),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )
        
    def forward(self, img, mask):
        # 将图像和掩码连接在一起
        img_mask = torch.cat([img, mask], dim=1)
        return self.model(img_mask)

# 数据集类
class PolypDataset(Dataset):
    def __init__(self, root_dir, dataset_name, split="train", transform=None):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        
        self.image_dir = os.path.join(root_dir, dataset_name, split, "images")
        self.mask_dir = os.path.join(root_dir, dataset_name, split, "masks")
        
        self.image_paths = sorted([os.path.join(self.image_dir, img_name) for img_name in os.listdir(self.image_dir) if img_name.endswith('.jpg')])
        self.mask_paths = sorted([os.path.join(self.mask_dir, img_name) for img_name in os.listdir(self.mask_dir) if img_name.endswith('.jpg')])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return {"image": image, "mask": mask}

# 损失函数
class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

# 添加Dice损失类
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        
        return 1 - dice  # 返回损失值，越小越好

# 定义组合损失函数，结合BCE和Dice
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        
    def forward(self, y_pred, y_true):
        # 注意：BCE损失在autocast中不安全，需要确保在使用时不在autocast上下文中
        bce = self.bce_loss(y_pred, y_true)
        dice = self.dice_loss(y_pred, y_true)
        return self.alpha * bce + (1 - self.alpha) * dice

# 定义训练函数
def train_model(generator, discriminator, dataloader, val_dataloader, num_epochs, device, dataset_name, save_dir, lr=0.0002):
    criterion_gan = GANLoss(gan_mode='lsgan').to(device)
    
    # 替换BCE损失为组合损失
    criterion_pixelwise = CombinedLoss(alpha=0.3).to(device)
    
    # 修改像素损失权重，防止模式崩溃
    lambda_pixel = 100  # 从10改回100
    
    # 为判别器和生成器设置不同的学习率
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr*0.5, betas=(0.5, 0.999))  # 降低判别器学习率
    
    # 使用StepLR代替CosineAnnealingLR，防止学习率降得过快
    lr_scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.9)
    lr_scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.9)
    
    # 设置梯度裁剪阈值
    max_grad_norm = 1.0
    
    scaler = GradScaler(enabled=True)  # 用于混合精度训练
    
    os.makedirs(os.path.join(save_dir, dataset_name), exist_ok=True)
    
    best_dice = 0.0
    
    # 添加早停参数
    patience = 10  # 如果10个epoch内验证指标没有提升，则停止训练
    patience_counter = 0
    
    # 用于记录训练过程中的指标，绘制训练曲线
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
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
            
            # 确保每个epoch都更新判别器
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
            # 确保数据类型一致
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
        
        epoch_g_loss /= len(dataloader)
        epoch_d_loss /= len(dataloader)
        epoch_pixel_loss /= len(dataloader)
        
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
            dataloader=val_dataloader,
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
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(save_dir, dataset_name, 'best_model.pth'))
            
            # 保存最佳模型的验证指标和预测结果（用于绘制最佳ROC曲线）
            best_val_metrics = val_metrics['metrics'].copy()
            best_val_predictions = val_metrics.get('all_preds', []).copy() if 'all_preds' in val_metrics else []
            best_val_targets = val_metrics.get('all_targets', []).copy() if 'all_targets' in val_metrics else []
            
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
            }, os.path.join(save_dir, dataset_name, f'model_epoch_{epoch+1}.pth'))
    
    # 使用visualization模块绘制训练曲线
    plot_training_history(train_losses, val_metrics_history, save_dir, dataset_name)
    
    # 保存最佳模型的ROC和PR曲线
    if best_val_predictions and best_val_targets:
        curve_path = os.path.join(save_dir, dataset_name, 'best_roc_pr_curves.png')
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
    return generator, {
        'train_losses': train_losses,
        'val_metrics_history': val_metrics_history,
        'best_val_metrics': best_val_metrics,
        'best_val_predictions': best_val_predictions,
        'best_val_targets': best_val_targets
    }

# 定义评估函数
def evaluate_model(model, dataloader, device, criterion, metrics=None, epoch=None, writer=None, prefix="val", gen_curves=False):
    """
    评估模型性能
    
    参数:
        model: 要评估的模型
        dataloader: 数据加载器
        device: 使用的设备
        criterion: 损失函数
        metrics: 可选，要计算的评估指标字典
        epoch: 可选，当前训练轮次
        writer: 可选，tensorboard写入器
        prefix: 可选，指标前缀
        gen_curves: 是否生成ROC和PR曲线
        
    返回:
        包含评估结果的字典
    """
    model.eval()
    metrics = metrics or {}
    metric_values = {name: 0 for name in metrics}
    num_samples = 0
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{prefix} evaluation"):
            # 读取batch数据 - 兼容不同的数据加载器格式
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
            for name, metric_fn in metrics.items():
                metric_values[name] += metric_fn(binary_outputs, binary_masks).item() * images.size(0)
            
            # 累计批次大小和损失
            num_samples += images.size(0)
            total_loss += loss.item() * images.size(0)
            
            # 收集预测和目标，用于ROC和PR曲线
            # 确保收集二进制格式的target
            if gen_curves:
                all_preds.extend(outputs.view(-1).cpu().numpy())
                all_targets.extend(binary_masks.view(-1).cpu().numpy())  # 使用二值化的掩码
    
    # 计算平均损失和指标
    avg_loss = total_loss / num_samples
    avg_metrics = {name: value / num_samples for name, value in metric_values.items()}
    
    # 打印结果
    print(f"{prefix.capitalize()} Loss: {avg_loss:.4f}")
    for name, value in avg_metrics.items():
        print(f"{prefix.capitalize()} {name}: {value:.4f}")
    
    # 记录到tensorboard
    if writer is not None and epoch is not None:
        writer.add_scalar(f'{prefix}/loss', avg_loss, epoch)
        for name, value in avg_metrics.items():
            writer.add_scalar(f'{prefix}/{name}', value, epoch)
    
    # 生成ROC和PR曲线
    curves_data = {}
    if gen_curves and all_preds and all_targets:
        # 创建保存路径
        save_path = None
        if epoch is not None:
            # 确保目录存在
            curves_dir = os.path.join('results', 'curves')
            os.makedirs(curves_dir, exist_ok=True)
            save_path = os.path.join(curves_dir, f"{prefix}_epoch_{epoch}.png")
        
        curves_data = generate_curves(
            predictions=all_preds, 
            targets=all_targets,
            save_path=save_path,
            title=f"Test Set - {prefix.capitalize()}" if prefix == "test" else None
        )
        
        # 如果是验证集且有writer，则记录曲线相关指标
        if writer is not None and epoch is not None and prefix == "val":
            writer.add_scalar(f'{prefix}/roc_auc', curves_data['roc_auc'], epoch)
            writer.add_scalar(f'{prefix}/pr_auc', curves_data['pr_auc'], epoch)
    
    # 返回结果
    return {
        'loss': avg_loss,
        'metrics': avg_metrics,
        'curves': curves_data,
        'all_preds': all_preds,  # 返回所有预测值
        'all_targets': all_targets  # 返回所有目标值
    }

# 主函数
def main(args):
    # 设置路径
    data_root = args.data_root
    results_dir = args.results_dir
    models_dir = args.models_dir
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # 设置随机种子
    set_seed()
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 定义数据集名称
    dataset_names = ['Kvasir-SEG', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'CVC-ColonDB']
    
    # 是否进行统一训练
    if args.unified_model:
        # 创建一个通用模型
        generator = UNetPlusPlus(in_channels=3, out_channels=1).to(device)
        discriminator = Discriminator(in_channels=4).to(device)
        
        # 整合所有数据集
        all_train_datasets = []
        all_val_datasets = []
        
        for dataset_name in dataset_names:
            # 加载数据集但不训练
            train_dataset = PolypDataset(data_root, dataset_name, split="train", transform=transform)
            val_dataset = PolypDataset(data_root, dataset_name, split="validation", transform=transform)
            all_train_datasets.append(train_dataset)
            all_val_datasets.append(val_dataset)
        
        # 合并数据集
        combined_train_dataset = ConcatDataset(all_train_datasets)
        combined_val_dataset = ConcatDataset(all_val_datasets)
        
        # 创建数据加载器
        train_loader = DataLoader(combined_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(combined_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        print(f"合并训练集大小: {len(combined_train_dataset)}, 合并验证集大小: {len(combined_val_dataset)}")
        
        # 训练统一模型
        print(f"在所有数据集上训练中...")
        generator, train_history = train_model(
            generator, 
            discriminator, 
            train_loader, 
            val_loader, 
            num_epochs=args.epochs,
            device=device,
            dataset_name="combined_datasets",  # 使用组合名称
            save_dir=models_dir,
            lr=args.lr
        )
        
        # 保存最终模型到统一位置
        unified_model_path = os.path.join(models_dir, "unified_model.pth")
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
        }, unified_model_path)
        
        print(f"统一模型训练完成，已保存到 {unified_model_path}")
    else:
        # 分别训练每个数据集
        all_train_histories = {}
        
        for dataset_name in dataset_names:
            print(f"\n处理数据集: {dataset_name}")
            
            # 创建数据集和数据加载器
            train_dataset = PolypDataset(data_root, dataset_name, split="train", transform=transform)
            val_dataset = PolypDataset(data_root, dataset_name, split="validation", transform=transform)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            
            print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
            
            # 创建模型
            generator = UNetPlusPlus(in_channels=3, out_channels=1).to(device)
            discriminator = Discriminator(in_channels=4).to(device)
            
            # 如果只是测试，则跳过该数据集
            if args.only_test:
                print(f"跳过训练 {dataset_name}，请使用test_model.py进行测试")
                continue
            
            # 训练模型
            print(f"在 {dataset_name} 上训练中...")
            generator, train_history = train_model(
                generator, 
                discriminator, 
                train_loader, 
                val_loader, 
                num_epochs=args.epochs,
                device=device,
                dataset_name=dataset_name,
                save_dir=models_dir,
                lr=args.lr
            )
            
            # 收集训练历史
            all_train_histories[dataset_name] = train_history
            
            print(f"{dataset_name} 训练完成，最佳模型已保存到 {os.path.join(models_dir, dataset_name, 'best_model.pth')}")
        
        # 绘制所有数据集的训练历史在同一张图上
        plot_combined_training_history(all_train_histories, models_dir)
    
    print("\n所有数据集训练完成，请使用test_model.py进行测试评估")

if __name__ == "__main__":
    # 这部分内容已移动到data_partitioning.py中
    pass 