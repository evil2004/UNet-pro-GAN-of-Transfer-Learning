import os
import argparse
import subprocess
import sys

def main():
    """
    运行finetune_model.py的主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行迁移学习训练')
    parser.add_argument('--data_root', type=str, default='data', help='数据集根目录')
    parser.add_argument('--results_dir', type=str, default='results_finetune', help='结果保存目录')
    parser.add_argument('--models_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='源模型训练轮次')
    parser.add_argument('--finetune_epochs', type=int, default=50, help='微调轮次')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    
    args = parser.parse_args()
    
    # 构建finetune_model.py的命令行参数
    cmd_args = [
        sys.executable,  # 当前Python解释器路径
        'finetune_model.py',
        f'--data_root={args.data_root}',
        f'--results_dir={args.results_dir}',
        f'--models_dir={args.models_dir}',
        f'--batch_size={args.batch_size}',
        f'--epochs={args.epochs}',
        f'--finetune_epochs={args.finetune_epochs}',
        f'--lr={args.lr}'
    ]
    
    print("=" * 80)
    print("迁移学习训练开始")
    print("=" * 80)
    print(f"数据集根目录: {args.data_root}")
    print(f"结果保存目录: {args.results_dir}")
    print(f"模型保存目录: {args.models_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"源模型训练轮次: {args.epochs}")
    print(f"微调轮次: {args.finetune_epochs}")
    print(f"学习率: {args.lr}")
    print("=" * 80)
    print("执行命令:", " ".join(cmd_args))
    print("=" * 80)
    
    # 执行finetune_model.py - 修改这部分以确保实时显示进度条
    try:
        # 使用subprocess.run直接运行，不捕获输出，让输出直接显示在控制台
        process = subprocess.run(cmd_args)
        
        if process.returncode == 0:
            print("\n" + "=" * 80)
            print("迁移学习训练成功完成！")
            print("=" * 80)
            print(f"\n训练结果保存在: {args.results_dir}")
            print(f"微调后的模型保存在: {os.path.join(args.models_dir, 'finetune')}")
            print(f"\n使用以下命令评估微调后的模型:")
            print(f"python test_model.py --models_dir {os.path.join(args.models_dir, 'finetune')}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print(f"迁移学习训练失败，返回代码: {process.returncode}")
            print("=" * 80)
        
    except Exception as e:
        print(f"执行finetune_model.py时发生错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 