import os
import argparse
import subprocess
import sys

def main():
    """
    运行test_finetune_model.py的主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试迁移学习模型')
    parser.add_argument('--data_root', type=str, default='data', help='数据集根目录')
    parser.add_argument('--results_dir', type=str, default='finetune_test_results', help='测试结果保存目录')
    parser.add_argument('--source_models_dir', type=str, default='models/finetune', help='源模型目录')
    parser.add_argument('--finetune_models_dir', type=str, default='models/finetune', help='微调模型目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    
    args = parser.parse_args()
    
    # 构建test_finetune_model.py的命令行参数
    cmd_args = [
        sys.executable,  # 当前Python解释器路径
        'test_finetune_model.py',
        f'--data_root={args.data_root}',
        f'--results_dir={args.results_dir}',
        f'--source_models_dir={args.source_models_dir}',
        f'--finetune_models_dir={args.finetune_models_dir}',
        f'--batch_size={args.batch_size}'
    ]
    
    print("=" * 80)
    print("迁移学习模型测试开始")
    print("=" * 80)
    print(f"数据集根目录: {args.data_root}")
    print(f"测试结果保存目录: {args.results_dir}")
    print(f"源模型目录: {args.source_models_dir}")
    print(f"微调模型目录: {args.finetune_models_dir}")
    print(f"批次大小: {args.batch_size}")
    print("=" * 80)
    print("执行命令:", " ".join(cmd_args))
    print("=" * 80)
    
    # 执行test_finetune_model.py
    try:
        # 使用subprocess.run直接运行，不捕获输出，让输出直接显示在控制台
        process = subprocess.run(cmd_args)
        
        if process.returncode == 0:
            print("\n" + "=" * 80)
            print("迁移学习模型测试成功完成！")
            print("=" * 80)
            print(f"\n测试结果保存在: {args.results_dir}")
            print(f"\n可视化报告目录: {os.path.join(args.results_dir, 'finetune_comparison')}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print(f"迁移学习模型测试失败，返回代码: {process.returncode}")
            print("=" * 80)
        
    except Exception as e:
        print(f"执行test_finetune_model.py时发生错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 