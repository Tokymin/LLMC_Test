#!/mnt/share/toky/CondaEnvs/LLMC/bin/python
import os
import sys
import argparse
import random
import socket
import subprocess
from typing import List


def find_available_port(min_port: int = 10000, max_port: int = 60000) -> int:
    """查找可用端口"""
    for _ in range(100):  # 最多尝试100次
        port = random.randint(min_port, max_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("找不到可用端口！")


def main(args):
    # 设置环境变量
    os.environ["PYTHONPATH"] = f"{args.llmc_path}:{os.environ.get('PYTHONPATH', '')}"

    # 查找可用端口
    master_port = find_available_port()

    # 构建命令行参数
    cmd = [
        sys.executable,  # 使用当前Python解释器
        "-m", "torch.distributed.run",
        "--nnodes", str(args.nnodes),
        "--nproc_per_node", str(args.nproc_per_node),
        "--rdzv_id", str(master_port),
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", f"{args.master_addr}:{master_port}",
        f"{args.llmc_path}/llmc/__main__.py",
        "--config", args.config,
        "--task_id", str(master_port)
    ]

    # 添加CUDA可见设备
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.cuda_visible_devices))

    # 打印执行命令
    print("[执行命令]", " ".join(cmd))

    # 直接运行（不后台执行）
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"执行失败！错误码: {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMC量化训练启动脚本")

    # 必选参数
    parser.add_argument("--llmc_path", required=True,
                        help="LLMC项目根目录路径")
    parser.add_argument("--config", required=True,
                        help="量化配置文件路径")

    # 可选参数
    parser.add_argument("--task_name", default="awq_w_only",
                        help="任务名称（用于日志记录）")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="分布式节点数量")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="每个节点的进程数")
    parser.add_argument("--master_addr", default="127.0.0.1",
                        help="主节点地址")
    parser.add_argument("--cuda_visible_devices", type=lambda s: [int(item) for item in s.split(',')],
                        help="可见的GPU设备ID列表，用逗号分隔")

    args = parser.parse_args()

    # 验证路径存在
    if not os.path.exists(args.llmc_path):
        raise ValueError(f"LLMC路径不存在: {args.llmc_path}")
    if not os.path.exists(args.config):
        raise ValueError(f"配置文件不存在: {args.config}")

    main(args)
