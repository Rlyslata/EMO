import subprocess

def main():
    # 构建原命令参数
    command = [
        "python3", "-m", "torch.distributed.launch",
        "--nproc_per_node=8", "--nnodes=1", "--use_env", "run.py",
        "-c", "configs/mobile/cls_emo",
        "-m", "test",
        "model.name=EMO_1M",
        "trainer.data.batch_size=2048",
        "model.model_kwargs.checkpoint_path=resources/EMO-1M/net.pth"
    ]
    
    # 调用该命令并捕获输出
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # 打印输出日志
    print(stdout.decode('utf-8'))
    if stderr:
        print(stderr.decode('utf-8'))

if __name__ == "__main__":
    main()
