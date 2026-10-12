# GenRec TIGER 云服务器复现

服务器项目目录：`/root/genrec-reproduction`。

源码来自 `https://github.com/phonism/genrec`，固定提交
`fb4bd1a1832df0312580a705d3440d1988856e31`。源代码 ZIP 保存在服务器目录中。
模型为官方 `sentence-transformers/sentence-t5-xl`，固定版本
`92e07434e0b0e93b36dd780c47c9b48d32fbcdaa`，下载后验证 SHA256。
Amazon 2014 原始数据来自 Stanford 原始发布地址；四组各使用 5-core reviews 和 metadata。

独立 Python 环境在 `.venv`，依赖来自官方 PyPI 和 PyTorch CUDA 12.4 下载站。
仓库 requirements.txt 中的固定版本保持不变。安装、数据下载、模型上传与训练使用独立后台进程。

2026-10-12：四组原始数据的八个 gzip 文件已通过完整解压 CRC 与 SHA256 校验，
文本模型已校验完成。服务器在北京时间 14:22 重启，准备进程被中断；随后使用
已从官方 PyPI 下载并逐一校验 SHA256 的 wheelhouse 完成离线安装。
`pip check`、模型导入和 CUDA 矩阵计算均通过，环境为 Python 3.10.12、
PyTorch 2.6.0+cu124、Tesla V100-SXM2-32GB。

四组实验按 Beauty、Sports、Toys、Home 顺序执行。每组包含：

1. 生成 Sentence-T5-XL 768 维物品嵌入。
2. 按 `config/tiger/amazon/rqvae.gin` 训练 RQ-VAE，完整 5000 轮。
3. 按 `config/tiger/amazon/tiger.gin` 训练 TIGER，最多 200 轮，验证集 Recall@10 早停。
4. 重新加载验证集选出的 best_model.pt，在完整测试用户集上评估。

固定随机种子为 42。默认仓库没有记录产生 README 四个单点结果的随机种子或对应模型检查点，因此同一配置的重新训练不保证逐位或逐小数完全一致。

V100 不支持原生 BF16。唯一的源码改动是检测设备是否支持 BF16；V100 评估使用 FP32。
检测调用明确使用 `torch.cuda.is_bf16_supported(including_emulation=False)`，
避免 PyTorch 将仅能创建 BF16 张量的模拟支持误认为原生硬件支持。
完整修改记录在 `manifests/compatibility.patch`。关闭 W&B 网络日志，所有训练日志保存在服务器本地。
`manifests/*-resolved.gin` 保存每组实际解析的配置；`pip-freeze.txt` 保存安装的依赖版本。

当前仓库默认使用三个语义 ID，并关闭冲突消歧。其指标按生成的语义 ID 匹配计算。
评估程序保留仓库的按批平均指标用于 README 比较，也提供按用户数加权的指标、语义 ID 冲突统计、生成候选不在实际目录中的比例。没有更改模型的训练目标或解码规则。

查看当前阶段：

```bash
cat /root/genrec-reproduction/status.json
tmux list-sessions
tail -n 20 /root/genrec-reproduction/logs/pipeline.log
```

查看具体训练（将 beauty 替换为其他数据集）：

```bash
tail -n 20 /root/genrec-reproduction/logs/beauty-rqvae.log
tail -n 20 /root/genrec-reproduction/logs/beauty-tiger.log
```

每组完成后生成 `results/<split>.json`。四组全部完成后生成
`results/comparison.csv` 和 `results/comparison.md`。
只有 `status.json` 的 `state` 为 `complete` 才表示四组训练与最终评估全部完成。
`waiting_for_setup` / `waiting_for_data` 表示正在准备，`running` 表示正在执行，
`failed` 会记录错误并保留日志和检查点。

四组均完成后，交付前运行额外的只读核验（仅写入核验报告）：

```bash
/root/genrec-reproduction/.venv/bin/python /root/genrec-reproduction/verify_results.py
```

该脚本核对固定提交的源码和允许的 V100 兼容改动、原始数据及文本模型哈希、
完整测试用户数和物品数、5000 轮 RQ-VAE 检查点及实际优化步数、原始训练参数、
验证集 Recall@10 选出的最优轮次、早停记录、最佳检查点哈希，以及最终测试指标
与该轮训练日志、README、comparison.csv 和 comparison.md 的一致性。
成功后生成 `results/verification.json`；如果核验失败，先诊断，不能宣布复现完成。

后台会话名为 `genrec-tiger`，SSH 断开不影响服务器端流水线运行。
可用只读快照脚本同时查看实际进程、准备状态、当前日志末尾和 GPU：

```bash
python3 /root/genrec-reproduction/check_reproduction.py
```

本地 SSH 桥接原会话 76992 已在服务器重启后失效，2026-10-12 重连的新会话为 47961。
若该会话也失效，请使用 `ssh_bridge.py` 重新连接；密码仅从聊天凭据输入，不写入文件。
14:48 的实测快照确认：流水线 PID 2519、Beauty RQ-VAE 主进程 PID 2521 均存活，
训练已超过第 400/5000 轮，首次编译后损失有限，保存与评估正常执行。
这些 PID 和进度仅为当时的快照；后续继续工作前应重新核对实际进程及日志。

2026-10-12 后续检查确认 Beauty 的 RQ-VAE 已结束，最终检查点 `checkpoint_epoch_4999.pt`
记录 `epoch=4999`，日志有第 5000 轮评估；TIGER 主进程 PID 3814 继续运行。
检查点的优化器和学习率调度器均记录 59990 次更新，进度条预估值为 60000。
上游在首步 KMeans 初始化时嵌套迭代同一个训练 DataLoader，Beauty 首轮日志进度仅 0–2。
此处保留上游完整 5000 轮流程，没有补步或更改训练算法；实际步数差异记录在
`manifests/beauty-rqvae-audit.json`，应在最终复现报告中注明。

如果需要重新启动已经退出的流水线，可执行：

```bash
tmux new-session -d -s genrec-tiger 'python3 -u /root/genrec-reproduction/pipeline.py >> /root/genrec-reproduction/logs/pipeline.log 2>&1'
```

该命令会跳过已经写出完整评估结果的组；已有最终 RQ-VAE 检查点会复用。
TIGER 当前上游训练器没有保存优化器恢复状态，未完成的 TIGER 阶段重新启动会从头训练。
