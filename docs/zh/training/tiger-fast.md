# TIGER：V100 加速配置与极小流程验证

当前 `config/tiger/amazon/{rqvae,tiger}.gin` 使用以下加速配置。它缩短训练并减少搜索和评估，是否能将 Recall/NDCG 的相对下降控制在 3% 内，需要完整数据的对照实验；本地流程检查不证明推荐精度。

| 阶段 | 参数 | 原值 | 当前值 |
| --- | --- | ---: | ---: |
| RQ-VAE | epochs | 5000 | 2000 |
| RQ-VAE | save_model_every | 50 | 200 |
| RQ-VAE | eval_every | 50 | 200 |
| TIGER | epochs | 200 | 100 |
| TIGER | beam_size | 30 | 15 |
| TIGER | eval_test_every_epoch | 2 | 5 |
| TIGER | eval_valid_every_n_epochs | 1 | 3 |
| TIGER | infer_batch_size | 96 | 256 |
| TIGER | early_stop_patience | 10 | 10 |

RQ-VAE 保留原来的 50 轮 warmup、学习率、batch、网络、码本和量化方式。第 2000 轮必定保存 `out/tiger/amazon/beauty/rqvae/checkpoint_epoch_1999.pt`，TIGER 配置中的输入路径已同步。

## 执行优化

- 两阶段启用 FP16 autocast 和动态 loss scaling。RQ-VAE 使用 Accelerate 的梯度裁剪接口，先解除梯度缩放再裁剪。TIGER 的评估精度跟随配置，去除原来的硬编码 BF16；V100 使用 FP16。
- RQ-VAE 编码器、解码器矩阵运算使用 AMP，量化距离保留 FP32，Sinkhorn 运输矩阵保留 FP64。修复 FP64 代价矩阵和 FP32 边缘分布的 dtype 混用。
- RQ-VAE 的日志去重统计用 `torch.unique` 替代 B×B 两两比较，并且只在需要日志时计算。K-means 初始化直接使用编码与量化路径，避免重建解码和日志统计。
- 去除模型中无条件执行的 `torch.compile`。需要编译时可显式设置 `train.compile_forward=True`，默认不启用，避免 Windows/Triton 和首次编译开销。
- 缓存的商品向量直接组成 tensor DataLoader，避免每个 item 的 ndarray→list→tensor 转换。RQ-VAE 默认无 worker；TIGER 的 worker 持续复用，collate 使用可序列化的 `partial`。
- TIGER 按 batch 动态左 padding。真实历史仍截断到 50 个商品；最多 150 个真实 token 可补齐到 152 个位置，以满足 8 的对齐。
- T5 可使用 PyTorch SDPA，保留 T5 的 `scale=1.0`、相对位置偏置、因果和 padding mask。偏置转换前限制负值范围，匹配 Q/K/V dtype 并使用连续布局，使融合内核可用。后端由 PyTorch 按硬件、形状和 mask 选择，不能保证每个 batch 都使用融合内核。
- beam search 首步在 eval 模式下只解码每个用户的一份相同起始 token，再扩展 KV cache。训练模式下保留各 beam 独立计算。beam 分数以 FP32 累计。
- TIGER 的 train/valid/test 共享一次生成的语义 ID 和用户历史，避免三次加载、量化及解析；不同 split 的样本仍独立生成。
- 使用非阻塞数据传输和 `zero_grad(set_to_none=True)`；loss 和指标在 GPU 累计，减少逐 batch 的 CPU 同步。

AMP 的缩放、解除缩放与裁剪顺序依据 [PyTorch 2.6 AMP 文档](https://docs.pytorch.org/docs/2.6/notes/amp_examples.html)。SDPA 的实现依据 [PyTorch 2.6 SDPA 文档](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html)。

## 评估与 checkpoint

验证在第 1 轮、每 3 轮及最后一轮执行。patience=10 仍指连续 10 次验证无提升，在新配置下大约对应 30 个训练 epoch。

测试独立每 5 轮执行，不再随每次验证提升额外触发，也不用于选模或早停。训练结束后，加载本次验证 Recall@10 最佳的 `best_model.pt`，额外执行一次最终测试并保存 `results.json`。即使第一次验证 Recall 为 0，也会保存本次 checkpoint，不会误用目录中的旧权重。

指标按用户数累计，不再平均每个 batch 的均值，因此最后一个较小的 batch 不会被过度加权。这是统计修正，旧版指标可能有轻微差异，不能将差异全部归因于加速配置。

## 云端运行

在仓库根目录、已激活的 `genrec` 环境执行。模型路径按服务器实际位置覆盖；以下假定模型在仓库的 `models_hub/sentence-t5-xl`。不要用两个并发进程跑这两个有依赖关系的阶段。

```bash
python -m genrec.trainers.rqvae_trainer config/tiger/amazon/rqvae.gin --split beauty \
  --gin "MODEL_HUB_SENTENCE_T5_XL='./models_hub/sentence-t5-xl'" \
  --gin "train.wandb_logging=False"

python -m genrec.trainers.tiger_trainer config/tiger/amazon/tiger.gin --split beauty \
  --gin "MODEL_HUB_SENTENCE_T5_XL='./models_hub/sentence-t5-xl'" \
  --gin "train.wandb_logging=False"
```

若要保留不同实验的结果，可同时覆盖 RQ-VAE 的 `train.save_dir_root`、TIGER 的 `train.pretrained_rqvae_path` 和 `train.save_dir_root`，保持输入输出路径配套。

回到 FP32：给对应阶段添加 `--gin "train.amp=False"`。关闭 SDPA 或动态 padding：给 TIGER 添加 `--gin "train.use_sdpa=False"` 或 `--gin "train.dynamic_padding=False"`。`--gin "train.first_beam_only=False"` 可用于首步 beam 复用的参考对照。

## 本地极小检查

```bash
python -m unittest discover -s tests -p test_tiger_speed.py -v
python scripts/tiger_smoke.py
```

smoke 脚本的边界固定：只读取原始 Beauty 文件前 8192 行作为来源；从中选 8 个真实用户、每人 7 条交互，复用 320 个已有 Sentence-T5-XL 向量。全部数据及输出放在新的 `out/tiger_speed_smoke/run_*` 子目录。

训练保持生产网络结构、768 维输入和 256 大小码本：RQ-VAE 2 个 optimizer step，TIGER 2 个 optimizer step，验证/测试各 8 个样本。只为此检查跳过大样本 K-means 初始化和完整文本编码；K-means 初始化另用 32×8 的小输入作单元检查。另有 batch=2、3 次测量的极小前向/反向和 beam=15 生成计时，可用 `--skip-timing` 跳过。

检查涵盖 FP32 下关闭 dropout 时 SDPA 的输出、loss、梯度和 KV cache 解码一致性，动态 padding，首步 beam 复用，语义 ID 共享，去重统计，以及不完整评估 batch 的正确计权。GPU smoke 检查实际 optimizer 更新、有限权重、RQ-VAE checkpoint/GradScaler 保存、TIGER 最佳 checkpoint 加载和最终测试。

极小检查中的 Recall/NDCG 不代表推荐质量，2 条历史的时延也不代表 V100 上 batch=256 的吞吐。完整准确率和实际总训练时间由云端实验确定。
