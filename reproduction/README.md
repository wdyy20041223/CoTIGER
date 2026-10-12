# 云端 GenRec 代码快照

本目录保存 `/root/genrec-reproduction` 中已部署的复现脚本及准备状态。
源代码基于 `fb4bd1a1832df0312580a705d3440d1988856e31`。
快照时间和每个文件的 SHA256 记录在 `cloud-snapshot.json`。
154 个文件已在下载后逐一校验；Git 提交不包含 SSH 密码。
清单同时记录原始文件哈希和统一为 LF 后的文本哈希，支持 Windows/Linux 换行比较。

- 仓库根目录：云端实际使用的 GenRec 源码。
- `cloud_scripts/`：云端实际部署的下载、安装、训练、评估和核验脚本。
- `manifests/`：实际解析的 Gin 配置、环境状态、数据与模型哈希等记录。

相对于上游源码，当前唯一的模型训练源码修改是
`genrec/trainers/tiger_trainer.py` 中的原生 BF16 检测：
`torch.cuda.is_bf16_supported(including_emulation=False)`。
V100 使用 FP32 评估。补丁保存在 `manifests/compatibility.patch`。

这次提交保存云端原始运行配置（RQ-VAE 5000 轮、TIGER 最多 200 轮）。
用户随后要求的 2000/100 轮加速参数及工程优化，应作为后续独立提交记录。
旧训练已按用户要求停止；快照中的历史进程号和准备状态不代表任务仍在运行。

后续工作方式：在本机 Git 分支修改源码，用 RTX 4060 做短程验证，
提交并推送后由云端 Git 拉取。云端无需在本地编辑代码期间保持开机。
`cloud_scripts/` 当前保留了已部署版本的绝对路径，部署新版本时应使用
Git 管理的部署脚本，避免通过 SSH 临时改源码。

数据、模型权重、训练日志、虚拟环境和检查点不进入 Git。
这些文件继续保存在云端，并另行保留本地副本；此代码快照不替代数据备份。
不要覆盖或删除此前训练的检查点。
