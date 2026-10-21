# 定稿清单与同步检查

本次版本标识为 `final_20261020_cutoff_seed20`，工程版本1.1.0。直接整理原项目 `uav_mec_final_1019`，服务器仍为 `/home/zhaojunan_25/uav_mec_final`，不另建一套活动项目。

## 保留的实验

只保留当前批次 `results/wide_compute1000_w5_cutoff_20261020_01/`。它包含8组实验：2个规则基线及6个学习方法；六组分别保留1000个完整训练episode、20次验证、网络更新日志、best/last及历史best检查点、原10场景的best/last评估与逐帧轨迹、曲线和两套方法对照图。

`evaluation_seed20_20261020_01/` 包含8组×20场景种子的160条评估记录，以及原始JSON、CSV、中文报告、完整场景配对审计和当时使用的评估脚本。每个方法均使用既有best，不根据新增评估重选。原始数据共623个文件、950,122,777字节，文件大小和SHA256见 `evidence/retained_results_manifest.json`。

保留原始失败与恢复记录：20种子评估全部一次完成，但第一次汇总时报告脚本尚未就位，汇总退出。随后仅重新生成报告，通过配对审计后写入最终完成状态；没有重跑或剔除评估。`failure.json`、原driver退出码、`status_before_report_recovery.json`和最终`job.done`用于完整追踪这一过程。

## 整理范围

- `priority_mec/`及对应的当前冻结运行实现保持原源码，不重训、不改变算法或物理模型，也不放宽模型源码校验。
- 活动实验配置仅保留当前8组；默认 `configs/environment.json` 与本轮宽场景停服配置相同。原先的旧默认参数、旧批次及不使用的纯SAC/DDPG/纯改进PPO入口移出归档。
- 保留全部共享内核和数学/物理/恢复测试。包内仍存在共用扩展分支，因为已有模型对完整源码进行严格校验；它们没有出现在本版实验入口中，也没有启用。
- 保留 `docs/系统模型/`、`docs/主算法框架/` 的用户原稿材料。
- 旧基线专用工具、旧启动说明、旧冻结目录和临时文件移出活动项目。`.git`及服务器`.venv`保留。

整理前备份和移出内容放在项目外：本地 `E:/postgraduateLife/paper_code/uav_mec_final_archives/final_20261020/`，服务器 `/home/zhaojunan_25/uav_mec_final_archives/final_20261020/`。具体移出文件清单记录在归档目录内。活动项目内的同步核对记录为 `evidence/sync_verification.json`。

## 校验定稿

在项目目录运行：

```powershell
python tools/verify_release.py --full-data --load-models
```

这会核对管理文件与全部保留结果的哈希，并在CPU加载六个best模型，不执行训练。`RELEASE_MANIFEST.json`记录本版代码、配置、文档和工具的文件摘要；`CURRENT_RELEASE.json`记录实验来源与结果指针。调参后出现配置摘要差异是预期现象，此工具用于核对原定稿状态，不能拿它把有意的参数修改恢复掉。

本地整理验证已通过263项测试、跳过5项，并成功加载六个best模型；复制数据上重新生成训练/验证曲线、能耗曲线与best轨迹，原结果未改写。服务器检查与两端最终同步以 `evidence/sync_verification.json` 为准。原训练的回归和短程恢复证据仍在 `SERVER_SERVICE_CUTOFF_VALIDATION.json`，它与本次目录整理检查是两份不同记录。
