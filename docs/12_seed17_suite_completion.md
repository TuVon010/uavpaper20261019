# Seed17主场景与参数敏感性实验：完整归档

归档核对时间：2026-10-21T07:28:30.463547+00:00。批次`seed17_sensitivity_20261020_01`的144组实验已经完成训练、best/last评估、完整下载和两端独立验收。没有追加算法优化、修改物理模型、改变预算或利用评估重新选模。旧seed1结果和冻结实现保留。

## 完成范围

| 项目 | 验收数量 |
|---|---:|
| 唯一物理配置 | 18 |
| 方法/配置运行 | 144 |
| 从零训练运行 | 108 |
| 规则基线运行 | 36 |
| 逐轮训练记录 | 108000 |
| 评估协议（两基线及学习方法best/last） | 252 |
| 逐评估episode | 2520 |
| 逐评估帧 | 151200 |
| 主场景合并训练CSV行数 | 6000 |
| case结果文件 | 7327 |
| 套件元数据与报告文件（含generated和锁文件） | 41 |
| 以上合计（不含本地下载说明） | 7368 |

原始结果共17,110,157,475字节，约17.1 GB。剩余参数扫描的分包压缩传输约6.87 GB；主场景原有下载经最终文件校验复用。大文件直传中断后采用受控分块续传，最终按压缩包SHA256及逐文件SHA256校验，未重跑实验。压缩包和原始结果均保留。

本地：`E:/postgraduateLife/paper_code/uav_mec_final_1019`。服务器：`/home/zhaojunan_25/uav_mec_final`。两端源代码和冻结实现的73个管理文件仍逐项一致，源码摘要`3ef6b15c31a7de84f7ce1fb64ee1e70f71b252e8ee7db35e05766d1db35ca870`。未修改受管理的`CURRENT_SUITE.json`或README；最新归档状态另见根目录`FINAL_SUITE_ARCHIVE.json`。

## 实验口径

所有训练与场景主种子为17。不同episode/时隙的外生任务仍变化；同配置的各方法配对相同外生场景。每个学习方法、每套唯一配置从零训练1000轮，每轮60帧。用户数20、CPU10GHz、带宽10MHz三点复用主场景，没有重复计算为独立证据。

四轴为用户数10/15/20/25/30，数据均值0.5/1/1.5/2/2.5 Mbit，单UAV CPU5/10/15/20/25 GHz，单UAV带宽5/10/15/20/25 MHz。均值数据采用`U[0.4μ,1.6μ] Mbit`，主场景仍是`U[0.5,2] Mbit`；每套配置只改变一个因素。

best沿用验证episode20000–20002的既定选择，last单独保留。评估是十个配对开发episode30000–30009；选定单场景为seed17/episode30000。它不是十个训练种子，也不是未使用测试集，不据此给多训练种子显著性结论。

J是平均`T_i/d_i`，平均时延单位秒，能量为一个60秒episode中三机推进与计算总能量。每架都须不超过12000J才算能量可行；三机合计小于36000J并不足以保证。计算关闭后仍累计飞行能量，关闭服务本身不是电池硬约束保证。

## 结果要点

两种AR的best均在17/18套配置降低J，但回报和能量可行性不是全面改善。best在全部十个开发场景能量可行的配置数：原混合PPO13/18，自适应AR12/18，固定AR=0.5为14/18；last相应7/18、3/18、11/18。不能据评估回报更高反向换选last。

两个贪心基线、SAC和DDPG在本轮各配置的best均未达到十场景中的能量可行episode；所有超预算记录保留。Pure PPO虽评估能量可行，却有明显时延长尾，10用户时出现单episode平均时延300.705秒；能量低不代表服务质量好，不能宣称其已经收敛。

主场景自适应AR best的J=0.408881、平均时延0.478539秒、总能量33.0015kJ、能量可行10/10。但其请求碰撞罚项使回报低于原PPO。完整best/last、单场景及四轴逐点数据见`docs/13_seed17_sensitivity_results.md`。

## 文件与画图入口

以下均相对项目根：

- `results/seed17_sensitivity_20261020_01/reports/main_training.csv`：六组主场景各1000轮的合并训练记录，便于画奖励/时延/能耗随轮次曲线。
- `results/seed17_sensitivity_20261020_01/reports/single_selected_scene17.csv`：252行所选单场景数据（包含best/last，基线只一份）。
- `results/seed17_sensitivity_20261020_01/reports/case_summary.csv`：288行十episode均值；按case_id、method、checkpoint筛选。
- `results/seed17_sensitivity_20261020_01/reports/all_episodes.csv`：2520行完整评估数据，保留长尾。
- `results/seed17_sensitivity_20261020_01/reports/sweep_best.csv`、`sweep_last.csv`：四轴各5点×8方法，每表160行；主要三指标为weighted_delay_mean、mean_delay_s、total_energy_j。
- `results/seed17_sensitivity_20261020_01/reports/REPORT.md`：自动汇总报告，`collection_audit.json`为调度器验收。
- `results/seed17_sensitivity_20261020_01/cases/<case>/jobs/<method>/<method>/`：原始episodes.csv、updates.jsonl、validation.csv、best.pt、last.pt、best_checkpoints及评估轨迹/图像。基线不含训练模型。
- `evidence/seed17_final_archive_manifest.json`：全部原始文件与分包SHA256；`seed17_final_case_transfers/`为18套配置传输验收。
- `evidence/seed17_full_independent_audit.json`、`seed17_server_full_audit.json`：两端独立原始数据验收。`seed17_full_analysis.json`保留统计核对及精确数值。
- `docs/08_seed17_sensitivity.md`：参数配置与命令；本文件记录最终归档状态。

各训练方法仍每episode记录一行；每100轮刷新图像不减少CSV采样。已有曲线与轨迹图均随结果保存，没有对原始数据平滑、删尾或覆盖模型。

## 独立核验与限制

两端独立从已保存的逐任务时延与deadline重算J、平均时延和QoS，并从association、data和实际速度重算推进/计算能耗及奖励；未重新求解香农速率或资源分配。同case所有方法及best/last的已保存episode/frame哈希和原始外生数据摘要一致。CPU/带宽轴与main的评估逐帧外生输入及训练完整场景哈希相同。原始报表与重算结果一致；没有重新生成信道随机流。

训练设置log_frames=false，因此训练只保存完整场景哈希；没有声称检查未保存的训练原始逐帧轨迹。模型核对原选模信息、存在性与原文件SHA，没有通过额外评估重新选择检查点。

可在项目根只读重验：

```bash
python evidence/audit_seed17_full.py --root results/seed17_sensitivity_20261020_01 --out evidence/seed17_recheck.json
```

本批归档结束后暂停自动跟进，不自动开启新训练或下一阶段优化。
