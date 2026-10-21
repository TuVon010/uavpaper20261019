# Seed17主场景完成与本地归档记录

2026-10-20，主场景8组已经完成训练和完整评估，并同步到本地。四类参数扫描仍由原队列运行；本文件不表示144组已经全部完成。没有修改场景、算法、奖励、预算或选模规则。

## 协议与验收

- 训练种子17、外生场景主种子17；六个学习方法均从零训练1000个episode，每集60时隙。
- best由既定验证集episode20000–20002按既定规则选择，未用评估重新选模；last单独保留。
- 每个模型评估10个配对开发episode30000–30009。选定单场景是seed17/episode30000，并非episode0。
- 8个job.done、14份评估（两基线各一份、六学习方法best/last各一份）完整。
- 六组各1000行训练记录，主场景合并CSV共6000行；训练episode场景哈希相同。
- 14×10×60帧的时延、期限和能量已独立重算，并与逐episode CSV、summary和汇总表一致；全部评估逐episode和逐帧外生场景哈希配对通过。
- 完整归档437文件，解压937,461,144字节。压缩包SHA256：`accff397df70ec987953b7bb9e724d31bab47b97994a98f624131b99e6de360f`。
- 冻结源码仍为`3ef6b15c31a7de84f7ce1fb64ee1e70f71b252e8ee7db35e05766d1db35ca870`；本次只同步结果和维护文档。旧seed1结果及冻结实现保留。

## 主场景10个开发episode均值

J是每时隙任务平均T_i/d_i；时延是秒；能耗是一个episode三机的计算与推进总能耗。能量可行要求每架都不超过12000J，不能只用三机平均总能量判断。基线没有checkpoint选择，best与last为同一结果视图。

| 方法 | 模型 | J↓ | 平均时延/s↓ | 三机总能量/kJ↓ | 能量可行 | 按期率 | 回报↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Priority Greedy | best | 0.704693 | 0.762915 | 40.2398 | 0/10 | 78.37% | -407.794 |
| Priority Greedy | last | 0.704693 | 0.762915 | 40.2398 | 0/10 | 78.37% | -407.794 |
| Greedy + Solver | best | 0.443072 | 0.517037 | 39.7883 | 0/10 | 98.65% | -321.158 |
| Greedy + Solver | last | 0.443072 | 0.517037 | 39.7883 | 0/10 | 98.65% | -321.158 |
| Pure PPO | best | 9.119062 | 10.932633 | 31.4269 | 10/10 | 67.63% | -3342.386 |
| Pure PPO | last | 19.875134 | 21.159253 | 31.3524 | 10/10 | 67.91% | -7214.252 |
| Hybrid PPO | best | 0.414197 | 0.484855 | 33.5780 | 7/10 | 100.00% | -131.261 |
| Hybrid PPO | last | 0.414352 | 0.484917 | 33.6654 | 7/10 | 100.00% | -127.058 |
| Hybrid SAC | best | 0.460674 | 0.537769 | 41.7470 | 0/10 | 98.48% | -174.042 |
| Hybrid SAC | last | 0.460255 | 0.536452 | 41.9946 | 0/10 | 98.51% | -180.040 |
| Hybrid DDPG | best | 0.473381 | 0.551656 | 42.7686 | 0/10 | 97.82% | -183.842 |
| Hybrid DDPG | last | 0.485051 | 0.564483 | 44.2790 | 0/10 | 97.37% | -189.589 |
| Hybrid PPO adaptive AR | best | 0.408881 | 0.478539 | 33.0015 | 10/10 | 100.00% | -143.664 |
| Hybrid PPO adaptive AR | last | 0.410688 | 0.480905 | 34.1944 | 6/10 | 100.00% | -125.843 |
| Hybrid PPO AR=0.5 | best | 0.411288 | 0.481349 | 32.8416 | 9/10 | 100.00% | -134.749 |
| Hybrid PPO AR=0.5 | last | 0.411768 | 0.481928 | 32.7615 | 9/10 | 100.00% | -130.886 |

## 先前选定单场景：seed17 / episode30000

下表只列既定best或基线；它与上面的10个episode均值是不同统计口径。

| 方法 | J↓ | 平均时延/s↓ | 三机总能量/kJ↓ | 能量可行 | 回报↑ |
|---|---:|---:|---:|---|---:|
| Priority Greedy | 0.763560 | 0.822210 | 41.5229 | False | -426.615 |
| Greedy + Solver | 0.464063 | 0.536359 | 41.0555 | False | -326.609 |
| Pure PPO | 2.235041 | 2.341244 | 31.3585 | True | -863.284 |
| Hybrid PPO | 0.431635 | 0.499108 | 32.9585 | True | -165.491 |
| Hybrid SAC | 0.477086 | 0.550364 | 42.6808 | False | -166.018 |
| Hybrid DDPG | 0.489526 | 0.563263 | 43.7893 | False | -177.835 |
| Hybrid PPO adaptive AR | 0.420925 | 0.486150 | 32.2288 | True | -126.277 |
| Hybrid PPO AR=0.5 | 0.421509 | 0.486515 | 32.1649 | True | -126.453 |

## 训练过程和解释边界

以下是训练原始episode记录的窗口均值，包含探索行为和随episode变化的任务，不能直接等同于确定性评估或证明算法收敛。

| 方法 | 前100轮回报 | 后100轮回报 | 后100轮J | 后100轮罚项 |
|---|---:|---:|---:|---:|
| Pure PPO | -7433.253 | -3491.263 | 9.660713 | 593.049 |
| Hybrid PPO | -156.035 | -136.420 | 0.424900 | 8.950 |
| Hybrid SAC | -188.587 | -173.060 | 0.465313 | 33.466 |
| Hybrid DDPG | -194.184 | -189.772 | 0.486784 | 43.737 |
| Hybrid PPO adaptive AR | -164.696 | -126.472 | 0.410465 | 3.332 |
| Hybrid PPO AR=0.5 | -153.227 | -124.702 | 0.410899 | 1.433 |

自适应AR的best在这10个开发episode上J最低（0.408881），平均时延0.478539s，能量可行10/10；原混合PPO为0.414197、0.484855s、7/10。改进不是所有指标同时占优：自适应AR回报低于原混合PPO，需要同时查看请求碰撞、越界和能耗罚项。AR=0.5的best能量可行9/10，不宣称全部可行。

自适应AR best每episode平均4.2个碰撞请求帧，对应21分罚项；原混合PPO全部罚项平均7.001781分。虽然J下降减少了时延成本，额外罚项仍使自适应AR的回报低12.403367分。自适应AR last回报提高到-125.843252，但能量可行降至6/10，不能用回报替代约束可行性或用评估反向换选last。

Pure PPO的best十场景平均时延10.932633s，last更差；真实长尾和失败均已保留。此前“Pure PPO在评估seed17较好”指旧seed1训练模型在seed17外生场景上评估。本轮改变的是训练种子并从头训练，不能承诺复现旧模型在该场景的结果，也不据此重新选模型或删除长尾。

Pure PPO best/last均无UAV计算切断帧，不能把它的长尾归因于能量切断。最后25次更新均触发KL早停，价值拟合解释方差接近0，仍不能称为收敛。完整训练、罚项和逐场景分析见`docs/11_seed17_main_analysis.md`。

本轮是单训练种子的参数敏感性研究；主场景seed17来自此前开发评估后的选择，不能将它当成未触碰测试集或多训练种子统计显著性证据。本阶段仅按已授权的144任务计划完成，不追加算法优化。

## 数据和操作入口

本地项目：`E:/postgraduateLife/paper_code/uav_mec_final_1019`；服务器：`/home/zhaojunan_25/uav_mec_final`。以下路径均相对项目根。

- `results/seed17_sensitivity_20261020_01/reports/main_training.csv`：6000行主场景训练数据，每轮一行。
- `results/seed17_sensitivity_20261020_01/reports/single_selected_scene17.csv`：选定单场景逐模型数据。
- `results/seed17_sensitivity_20261020_01/reports/case_summary.csv`：best/last十episode均值；应按`case_id=main`筛选主场景。
- `results/seed17_sensitivity_20261020_01/reports/all_episodes.csv`：保留每个评估episode，查看长尾。
- `results/seed17_sensitivity_20261020_01/reports/sweep_best.csv`、`sweep_last.csv`：扫描长表，运行中仍是部分结果。
- `results/seed17_sensitivity_20261020_01/reports/REPORT.md`：自动汇总报告；本地为有时间戳的下载快照。
- `results/seed17_sensitivity_20261020_01/cases/main/jobs/<方法>/<方法>/`：episodes.csv、updates.jsonl、validation.csv、best.pt、last.pt、评估summary/逐帧轨迹/plots。基线无需训练CSV和模型。
- `evidence/seed17_main_independent_audit.json`：独立验收证据；`seed17_main_archive_manifest.json`：437文件校验和；`seed17_main_download.json`：本地下载范围。

本地主场景文件已齐全，扫描任务文件尚未完整下载；reports可能已有更晚扫描进度。以`LOCAL_DOWNLOAD_STATE.json`明确下载范围。服务器`suite.done`和`collection_audit.complete=true`且全部144组文件验收后，才算整批完成。
