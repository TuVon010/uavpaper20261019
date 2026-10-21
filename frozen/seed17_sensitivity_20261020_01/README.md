# UAV–LEO 任务卸载：Seed 17 主场景与参数敏感性工程

当前阶段：以训练/场景主种子17从零运行8组主场景实验，并扫描用户数、平均任务大小、单UAV算力和带宽。主物理参数保持原值，仅各扫描点改变指定因素；18个唯一配置共144组运行。每个训练episode保存一行原始数据，自动重绘每100个episode一次。配置、命令、记录字段、恢复策略与结果路径见[Seed 17实验说明](docs/08_seed17_sensitivity.md)；实际进度以对应suite状态和评估文件为准。

下面原seed1训练与20种子评估是已完成的历史参考，完整保留。新功能支持配置化用户数；旧模型仍须使用其原冻结实现加载，不放宽版本检查。

用户本地项目：`E:/postgraduateLife/paper_code/uav_mec_final_1019`；服务器项目：`/home/zhaojunan_25/uav_mec_final`。以本目录配置和运行时保存的 `effective_*.json` 为准。用户自行整理的 `docs/系统模型/`、`docs/主算法框架/` 目录保留，本文档描述的是代码实际执行的版本。

## 当前保留内容

- 8组方法：两个贪心基线、纯PPO、混合PPO/SAC/DDPG、自适应AR混合PPO、AR=0.5混合PPO。
- 已完成批次 `results/wide_compute1000_w5_cutoff_20261020_01/`：训练日志、检查点、原best/last各10场景评估、曲线与轨迹，以及新增加的20种子评估。
- 对应的 `frozen/wide_compute1000_w5_cutoff_20261020_01/`：严格加载已有检查点所需的原运行实现。不要编辑冻结目录。
- 参数、运行、结果与复现说明；旧批次和不再使用的入口移出当前工作目录归档，清单见根目录定稿记录。

共享 `priority_mec/` 源码增加了可变用户数与参数扫描支持；已有seed1检查点对应的原实现保留在上述冻结目录。当前8组方法和求解器逻辑不因扫描而更换，检查点严格校验继续保留。

## 先看结果

20种子评估固定使用原best检查点：场景种子1–20，每个种子只运行episode 30000一次。包含训练主种子1，但没有重用训练episode。这是固定模型对20个场景的表现，不是20次独立训练。

| 方法 | 加权时延成本↓ | 按期率 | 总能量/kJ | 每机能量均可行 | 回报↑ |
|---|---:|---:|---:|---:|---:|
| Priority Greedy | 0.7031 | 77.80% | 40.523 | 0/20 | −395.72 |
| Greedy＋Solver | 0.4474 | 98.50% | 40.078 | 0/20 | −311.24 |
| Pure PPO | 7.8642 | 68.23% | 31.405 | 20/20 | −2910.21 |
| Hybrid PPO | 0.4135 | 100% | 33.457 | 19/20 | −145.81 |
| Hybrid SAC | 0.4594 | 98.56% | 41.495 | 0/20 | −185.36 |
| Hybrid DDPG | 0.4626 | 98.37% | 41.191 | 0/20 | −189.45 |
| Hybrid PPO AR-adaptive | 0.4099 | 100% | 33.957 | 19/20 | −130.73 |
| Hybrid PPO AR=0.5 | 0.4113 | 100% | 32.405 | 20/20 | −124.90 |

详细结果与160条逐种子记录：

- [20种子完整报告](results/wide_compute1000_w5_cutoff_20261020_01/evaluation_seed20_20261020_01/REPORT.md)
- [逐种子指标CSV](results/wide_compute1000_w5_cutoff_20261020_01/evaluation_seed20_20261020_01/per_seed_metrics.csv)
- [结果解释与局限](docs/05_results_and_analysis.md)

## 常用操作

在本项目目录执行。先安装 `requirements.txt` 中的依赖；服务器已有独立 `.venv`，不随整理删除。

```powershell
python run.py list
python run.py dashboard --run results/wide_compute1000_w5_cutoff_20261020_01 --port 8765
python run.py plot --run results/wide_compute1000_w5_cutoff_20261020_01/hybrid_ppo_ar05
python run.py trajectories --run results/wide_compute1000_w5_cutoff_20261020_01/hybrid_ppo_ar05 --checkpoint best
python run.py compare --batch results/wide_compute1000_w5_cutoff_20261020_01 --checkpoint best
```

新实验使用新tag，不覆盖上面已完成批次：

```powershell
python run.py run --experiment hybrid_ppo_ar05 --tag trial01 --smoke --device cpu
python run.py run --experiment hybrid_ppo_ar05 --tag trial02 --device auto
```

`--smoke` 会缩短episode和训练，只检查流程，不能当正式实验。`episodes.csv` 每完成一个episode记录一行；训练曲线、验证曲线和最终轨迹均可重新生成。

## 文档索引

1. [系统、时序、信道、能耗与奖励](docs/01_system_model.md)
2. [8组方法、关联与KKT、AR与延续价值](docs/02_methods.md)
3. [默认参数及每组独立调参](docs/03_parameters.md)
4. [训练、恢复、评估、绘图与服务器操作](docs/04_operations.md)
5. [原10场景与新20种子结果、解释和局限](docs/05_results_and_analysis.md)
6. [数据结构、冻结源码、配对场景与复现检查](docs/06_reproducibility.md)

7. [定稿清单、归档位置与完整性检查](docs/07_release_inventory.md)

8. [Seed 17主场景、四类单因素扫描与新结果操作](docs/08_seed17_sensitivity.md)
