# UAV-MEC 实验定稿：seed17 主场景与参数敏感性

本定稿包含已完成的 **18个唯一场景、144组实验**：2个基线和6个学习方法，学习组各1000轮。固定训练/场景种子17；各方法在同场景的10个episode配对评估。1000轮训练记录、best/last模型、原始评估和逐帧轨迹全部保留。这次只整理、绘图和补充操作工具，没有重新训练或改变物理模型、算法、奖励。

## 先看图

直接打开 [全部图表索引](results/seed17_sensitivity_20261020_01/figures/paper/index.html)，按场景查看图表和下载对应CSV、SVG、PDF。

- [主场景训练奖励：混合方法](results/seed17_sensitivity_20261020_01/cases/main/figures/paper/training_reward_hybrid.png)
- [主场景三项指标：全部方法](results/seed17_sensitivity_20261020_01/cases/main/figures/paper/metrics_best_all.png)
- [主场景三项指标：混合方法](results/seed17_sensitivity_20261020_01/cases/main/figures/paper/metrics_best_hybrid.png)
- [主场景无人机轨迹](results/seed17_sensitivity_20261020_01/cases/main/figures/paper/trajectories_best.png)
- [用户数变化](results/seed17_sensitivity_20261020_01/figures/paper/sweep_users_best_all.png)
- [平均任务大小变化](results/seed17_sensitivity_20261020_01/figures/paper/sweep_mean_data_mbit_best_all.png)
- [单机CPU变化](results/seed17_sensitivity_20261020_01/figures/paper/sweep_uav_cpu_ghz_best_all.png)
- [单机带宽变化](results/seed17_sensitivity_20261020_01/figures/paper/sweep_uav_bandwidth_mhz_best_all.png)

每个case的比较图放在 `cases/<case>/figures/paper/`。每组实验自己的训练、逐帧指标、单机累计能耗和轨迹总览放在 `cases/<case>/jobs/<方法>/<方法>/figures/paper/overview_best.png`。全部图均有PNG、SVG、PDF和来源CSV。

## 自己选择展示算法与重画

在已安装依赖的Python环境中，项目根目录运行：

```powershell
python tools/plot_results.py
```

只改 [configs/figures.json](configs/figures.json) 的 `methods` 列表即可控制显示：将 `hybrid_ppo_ar05` 的 `enabled` 改为 `false`，它会从训练图、柱状图、扫描图、轨迹和索引中一同移除，原始数据与模型保留。`label`控制图例名，`short_label`控制柱状图横坐标，颜色、线型、标记、平滑窗口也可改。

也可不修改配置，临时只展示两种方法并另存：

```powershell
python tools/plot_results.py --methods hybrid_ppo hybrid_ppo_ar_adaptive --output-tag selected
```

只画主场景的last模型：

```powershell
python tools/plot_results.py --cases main --groups training performance trajectories per_run --checkpoint last --output-tag main_last
```

同一output-tag会刷新派生图和索引，**不会改原始实验文件**。要保留多个展示版本请换tag；索引只列本次选定的图，旧派生图文件不作为本次展示内容。详细说明见 [绘图操作手册](docs/14_plotting.md)。

## 后面自己重跑

每种方法可独立调 `configs/experiments/<method>.json`，公共设置在 `configs/common.json`，物理参数在 `configs/environment.json`。方法配置优先于公共配置；实际生效值会写入新运行的 `effective_*.json`。改进混合PPO的降噪和学习率衰减仍是独立开关。

精确复现旧配置与试新参数是两种操作，见 [重跑说明](docs/15_rerun_experiments.md)。例如先预览只重跑主场景原PPO与自适应AR：

```powershell
python tools/reproduce_suite.py --tag rerun_main01 --cases main --methods hybrid_ppo hybrid_ppo_ar_adaptive --dry-run
```

默认不启动训练；服务器上使用`--execute`才启动。全部原实现保存在 `frozen/seed17_sensitivity_20261020_01`，不要修改冻结文件或已有结果目录。

## 数据与口径

| 内容 | 位置 |
|---|---|
| 1000轮训练记录汇总（main六组，6000行） | `results/seed17_sensitivity_20261020_01/reports/main_training.csv` |
| 所有case的best/last指标 | `reports/case_summary.csv`（以下均相对本批结果根目录） |
| 所有逐episode评估 | `reports/all_episodes.csv` |
| 选定单场景（episode30000） | `reports/single_selected_scene17.csv` |
| 四类扫描数据 | `reports/sweep_best.csv`、`reports/sweep_last.csv` |
| 完成与核验 | `reports/REPORT.md`、`reports/collection_audit.json`、`suite.done` |

平均时延成本是`mean(T_i/d_i)`，无量纲；平均时延单位秒；总能量是3架UAV在60秒内推进与计算能量之和，图上用kJ。能量可行须每架机均不超过12000J，不能只看总能量是否低于36kJ。

奖励曲线每个完成的episode记录一次；浅线为原值，实线为25轮向后滑动均值，不是多训练种子置信带。柱图默认10场景均值与min–max，单场景图明确另标。best按训练时验证协议选出，不根据本次作图重选；last独立保留。单训练种子的结果不能当作跨训练种子的统计结论。Pure PPO长尾完整保留，额外提供明确标注的混合方法视图和symlog副图。

## 文档、完整性与同步

- [系统模型](docs/01_system_model.md)、[方法](docs/02_methods.md)、[物理/训练参数](docs/03_parameters.md)
- [主场景分析](docs/11_seed17_main_analysis.md)、[全套结果分析](docs/13_seed17_sensitivity_results.md)
- [本次整理说明](docs/16_final_release_guide.md)

```powershell
python tools/verify_plot_release.py
python tools/verify_plot_release.py --full-data
```

第一条检查定稿代码、文档、图表与CSV清单；第二条另读取约17.1GB原始数据核对SHA256。本地目录为 `E:/postgraduateLife/paper_code/uav_mec_final_1019`，服务器为 `/home/zhaojunan_25/uav_mec_final`。旧seed1批次已可恢复地移到两端项目外的 `uav_mec_final_archives/figures_release_20261021/`；当前`results/`只保留本批seed17全套数据。原模型所用`priority_mec`与冻结运行版本保持一致。
