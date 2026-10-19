# 输出、图表与代码位置

## 目录结构

```text
uav_mec_final/
  run.py                      统一入口
  configs/environment.json    物理/奖励/求解参数
  configs/common.json         训练及评估公共参数
  configs/experiments/         十组差异配置
  priority_mec/               环境、求解器、RL、训练、绘图
  tools/live_dashboard.py      独立实时查看入口
  tests/                      数学、物理、训练恢复、图表检查
  docs/                       操作文档和方法说明
  results/标签/               运行时生成，默认git忽略
```

结果批次目录包含 `experiment_manifest.json`，每组位于对应ID目录。不要只复制PNG而删掉清单、CSV/JSON，否则无法检查配对场景或重新绘图。

## 每轮日志

`episodes.csv` 是训练曲线的数据源，**每个完成的episode一行**，并行环境按完成顺序写入。episode编号为物理场景编号；`end_step`为全局采样transition索引，不能当本episode长度。

常用字段：episode、episode_return、frames、weighted_delay_mean、mean_delay_s、deadline_completion、per_uav_energy_j、total_energy_j、total_propulsion_energy_j、total_compute_energy_j、penalty_sum、collision_attempt_frames、boundary_attempt_frames、fully_feasible、scenario_hash、end_step。并行环境同时完成的回合会依次记录，不做跨组随机重排。

- `episode_return`：60个即时奖励之和，越大越好；包含时延与罚项。
- `weighted_delay_mean`：本回合各槽J的均值，越小越好，无量纲。
- `mean_delay_s`：平均实际完成时间，秒。
- `deadline_completion`：按期且资源有效的任务比例。
- `per_uav_energy_j`：各机整个回合累计J；`total_energy_j`是三机之和，不能直接拿总和与单机12000J比较。
- `penalty_sum`：总罚项正值，等于−episode_return−frames·weighted_delay_mean。

`fully_feasible`指电池与任务期限均满足；请求碰撞/越界另见request_safe及对应计数。执行轨迹经过安全修正，不应把“执行安全”和“原请求没有违规”混为一项。

`updates.jsonl` 每次网络更新一行，记录PPO损失、KL、ρ、有效探索尺度、学习率，或SAC/DDPG各自指标。不同算法的loss不应直接比大小。`validation.csv` 是固定验证场景随训练步数的指标；训练噪声回报与确定性验证回报应分别阅读。

`train.csv`仅在log_frames=true时生成，适合定位某个槽的动作饱和、求解证书或罚项。默认不记录大体积逐槽训练日志，但训练episode日志始终保留。

## 检查点与评估

- `last.pt`：完整恢复状态，含环境/随机流/优化器/采样缓存/episode累计量。
- `best.pt`：验证集按“整组可行优先、回报再优先、J作平局”选取的模型，仅评估用。
- `best_checkpoints/`：被选中的历史模型；保留来源便于审计。
- `effective_*.json`、`experiment.json`、`manifest.json`：实际配置、代码哈希、动作/观测维度、设备和协议。
- `status.json`：步骤、完成状态、session_fps、rollout_fps、更新/验证/绘图墙钟和错误。
- `job.done`：最终评估与绘图全部成功后才写；异常另有job.failed。

训练组 `evaluation_best/` 与 `evaluation_last/` 分别包含summary.json、episodes.csv、trajectories.jsonl；基线使用 `evaluation/`。最后模型不会被最佳模型静默替换。可选sample评估另存带 `_sample` 后缀目录。

summary.json保留每episode记录、场景及逐帧哈希、实际环境和执行方式。trajectories.jsonl包含槽首用户/UAV位置、槽末位置、请求与执行速度、每任务去向、资源、时延、能耗、安全修正和卫星信道。轨迹是实际执行位置，不是Actor请求的假想位置。

## 图表

每组 `plots/`：

- training_curves：episode原始值与MA25；奖励、J、总能量、完成率、罚项、整回合可行性。
- training_energy：三架机分别的能量及单机预算线。
- validation_curves：验证奖励/J/时延等随训练步数变化。
- trajectory_best、trajectory_last：所选开发episode的三机完整路径，起终点、用户位置/轨迹。60帧按q0…q60画61点，静止路径也标起终点。

批次 `comparison/best/`、`comparison/last/`：最终各指标、逐场景配对表、所有方法轨迹网格及训练曲线叠加。训练图只画8个学习组，基线没有虚构训练曲线。只有环境配置与逐episode/逐frame外生哈希相同，才允许配对比较。

原始episode值保留并淡画；MA25仅帮助阅读，不代表新评估数据或置信区间。单训练种子1不支持跨训练种子的统计显著性结论。开发场景的跨episode变化也不是训练稳定性置信区间。

## 核心脚本

| 文件 | 作用 |
|---|---|
| experiments.py | 读配置、十组批处理、保护结果、恢复、best/last评估及图表 |
| config.py / scenarios.py | SI参数校验、键控场景、任务与用户移动 |
| physics.py | 信道、卫星几何、推进功率、安全QCQP |
| environment.py | 槽首服务、动作解析、资源/能耗、奖励和状态保存 |
| greedy_baseline.py | 速率贪心轨迹、优先级贪心预留资源 |
| solvers.py | 共同成本势函数关联、可行修复、缓存剪枝和停止证书 |
| resources.py | CPU闭式、带宽KKT数值解、活动期限、凸证书 |
| rl/ppo.py / value_components.py | PPO、AR、延续价值、PopArt、调度 |
| rl/sac.py / deterministic.py | 原SAC、DDPG；共享实现保留内部TD3支持，不列入首批实验 |
| training.py / episode_log.py | 向量采样、每回合累计、精确恢复、选best |
| vector.py | CPU多进程环境，每帧网络批量推理 |
| reporting.py | 不依赖Torch的CSV/JSON绘图与只读本地面板 |
| io.py | 原子检查点/JSON、配置与源码核对、随机状态 |

常用断点放在environment.step、solvers.solve、resources._node_allocate、PPO.update。断点调试时使用单进程/较少环境，正式测速度再启用多进程。不要在源码内直接手改随机种子或吞掉求解失败。
