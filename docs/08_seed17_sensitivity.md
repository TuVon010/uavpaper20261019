# Seed 17 主场景与参数敏感性实验

本阶段先在原物理主参数下，以训练种子和场景主种子 **17** 从零运行全部8种方法，保留每个训练episode的数据；随后进行4类单因素参数扫描。本文说明预先确定的协议、配置和操作，**不表示实验已经完成**。进度、失败和完成情况以对应结果目录的状态、日志、`job.done`、评估summary及实际进程为准。

这4类扫描属于**参数敏感性实验**，不是移除AR、延续价值、降噪或学习率衰减模块的算法消融。若以后研究模块贡献，应另设同场景、同训练预算的模块开关实验。

## 1. 主场景和保持不变的设置

| 类别 | 主场景值 | 本次扫描是否改变 |
|---|---|---|
| 场景/训练主种子 | 17 / 17 | 全部固定 |
| 用户数 | 20 | 仅用户数扫描改变 |
| 初始用户区域 | 地图中心(500,500)周围500×500m正方形 | 不变 |
| UAV / 卫星数量 | 3 / 1 | 不变 |
| 区域、时隙、episode | 1000×1000m；1s；60时隙 | 不变 |
| UAV高度、最大速度、安全距离 | 50m；20m/s；50m | 不变 |
| 本地 / 单UAV / 卫星算力 | 1 / 10 / 50GHz | 仅单UAV算力扫描改变中间值 |
| 单UAV / 卫星带宽 | 10 / 100MHz | 仅单UAV带宽扫描改变前者 |
| 主场景任务大小 | U[0.5,2]Mbit，均值1.25Mbit | 仅任务大小扫描改变 |
| 任务计算密度 | 1000cycles/bit | 不变 |
| 期限、权重 | U[0.5,2]s；wᵢ=1/dᵢ | 不变 |
| 每架UAV能量预算 | 12000J，计算＋含悬停的推进能量 | 不变 |
| 能量触发停服 | 槽首累计能量达到阈值后停止接收计算任务，飞行继续 | 不变 |
| 服务时序、信道、移动、能量模型 | 沿用[系统模型](01_system_model.md) | 不变 |
| 时延奖励权重 | 5 | 不变 |

原主场景任务大小仍为U[0.5,2]Mbit；**不会为了对齐扫描刻度，把主场景均值改成1或1.5Mbit**。所有配置中容量使用Hz、带宽使用Hz、数据使用bit，表格中的GHz、MHz、Mbit仅用于阅读。

用户数可从配置调整。令N为用户数，基础观测维度为18N+16；纯PPO动作维度为6+7N，混合方法仍输出6个UAV速度分量。每个配置重新构造对应网络，不能把20用户的纯PPO检查点直接用于10或30用户。

## 2. 四类单因素扫描

每个扫描点只改变本表列出的变量，其余使用第1节主参数。

| 扫描 | 档位 | 修改的环境配置字段 |
|---|---|---|
| 用户数N | 10、15、20、25、30 | `num_users` |
| 平均任务大小μ | 0.5、1、1.5、2、2.5Mbit | `data_min_bit`、`data_max_bit` |
| 单UAV最大CPU | 5、10、15、20、25GHz | `uav_cpu_hz` |
| 单UAV带宽 | 5、10、15、20、25MHz | `uav_bandwidth_hz` |

用户已确认任务大小扫描采用固定相对宽度 **D~U[0.4μ,1.6μ]**，不是把每个任务都设置成μ。

| μ / Mbit | 数据范围 / Mbit |
|---:|---:|
| 0.5 | [0.2,0.8] |
| 1.0 | [0.4,1.6] |
| 1.5 | [0.6,2.4] |
| 2.0 | [0.8,3.2] |
| 2.5 | [1.0,4.0] |

主场景μ=1.25Mbit也满足该相对宽度。它不在上述5个指定均值刻度中，如绘图额外展示主场景参考点，必须标明“主场景参考”，不能把它误计为指定扫描档位。

用户数20、CPU10GHz、带宽10MHz这3个档位均对应同一原主配置，运行时共享主场景结果以避免重复训练。总计**18个唯一环境配置×8种方法＝144个方法–配置运行**，包括108次学习方法训练和36次基线评估。108次训练均为1000个60时隙episode，合计648万环境训练步；每次运行独立保存数据。

## 3. 每个配置运行的8种方法

| ID | 方法 | 是否重新训练 |
|---|---|---|
| `baseline_priority_greedy` | 低分辨率贪心轨迹＋独占时延选节点后一次均分资源 | 否，直接运行同场景 |
| `baseline_greedy_solver` | 相同贪心轨迹＋当前关联/KKT求解器 | 否，直接运行同场景 |
| `pure_ppo` | 纯PPO联合决定速度、卸载和资源偏好，不调用求解器 | 是 |
| `hybrid_ppo` | 原始PPO速度决策＋关联/KKT求解器 | 是 |
| `hybrid_sac` | 原始SAC速度决策＋关联/KKT求解器 | 是 |
| `hybrid_ddpg` | 原始DDPG速度决策＋关联/KKT求解器 | 是 |
| `hybrid_ppo_ar_adaptive` | 自适应AR＋延续价值PPO＋求解器 | 是 |
| `hybrid_ppo_ar05` | AR=0.5＋延续价值PPO＋求解器 | 是 |

所有学习组**每个唯一配置都从零训练**，不把上一档模型续训到下一档，也不复用旧seed1检查点。两个改进混合PPO仍按各自recipe启用降噪和学习率衰减，可另建suite或独立运行参数副本关闭；本轮正式比较不在运行中修改recipe。求解器目标、求解精度和期限处理保持一致，不为吞吐量降低优化精度。

训练参数由公共配置和各方法独立recipe合并：1000个episode、8个并行环境、hidden128；PPO每480步更新，actor/critic学习率1e−4/3e−4，γ=1、GAE λ=0.99，其他参数以保存的`effective_*.json`为准。SAC/DDPG使用各自现有recipe，不通过改算法配置伪装成同一算法。

## 4. 训练、选模与评估协议

1. 先完整完成主场景的8组训练/评估，再调度其他17个唯一配置。
2. 所有学习组训练种子17、场景主种子17，训练episode编号0–999。不同episode以及同一episode的不同时隙仍产生不同任务；固定种子不表示重复同一批任务。
3. 每3000训练步验证一次，使用场景主种子17、episode20000–20002。按原`feasible_return`协议选择best，保持原选模规则，不能用后续比较结果重新挑选检查点。
4. 每组best和last分别评估场景主种子17、episode30000–30009，共10个配对episode。基线使用相同10个episode；它没有best/last模型差别，汇总时复用同一基线评估并明确其身份。
5. 基于actor确定性条件均值执行，沿用原路由公共随机流。纯PPO的路由仍由同场景已记录的分类采样随机数产生，不能把确定性actor均值误写成“每个任务始终确定选择argmax节点”。
6. 预留episode50000起的最终测试不使用。全部结果仍属于用户指定主场景下的开发比较。

同一配置、同一episode/frame的任务、移动和信道随机输入在各方法间一致，并通过场景和逐帧哈希核对。CPU/带宽扫描仅改资源容量，外生场景哈希应不变；用户数扫描共享前min(N₁,N₂)用户的外生随机前缀；数据均值扫描共享随机分位，但数据值和场景哈希会随范围改变。跨配置不能要求整个场景哈希一律相同。

**种子17是用户在看过旧seed1模型的20场景结果后指定的主场景。** 它不是预先未见过的独立最终测试。本轮固定一个训练种子，不能据此声称方法在多训练种子下稳定，也不能只保留最有利的档位或方法。

## 5. 主要记录指标与训练曲线

| 指标 | 字段 | 单位与含义 |
|---|---|---|
| 平均加权时延成本 | `weighted_delay_mean` | 无量纲；每槽J=(1/N)ΣᵢTᵢ/dᵢ，再对60槽和评估episode取平均 |
| 平均任务时延 | `mean_delay_s` | 秒；实际Tᵢ的算术平均 |
| UAV总能量 | `total_energy_j` | J；3架UAV的推进与计算能量总和；绘图可除1000显示kJ |

同时保留按期率、每架UAV能量、能量可行性、计算停服比例、违规罚项、回报和决策时间，用于解释主指标差异。每机能量是否≤12000J分别判断，不能用三机总和与12000J比较。

用户数变化时使用1/N归一化，避免仅因任务总数增加就机械放大平均时延成本。原始加权时延和另存为`weighted_delay_sum_per_frame=N×J`，不要把它和平均成本混用。有限但极大的真实时延不截断；因此纯PPO均值可能受少量坏链路任务影响，应连同按期率和逐episode值阅读。

每完成一个训练episode，`episodes.csv`写入一行，包括回报、J、平均时延、总/单机能量与可行性等；保留原始1000行供后续自行作图。`render_every_episodes=100`只降低自动重绘频率，不降低数据记录频率。`validation.csv`保存每次验证，`updates.jsonl`保存学习更新诊断；逐帧`train.csv`仅在`log_frames=true`时写入，本轮默认关闭以减少I/O。`best.pt`和`last.pt`分别对应选定与最终模型，评估保留逐episode指标和完整轨迹。基线无训练过程，不伪造训练曲线。

训练后可按需重绘：

```powershell
python run.py plot --run results/seed17_sensitivity_20261020_01/cases/main/jobs/hybrid_ppo_ar05/hybrid_ppo_ar05
python run.py trajectories --run results/seed17_sensitivity_20261020_01/cases/main/jobs/hybrid_ppo_ar05/hybrid_ppo_ar05 --checkpoint best
```

若生成计划中的主场景case ID不同，以`suite_plan.json`中的`case_id`和输出路径为准，不能把文档示例当成已有结果。

## 6. 配置、生成与运行

在项目根目录执行。套件定义为`configs/suites/seed17_sensitivity.json`，方法参数仍分别位于`configs/experiments/`。先改套件配置，再使用新tag生成计划；生成后不在原计划中途改变物理参数或方法recipe。

```powershell
python tools/sensitivity_suite.py generate --project . --suite configs/suites/seed17_sensitivity.json --tag seed17_sensitivity_20261020_01
python tools/sensitivity_suite.py run --project . --runtime frozen/seed17_sensitivity_20261020_01 --root results/seed17_sensitivity_20261020_01 --parallel 4 --device cuda --plan-only
python tools/sensitivity_suite.py run --project . --runtime frozen/seed17_sensitivity_20261020_01 --root results/seed17_sensitivity_20261020_01 --parallel 4 --device cuda
python tools/sensitivity_suite.py collect --root results/seed17_sensitivity_20261020_01
```

`generate`输出`results/<tag>/suite_plan.json`、`generated/environment/<case>.json`和`generated/common.json`。冻结运行目录需先由部署流程创建并校验；不要在冻结目录手动调参。`--plan-only`只展示待执行命令，不启动训练。

服务器复用项目自有`.venv`，进入`/home/zhaojunan_25/uav_mec_final`后以`.venv/bin/python`替代`python`。长期任务通过独立tmux会话运行，断开SSH或关闭本地电脑后继续执行。并行度只是调度上限，不改变单组训练预算；每组8个环境进程，4个学习组同时运行时最多约32个环境工作进程，另有主进程和评估开销。

网络推理和更新使用GPU；物理环境、信道、关联与凸资源求解主要使用CPU。GPU利用率低不一定是未启用GPU，可能是CPU求解瓶颈。默认限制BLAS/torch CPU线程，避免每个工作进程再启动大量线程。以日志中的环境采样速度、更新耗时、含验证总FPS和GPU/CPU占用判断瓶颈；不得把历史累计步数除以新恢复进程时间计算FPS。

## 7. 中断、续调度与失败保留

正常续调度使用同一计划：

```powershell
python tools/sensitivity_suite.py run --project . --runtime frozen/seed17_sensitivity_20261020_01 --root results/seed17_sensitivity_20261020_01 --parallel 4 --device cuda --continue
```

`--continue`用于继续套件调度：仅跳过经过严格检查的完整任务，拒绝重复启动仍有活动PID的任务。它**不等于自动覆盖残缺目录或从头重训失败任务**。训练完成也不等于评估完成；完整任务必须同时有`job.done`、应有的best/last评估summary和一致的状态记录。

发生异常时，先保存失败日志、退出码、原始目录、last.pt和计划。若能从完整last.pt恢复，使用原冻结实现与原配置，将恢复输出写到新的目录；若必须从零重跑，创建新tag并保留旧失败记录，不删除或覆盖证据。恢复沿用原优化器、replay/未更新rollout、随机状态和调度周期，不能因延长训练而重启降噪。训练已完成而评估中断时按`run.py finish`处理，半份评估目录先改名保留；具体命令见[操作说明](04_operations.md)。

历史seed1批次及其20种子评估保持原样。新功能修改了可变用户数实现和源码标识，旧检查点必须继续通过`frozen/wide_compute1000_w5_cutoff_20261020_01/`对应的原实现加载，**不放宽源码或检查点版本校验**。

## 8. 结果目录和汇总

```text
results/<tag>/
  suite_plan.json
  generated/common.json
  generated/environment/<case>.json
  cases/<case>/jobs/<method>/<method>/
    effective_*.json
    episodes.csv
    validation.csv
    updates.jsonl
    best.pt / last.pt
    evaluation_best/ / evaluation_last/
    plots/
    status.json / job.done
  reports/
    case_summary.csv
    all_episodes.csv
    main_training.csv
    single_selected_scene17.csv
    sweep_best.csv
    sweep_last.csv
    REPORT.md
```

各方法使用独立父目录，避免并发修改同一实验manifest。基线直接保存`evaluation/`，没有模型文件或训练日志；学习方法分别保存best/last评估。

`collect`从现有已完成文件汇总，不重训、不改选best。`all_episodes.csv`保留逐配置、逐方法、逐episode的主指标与协议；`sweep_best.csv`和`sweep_last.csv`分开整理4类参数变化，绝不把两种检查点混成一条曲线。完成与失败情况必须写入报告；缺失指标保留缺失，不能默认为0。

最终结论应同时比较J、平均时延、能量和可行性，并区分“时延最低但超预算”与“能量可行的综合结果”。单seed曲线不提供多训练种子置信区间，参数扫描也不保证提出方法在所有档位均最优。

`reports/main_training.csv`合并主场景已完成学习方法的逐轮训练记录（最终6×1000行）；`single_selected_scene17.csv`单独列出seed17/episode30000，`case_summary.csv`保留10个episode的均值。
正式调度还要求通过文件`ops/<tag>/validation.json`与冻结源码及文件清单一致，验证未通过不启动。
