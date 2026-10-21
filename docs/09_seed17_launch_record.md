# Seed17主场景与参数扫描：启动记录

2026-10-20已完成配置、代码同步与启动。**本记录不代表144组正式实验已经完成。** 最新进度以服务器套件状态、已完成标记和实际进程为准；本地下载的是带时间戳的阶段快照。

- 本地：`E:/postgraduateLife/paper_code/uav_mec_final_1019`。
- 服务器：`/home/zhaojunan_25/uav_mec_final`。
- 批次：`seed17_sensitivity_20261020_01`。
- 冻结运行目录：`frozen/seed17_sensitivity_20261020_01`。
- 源码摘要：`3ef6b15c31a7de84f7ce1fb64ee1e70f71b252e8ee7db35e05766d1db35ca870`。
- 本地、服务器和新冻结目录的73个管理文件逐项一致。旧seed1结果及其冻结实现保留。
- 本地回归307项通过、5项跳过；服务器核心回归302项通过、1项跳过，新增调度器9项通过。
- GPU软件联调13组通过，覆盖8种方法、10/30用户纯PPO、30用户AR、不同用户数的SAC/DDPG，以及30用户真实检查点续训和评估。20用户与原实现的物理/动作/奖励输出逐位一致，求解精度未降低。
- 第一版短测脚本因同名实验共享清单被拒绝，原日志保留；第二版按调用隔离目录后通过。没有因此改变算法或正式预算。

## 正式任务与日志

主场景优先运行8种方法。之后依次自动排队四类单因素扫描，合计18个唯一配置、144个运行，其中108组从零训练、36组规则基线评估。每个学习方法1000个episode，每集60秒；训练和外生场景种子均为17。

用户确认任务大小均值单位为Mbit，相对均匀区间为`[0.4μ,1.6μ]`。主场景仍为`[0.5,2] Mbit`，不为对齐刻度修改它。其他设置与单因素约束见 `docs/08_seed17_sensitivity.md`。

服务器tmux的socket为`mec-final`，会话为`mec-seed17-suite-20261020`，最多4个运行并行。每个学习运行内部有8个环境；网络使用CUDA，关联/KKT与物理仿真使用CPU。长任务与SSH连接独立，本地电脑关闭后服务器会继续。

```bash
cd /home/zhaojunan_25/uav_mec_final
tmux -L mec-final attach -t mec-seed17-suite-20261020
cat results/seed17_sensitivity_20261020_01/suite_state.json
tail -n 30 ops/seed17_sensitivity_20261020_01/suite.log
```

每完成一个episode保存一行训练CSV；`render_every_episodes=100`仅降低重绘频率。主场景6个学习方法完成后，`reports/main_training.csv`共6000行，方便统一画训练曲线；每组原始`episodes.csv`仍保留。

## 表格与完成条件

结果目录为`results/seed17_sensitivity_20261020_01/`：

- `reports/main_training.csv`：主场景逐轮训练数据，尚未完成的方法不会伪造记录。
- `reports/single_selected_scene17.csv`：各配置在seed17/episode30000的单场景结果；主配置对应此前选定的外生场景。
- `reports/case_summary.csv`：各配置、方法的best/last在10个开发episode上的均值。
- `reports/sweep_best.csv`、`sweep_last.csv`：四类扫描长表，记录加权时延成本、平均秒时延和三机总能量，并保留完成率、回报和能量可行数。
- `reports/all_episodes.csv`：逐评估episode数据。
- `reports/REPORT.md`、`collection_audit.json`：自动刷新报告与配对核验。

只有144个运行全部通过训练/评估、配置、模型和同场景哈希检查，才写`suite.done`。运行失败或残缺输出会保留证据并标记，调度器不覆盖重跑；不能仅凭退出码或训练达到60000步判断整批完成。

后台跟进已启用，每30分钟检查一次。正常运行且无重要变化时不通知；主场景或全批完成时核对、下载数据与报告。该跟进仅负责本批实验收尾，不自动引入新的方法或参数优化。
