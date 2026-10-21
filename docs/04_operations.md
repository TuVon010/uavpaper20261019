# 本地与服务器操作

## 1. 已完成结果：只查看即可

本地目录为 `E:/postgraduateLife/paper_code/uav_mec_final_1019`，服务器为 `/home/zhaojunan_25/uav_mec_final`。下列命令在项目根目录执行；服务器使用 `.venv/bin/python` 代替 `python`。本次定稿不会再次训练或重新评估已完成批次。

```powershell
python run.py dashboard --run results/wide_compute1000_w5_cutoff_20261020_01 --port 8765
python run.py plot --run results/wide_compute1000_w5_cutoff_20261020_01/hybrid_ppo_ar05
python run.py trajectories --run results/wide_compute1000_w5_cutoff_20261020_01/hybrid_ppo_ar05 --checkpoint best
python run.py trajectories --run results/wide_compute1000_w5_cutoff_20261020_01/hybrid_ppo_ar05 --checkpoint last
python run.py compare --batch results/wide_compute1000_w5_cutoff_20261020_01 --checkpoint best
python run.py compare --batch results/wide_compute1000_w5_cutoff_20261020_01 --checkpoint last
```

面板只绑定127.0.0.1。图表由已有CSV/JSON重绘，不更新模型。原始episode曲线与MA25同时保留，平滑线不是额外训练或统计置信区间。每组 `plots/` 包含训练/验证曲线、单机能耗和best/last轨迹；整批 `comparison/best/`、`comparison/last/` 含各方法对照与轨迹网格。实际轨迹按q₀…q₆₀画61点，与Actor请求速度区分。

20种子数据在同一批次的 `evaluation_seed20_20261020_01/`：直接看REPORT.md、summary.csv、per_seed_metrics.csv和weighted_delay_by_seed.csv。只有需要由原始种子JSON重新整理表格时才运行：

```powershell
python tools/report_seed20.py --root results/wide_compute1000_w5_cutoff_20261020_01/evaluation_seed20_20261020_01
```

## 2. 新实验

```powershell
python -m pip install -r requirements.txt
python run.py list
python run.py batch --smoke --tag smoke_trial01 --device cpu
python run.py run --experiment hybrid_ppo_ar05 --tag trial01 --device auto
python run.py batch --tag trial02 --device auto
```

本定稿全批默认就是8组。`--smoke`会将episode缩短为4槽并精简网络、预算和基线搜索，仅检查接口；不能将其指标作为正式实验。每个正式学习组默认1000个完整episode/60000步；两种基线无训练，直接同场景评估。

独立环境/公共训练副本示例：

```powershell
python run.py run --experiment pure_ppo --tag pure_trial01 --environment configs/environment_trial01.json --common configs/common_trial01.json
python run.py batch --experiments hybrid_ppo hybrid_ppo_ar_adaptive hybrid_ppo_ar05 --tag ppo_trial01
```

先复制所需JSON再改少量参数；每个组自身的学习率、PPO设置等可直接写入对应recipe。不要在已有结果目录改配置后覆盖运行。

## 3. 改进PPO可选调度

两个改进混合PPO默认 `noise_decay=true`、`lr_decay=true`，可独立关闭：

```powershell
python run.py run --experiment hybrid_ppo_ar05 --tag ar05_none01 --no-noise-decay --no-lr-decay
python run.py run --experiment hybrid_ppo_ar05 --tag ar05_noise01 --noise-decay --no-lr-decay
python run.py run --experiment hybrid_ppo_ar05 --tag ar05_lr01 --no-noise-decay --lr-decay
```

这两个命令行开关仅允许对应改进组；不要传给含原PPO/基线的全批命令。全批改变调度请先修改两个recipe的options。

## 4. 停止、恢复与最终评估

前台Ctrl+C会在完成当前环境调用后保存完整last.pt，停止整批；solver调用较慢时并非立即退出。best.pt只供评估，恢复用last.pt，目标写到新tag：

```powershell
python run.py run --experiment hybrid_ppo_ar05 --resume results/trial01/hybrid_ppo_ar05/last.pt --tag resume_trial01
python run.py run --experiment hybrid_ppo_ar05 --resume results/trial01/hybrid_ppo_ar05/last.pt --episodes 1500 --tag extend_trial01
```

恢复保存优化器、环境、随机状态、未更新rollout或replay、已完成日志和选模记录；延长训练不重启原衰减周期。恢复使用检查点内配置，不能借修改JSON悄悄改变已保存运行。源码必须匹配；CPU/GPU可用于另行评估，精确恢复要求原设备类型。已有停服批次使用冻结实现严格复现，不修改版本检查。

训练完成但评估/绘图中断时：

```powershell
python run.py finish --run results/trial01/hybrid_ppo_ar05 --device auto
```

已有完整且协议匹配的评估可复用；未完成评估目录先改名保留，不能覆盖半份证据。完整实验要求job.done、相应summary和无实际训练/评估进程相互一致；仅status写训练完成不够。

单模型另行评估示例：

```powershell
python run.py evaluate --checkpoint results/trial01/hybrid_ppo_ar05/best.pt --episodes 20 --episode-start 31000 --out results/manual_eval_trial01 --device auto
```

此命令是固定训练种子下的20个连续episode，不是20个不同主种子。要复核本次“种子1–20、各episode 30000”的固定原best协议，用以下工具，输出必须是不存在的新目录；工具固定读取原批次与冻结源码，不能用于重新选best：

```powershell
python tools/evaluate_seed20.py --project E:/postgraduateLife/paper_code/uav_mec_final_1019 --out E:/postgraduateLife/paper_code/uav_mec_final_1019/results/wide_compute1000_w5_cutoff_20261020_01/evaluation_seed20_recheck01 --parallel 4 --device auto
python tools/report_seed20.py --root results/wide_compute1000_w5_cutoff_20261020_01/evaluation_seed20_recheck01
```

新研究的最终测试仍应预先锁定协议。20000起用于选模，30000起以及这次20种子结果属于开发比较，50000起预留测试未作为本次结论来源。

## 5. 服务器后台与同步

不在文档、命令或新脚本中保存登录凭据。服务器进入已有项目，复用其独立环境；新实验示例：

```bash
cd /home/zhaojunan_25/uav_mec_final
tmux list-sessions
tmux new-session -d -s mec-trial01 '.venv/bin/python -u run.py batch --tag server_trial01 --device auto > server_trial01.log 2>&1'
tmux attach -t mec-trial01
```

先检查是否已有同名会话或同目录任务，避免重复启动。Ctrl+B后按D脱离，客户端关机不影响tmux。调试可加 `--synchronous`、`--num-envs 1`；正式速度测试再恢复多进程。GPU只加速网络，求解器仍在CPU。

本地/服务器同步应核对文件哈希及有效配置，不凭目录名或修改时间判断。本版同步证据见根目录定稿清单；已完成结果及冻结源码不跟随后续源码改动重写。
