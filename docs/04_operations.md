# 运行与调试操作

以下命令均在新项目目录中执行。建议Python3.10–3.12；本地检查使用Python3.10、CPU版PyTorch，依赖版本见验证记录。项目不硬编码本机旧虚拟环境位置，也不包含任何服务器登录资料。

## 第一次运行顺序

```powershell
python run.py list
python run.py batch --smoke --tag smoke01 --device cpu
python run.py batch --tag main01 --device auto
```

正式默认预算每个学习组1000个完整episode，8并行环境，总60000步。两种基线不训练，直接评估与训练组相同场景。批处理顺序如下：

1. baseline_priority_greedy，baseline_greedy_solver。
2. pure_ppo，pure_sac，pure_ddpg。
3. pure_ppo_ar_adaptive，pure_ppo_ar05。
4. hybrid_ppo，hybrid_ppo_ar_adaptive，hybrid_ppo_ar05。

同一批不能混合smoke与正式运行，也不能对已完成目录改参数后继续覆盖。`--smoke`明确将episode改成4帧、只跑16步、精简网络及贪心搜索，仅检查软件流程。

单组或选择部分组：

```powershell
python run.py run --experiment pure_ppo --tag pure01 --device auto
python run.py batch --experiments hybrid_ppo hybrid_ppo_ar_adaptive hybrid_ppo_ar05 --tag hybrid01 --device auto
```

单组都可用 `--episodes 1000 --num-envs 8` 覆盖；预算与周期须能被并行数整除。`--synchronous`让环境在同进程顺序执行，适合断点调试；`--no-synchronous`恢复多进程。

## 调度开关

推荐在两个混合改进JSON里改 `options`。命令行也可覆盖：

```powershell
python run.py run --experiment hybrid_ppo_ar05 --tag ar05_none --no-noise-decay --no-lr-decay
python run.py run --experiment hybrid_ppo_ar05 --tag ar05_noise --noise-decay --no-lr-decay
python run.py run --experiment hybrid_ppo_ar05 --tag ar05_lr --no-noise-decay --lr-decay
python run.py run --experiment hybrid_ppo_ar05 --tag ar05_both --noise-decay --lr-decay
```

开关仅用于这两组。不要把 `--no-noise-decay`直接传给含原始算法和基线的全批命令，后者会拒绝；全批需要关闭时，先改两个对应JSON的options，再运行新tag。

## 停止、恢复、继续整批

前台按Ctrl+C：保存完整last.pt后停止整批，不再启动下一组。信号处理中会先完成当前环境调用，solver较慢时不会立刻退出。`best.pt`是只供评估的所选模型，恢复必须用last.pt。

恢复到**新目录**，原目录保留：

```powershell
python run.py run --experiment hybrid_ppo_ar05 --resume results/main01/hybrid_ppo_ar05/last.pt --tag resumed01
```

延长训练到1500个episode：

```powershell
python run.py run --experiment hybrid_ppo_ar05 --resume results/main01/hybrid_ppo_ar05/last.pt --episodes 1500 --tag extended01
```

延长预算保留原探索衰减长度，不从头升温。恢复带走优化器、replay或未用完的PPO rollout、环境位置与随机状态、episode累计指标、历史曲线和best选择。修改当前JSON不影响恢复的保存配置；CLI非预算覆盖会被拒绝，避免记录与实际执行不一致。精确恢复保留设备类型，另行评估允许CPU/GPU切换。

源码哈希不同会拒绝旧检查点；配置JSON不计入源码哈希，实际环境/训练配置单独保存在检查点内严格检查。新项目不放宽旧版本检查，也不直接承接历史项目模型。改源码后应新开实验；旧结果仍可独立画CSV/JSON。

如果训练已完成、但绘图或最终评估被打断，先查看status.json，然后：

```powershell
python run.py finish --run results/main01/hybrid_ppo_ar05 --device auto
```

finish复用已完成且检查点/协议一致的评估目录。若某评估只有一半数据、没有summary，它不会覆盖；先将该不完整评估子目录改名保留，再重跑finish。

整个批次可用以下命令跳过已完整完成且配置完全相同的组：

```powershell
python run.py batch --tag main01 --device auto --skip-complete
```

未完成组不会盲目覆盖，请先显式恢复，或者用新的tag选择尚未运行的组。完成标志是 `job.done` 且best/last评估summary都存在；仅训练结束不算全实验结束。

## 单独评估与最终保留测试

```powershell
python run.py evaluate --checkpoint results/main01/hybrid_ppo_ar05/last.pt --episodes 10 --episode-start 30000 --out results/manual_eval01 --device auto
```

默认连续动作条件均值执行；`--sample` 使用独立seed1生成器，跨episode连续，不污染训练随机流。资源分类路由仍沿用原键控随机机制。模型在评估时使用其已保存的噪声尺度/AR，不临时另调参数。

默认20000起的3个场景用于训练中选best；30000起10场景用于开发比较。它们不能反复调参后冒充最终独立测试。50000起保留测试默认不运行；定稿锁定配置后再自行指定，对所有方法用同一组新编号。

## 服务器后台运行示例

本次交付只整理本地项目，不启动服务器长实验。把项目源码复制到服务器新目录，安装依赖后，可用已有tmux：

```bash
cd /your/path/uav_mec_final
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
tmux new-session -d -s mec-final '.venv/bin/python -u run.py batch --tag main01 --device auto > main01.log 2>&1'
tmux attach -t mec-final
```

Ctrl+B后按D可脱离tmux，客户端断网/关机后继续跑。是否已有同名会话先用 `tmux list-sessions` 检查，不重复启动。GPU驱动/PyTorch安装按实际机器匹配，启动日志和manifest记录实际设备。

本机实时面板：

```powershell
python run.py dashboard --run results/main01 --port 8765
```

服务只绑定127.0.0.1，Ctrl+C结束查看器不影响其他训练进程。要查看服务器结果，可以下载整个批次目录后在本地 `compare` / `dashboard`；清单使用相对组路径便于搬运。单组 `--out` 自定义路径时，其父目录作为批次目录，不要与已有别的批次混用。
