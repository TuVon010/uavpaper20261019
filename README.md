# UAV-MEC 可调试实验项目

本文件夹是独立的新项目。原 `uav_mec_priority`、历史代码和结果不改动。本项目不包含旧实验结果、虚拟环境、服务器凭据或旧版本冻结目录。

默认物理参数、槽首服务时序、期限优先级和原奖励函数保持当前版本。这里整理的是可复现实验工具，不预设改进算法一定优于原 PPO。

## 先运行

要求 Python 3.10 或以上。在本文件夹打开终端：

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py list
```

先确认版本至少3.10；如果系统默认还是3.9，请使用已安装的3.10或以上解释器创建虚拟环境。后面命令中的 `python` 指安装了这些依赖的解释器。也可直接用 `.\.venv\Scripts\python.exe`，不必激活虚拟环境。Linux 对应 `.venv/bin/python`。

先做软件联调（每组只训练 16 步，不代表效果）：

```powershell
python run.py batch --smoke --tag smoke01 --device cpu
```

正式运行全部 **10 组**，默认每个学习方法 1000 episode × 60 帧 = 60000 环境步：

```powershell
python run.py batch --tag main01 --device auto
```

批处理按顺序运行，先两个基线，再纯 RL，最后混合方法。不会自动覆盖已有结果。单独运行：

```powershell
python run.py run --experiment hybrid_ppo_ar_adaptive --tag tune01 --device auto
```

## 调参数

| 文件 | 修改内容 |
|---|---|
| `configs/environment.json` | 场景、任务、信道、飞行、能耗、奖励罚项、求解精度 |
| `configs/common.json` | 总 episode、并行环境数、学习率、训练/验证周期、最终评估场景 |
| `configs/experiments/*.json` | 各方法的差异配置，覆盖 common 中的同名训练参数 |

参数顺序：公共训练配置 → 方法配置 → 明确指定的命令行参数。CLI 没指定的值不会覆盖 JSON。恢复训练严格保留原配置及衰减进度。

两个混合改进 PPO 默认开启降噪和 Actor 学习率衰减；可以独立关闭：

```powershell
python run.py run --experiment hybrid_ppo_ar05 --tag no_decay01 --no-noise-decay --no-lr-decay
```

其余方法默认没有这两个调度，开关也不允许误用于其他实验。AR=0.5 指时间自回归系数，不是动作放大 0.5 倍。

## 看训练曲线和最终轨迹

训练每完成一个 episode 就在 `episodes.csv` 记录一行。默认每累计 10 个完成的 episode 重画静态曲线；实时网页每 5 秒读取最新日志。

```powershell
python run.py dashboard --run results/main01 --port 8765
```

打开 [本机实时面板](http://127.0.0.1:8765)。也可把 `--run` 换成单个方法目录。

实验结束自动生成：奖励、时延成本、实际时延、完成率、能耗、违规曲线，以及 **best/last 两套三机轨迹和方法对比图**。手动重画：

```powershell
python run.py plot --run results/main01/hybrid_ppo_ar_adaptive
python run.py trajectories --run results/main01/hybrid_ppo_ar_adaptive --checkpoint last
python run.py compare --batch results/main01 --checkpoint last
```

- 单组图：`results/main01/方法名/plots/`。
- 最佳模型对比：`results/main01/comparison/best/`。
- 最后模型对比：`results/main01/comparison/last/`。
- 图像同时输出 PNG、SVG、PDF；原始 CSV/JSON 可直接做自己的图。

轨迹包含槽首初始点和全部槽末位置，60 帧有 61 个位置点，标出起点、终点、用户轨迹。两组基线无需训练，不会伪造奖励训练曲线。

详细说明：

- [系统模型与奖励](docs/01_system_model.md)
- [10 组方法与求解流程](docs/02_methods.md)
- [参数调试说明](docs/03_parameters.md)
- [运行、恢复与服务器命令](docs/04_operations.md)
- [日志、图表和代码结构](docs/05_outputs_and_code.md)
- [联调验证记录](docs/06_validation.md)
