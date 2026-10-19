# 交付联调记录

本轮交付目标是独立、可调参、可恢复、可看图的代码版本；没有启动正式1000episode服务器长训练。以下都是软件验证，不用于宣称新算法优于基线或已经收敛。

## 已通过

- 全套检查：**215 passed，4 skipped**。4项CUDA检查因本机没有GPU跳过。
- 十组完整短测：`results/validation_smoke_final/`。2基线完成评估，8学习组各训练16步、4个4帧episode，均有best/last评估、原始日志、PNG/SVG/PDF图像和完成标记。
- 每组学习方法episodes.csv正好4行，无重复episode。回报、J及罚项累计恒等关系正确；默认没有逐帧train.csv。
- 十组相同评估episode及逐帧外生哈希完全相同，不同帧任务随机量确实变化。
- 两套比较报告均验证10组场景配对，训练曲线图各含8个学习组，没有给基线伪造训练数据。
- 最终源码与所有短测检查点、评估summary记录的源码哈希一致。
- Windows真实多进程检查：`results/validation_process60/pure_ppo/`，2并行环境、2个60帧episode、120环境步；best/last轨迹各有 **61个位置点**。
- 默认参数下，新旧项目3个非零/零动作槽的信道、位置、关联、CPU/带宽、任务时延、奖励、推进/计算能量逐位一致。原项目未修改。
- 两个贪心基线的请求/执行速度、用户/UAV轨迹和推进能量一致，关联/分配结果不同。改变预算、能耗系数和累计能量不会改变贪心轨迹规划。
- 纯RL延续价值测试将求解器及prepare_service替换为报错函数，真实146维rollout与更新仍通过；未来任务表变化不会改变当前网络观测或当前价值。
- 在episode中间与PPO rollout中间断点恢复，最终模型数组、环境观测、随机状态和完整episode日志与不中断运行一致。绘图消费随机数的模拟也不改变训练状态。
- 可选调度、JSON参数传递、恢复保留衰减周期、不同方法禁止误标恢复、输出防覆盖、停止整批、相对路径搬运、缺失评估拒绝比较均已检查。
- 实时面板的真实HTTP读取、刷新、只读请求和路径限制已测试；绘图/面板入口不导入Torch。

结构化证据见 `validation_evidence.json`。当前核心源码SHA256：

```text
aed565f13500f850fad53b395c0cd3c8bee3d0892b3a4ddd980678fdf4d99fe4
```

## 验证环境

Python3.10.16，PyTorch2.0.1+cpu，NumPy1.26.4，SciPy1.15.3，Matplotlib3.7.1，Pandas2.0.2。新项目未复制虚拟环境，运行路径由使用者选择。

```powershell
python -m pip install -r requirements-dev.txt
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:COLUMNS='120'
python -m pytest tests -q --color=no -p no:cacheprovider
```

CUDA实际吞吐尚未在本轮本机验证。GPU/CPU由--device选择，manifest会记录实际硬件；短测FPS、16步曲线和两回合60帧轨迹都不能作为正式性能结果。正式实验应去掉--smoke并使用新tag。

开发检查的临时pytest目录和较早短测已移出交付项目，保存于当前工作区research下的验证归档中；新项目仅留下最终短测样例与文档。样例结果可独立画图，不会作为正式main01数据读取。
