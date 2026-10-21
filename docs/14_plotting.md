# 图表生成与自定义

## 1. 环境与入口

绘图只需要Python、numpy、matplotlib，不加载torch、检查点或仿真器，不启动训练。沿用项目requirements环境即可；本地已使用conda环境`py10`验证，服务器使用项目`.venv/bin/python`。所有命令从项目根目录执行。

```powershell
python tools/plot_results.py --workers 2
```

Linux为`.venv/bin/python tools/plot_results.py --workers 2`。`workers`是并行场景绘图进程数，默认2；内存紧张时改1。GPU不参与静态画图。

## 2. 输出内容

18个case各生成：3张训练奖励图（全部学习方法、混合方法、全部方法symlog副图），8张三指标柱状图（best/last、all/hybrid、10场景均值/选定单场景），2张无人机轨迹对照图（best/last），8张各方法自己的总览图。另有4类扫描×best/last×all/hybrid共16张折线图。默认共394张图，每张导出PNG、SVG、PDF和来源CSV；索引及manifest另计。

| 类型 | 路径（相对本批results根目录） |
|---|---|
| 总索引、扫描图 | `figures/paper/index.html`、`figures/paper/sweep_*.png` |
| 各case比较图 | `cases/<case>/figures/paper/` |
| 各实验独立总览 | `cases/<case>/jobs/<method>/<method>/figures/paper/overview_best.png` |
| 每图来源CSV | 同目录`source_data/<同名>.csv` |
| 图表与来源文件SHA256 | `figures/paper/manifest.json` |

打开本地HTML索引可按场景浏览、查看图的统计说明和下载不同格式；不需要启动网页服务。SVG文字可编辑，PDF保留矢量曲线。PNG用于预览。轨迹CSV包含time0到time60的真实初值与动作后坐标，不从第一步后的点冒充初始位置。

## 3. 算法显示开关

编辑`configs/figures.json`的`methods`。例如隐藏AR=0.5：

```json
{"id":"hybrid_ppo_ar05","label":"Hybrid PPO AR=0.5","short_label":"AR-0.5","enabled":false,"color":"#CB76A5","marker":"s","linestyle":":","hatch":"--"}
```

保留该条其余属性，只改`enabled`即可。全局关闭对all、hybrid视图、轨迹、各实验总览都生效；不会删除该方法的CSV或模型。顺序由`methods`列表顺序控制，图例名改`label`，柱状图短名改`short_label`。

不动配置，临时另存一组：

```powershell
python tools/plot_results.py --methods baseline_priority_greedy baseline_greedy_solver hybrid_ppo hybrid_ppo_ar_adaptive --output-tag four_methods
```

`views`是额外的比较子集。默认all包含所有启用的方法；hybrid只包含5种混合学习方法并在标题明确标注。`enabled=false`优先于所有view。命令行显式请求了已禁用/不存在的方法会报错，不会悄悄换方法。

同一`output-tag`重跑会覆盖同名派生图、来源CSV和本次索引；该tag目录里之前生成但本次未选的旧图可能仍在磁盘，但不会进入新索引或manifest。推荐展示方案变化时用新tag。已发布的`paper`版本若修改后重画，完整性校验会提示它与定稿清单不同，这是预期行为；要保留原定稿请另存tag。

## 4. 常用命令

只重画主场景（不画扫描）：

```powershell
python tools/plot_results.py --cases main --groups training performance trajectories per_run --output-tag main_only
```

只画四类参数扫描：

```powershell
python tools/plot_results.py --groups sweeps --output-tag sweep_only
```

只画20用户的best轨迹：

```powershell
python tools/plot_results.py --cases main --groups trajectories --checkpoint best --output-tag trajectory_only
```

只看last：

```powershell
python tools/plot_results.py --checkpoint last --output-tag last_only
```

指定别的已完成套件：

```powershell
python tools/plot_results.py --suite results/rerun_main01 --groups training performance trajectories per_run --output-tag paper
```

数据读取会检查完整评估和真实初值。当前绘图协议针对本项目60槽/10评估episode；新实验若改变这两个长度，需要同步更新读取校验，不应把缺少的数据当完整结果。仅有部分case的重跑套件建议明确指定上述四组，避免索取未运行的扫描点。

## 5. 平滑、误差线、单位

- `training.smoothing_window=25`：向后窗口均值；前24个点用实际可用样本数，不补0。`show_raw=true`保留浅色原曲线。没有裁剪异常值，没有用平滑后的值替换原始结果。
- `training.all_symlog_companion=true`：另存对称对数坐标副图，用于Pure PPO极端长尾；原线性全范围图仍保留。
- `performance.error_bars="range"`：min–max范围来自同一模型的10个配对场景，**不是置信区间或10次独立训练**。改为`"none"`关闭误差线。
- `performance.selected_scene=true`：额外保存episode30000单场景图，标题明确；不用它冒充10场景均值。
- `trajectories.episode=30000`：可选择已存30000–30009中的其他场景；best/last模型不变。
- 柱状图能量上方`k/10`是**全部3架机分别满足预算**的episode数量。36kJ虚线仅作总预算参照；一个UAV超预算即不可行。
- 扫描图直接连接已运行点，不拟合、不插值、不改变横坐标。数据尺寸单位Mbit，主场景数据范围0.5–2Mbit；尺寸扫描使用[0.4μ,1.6μ]。

`formats`默认`["png","svg","pdf"]`；`dpi=220`只影响PNG及可能的栅格元素。调图例、颜色、平滑窗口、格式等无需重新训练。

## 6. 可追溯性

每张图的来源CSV保留方法ID、case、checkpoint、逐episode或逐槽值。`training_reward_*.csv`同时保存原始episode ID、完成顺序、end_step、raw_return、trailing_mean。训练横轴是完成episode数，不是网络梯度更新次数。由于单训练种子，图不宣称跨训练种子的统计显著性。

模型best的选择沿用训练时验证协议，作图工具没有任何选checkpoint功能；last只读最后模型的既有评估。图看起来不理想的长尾、能量超限和失败均保留。原始文件的核验见`python tools/verify_plot_release.py --full-data`。
