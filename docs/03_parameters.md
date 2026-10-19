# 如何调参数

## 推荐步骤

1. 复制一份 `configs/environment.json` 或 `configs/common.json` 为自己的配置，例如 `configs/env_trial01.json`。
2. 每次只改少数相关参数，用新的tag运行。旧结果不覆盖。
3. 先 `--smoke` 排查接口，再去掉该参数做正式训练。smoke会改成4帧episode和小网络/短训练，不能用作物理结果。
4. 比较时确保所有方法用同一环境副本、同一最终评估编号和种子。`compare`会检查环境与逐episode/逐frame外生哈希，不一致就拒绝比较。

```powershell
python run.py run --experiment hybrid_ppo_ar05 --tag trial01 --environment configs/env_trial01.json --common configs/common_trial01.json
```

## 场景字段（SI单位）

| 目的 | JSON字段 | 说明 |
|---|---|---|
| 回合/槽长 | episode_frames / frame_s | 默认60/1；总步数=episode×frames |
| 区域与用户初始分布 | area_m / user_cluster_halfwidth_m / user_cluster_center_x_fraction / user_cluster_center_y_fraction | 默认1000/100/.5/.5；正方形须在区域内 |
| 速度/高度/防撞 | speed_max_mps / altitude_m / safe_distance_m | 默认20/50/50 |
| CPU | local_cpu_hz / uav_cpu_hz / satellite_cpu_hz | Hz；10GHz应写10000000000 |
| 带宽 | uav_bandwidth_hz / satellite_bandwidth_hz | 每架独立10MHz / 卫星100MHz，Hz |
| 数据 | data_min_bit / data_max_bit / cycles_per_bit | bit；1MB=8000000bit，不要误写为1000000 |
| 截止时间 | deadline_min_s / deadline_max_s | 秒，同时改变优先级1/d |
| 用户速度过程 | user_speed_alpha / user_speed_noise_variance | .6 / 2，后者是方差 |
| 用户方向过程 | user_direction_alpha / user_direction_noise_std_rad | .9 / 2，后者是标准差rad |
| UAV大尺度信道 | los_a / los_b / los_extra_db / nlos_extra_db | 9.61 / .16 / 1 / 20 |
| UAV小尺度 | rician_k_db | LoS为6dB；NLoS Rayleigh |
| 噪声 | noise_reference_dbm / noise_reference_bandwidth_hz / noise_figure_db | −114 / 1e6 / 7 |
| 卫星 | satellite_altitude_m / satellite_bandwidth_hz / satellite_gt_db_k / satellite_extra_loss_db / satellite_min_elevation_deg | 600000 / 1e8 / 1.1 / 3 / 20 |
| 电池/计算 | energy_budget_j / uav_joule_per_cycle | 每架每回合12000J / 1e−9J/cycle |
| 罚项 | deadline_penalty / energy_penalty / collision_penalty / boundary_penalty / allocation_failure_penalty | 默认1/2/5/2/1 |
| 求解精度 | solver_epsilon / solver_certificate_tolerance / feasibility_tol | 1e−7 / 1e−6 / 1e−7；勿为提速任意放松证书 |
| 关联扫描 | solver_max_sweeps | 默认5；上限截断须如实报告 |
| 贪心搜索 | greedy_direction_count / greedy_speed_levels / greedy_refine_rounds | 默认72/5/2，有限网格而非连续全局最优 |

当前动作/观测布局固定20用户、3UAV，`num_users`、`num_uavs`不是可直接改的通用规模开关，校验会拒绝其他数值。卫星LoS/阴影/衰落表在physics.py中；修改源码会使旧检查点版本核对不通过。

## 训练与评估

`common.json` 顶层 `episodes=1000`。`training` 保存 `TrainConfig` 字段；`evaluation` 控制训练结束的独立评估。

| 字段 | 默认 | 调试含义 |
|---|---|---|
| seed | 1 | 训练及场景主种子；全方法一致 |
| num_envs / use_processes | 8 / true | 并行采样；solver在CPU进程，网络批量在GPU |
| device | auto | 有CUDA则选GPU，否则CPU；启动打印实际设备 |
| hidden | 128 | 网络隐层宽度 |
| ppo_rollout_frames | 480 | 每次更新汇总的环境transition数，不是单环境帧数 |
| ppo_epochs / ppo_critic_epochs | 10 / 10 | Actor/Critic更新轮数 |
| ppo_minibatch_size | 128 | PPO minibatch |
| ppo_lr / ppo_critic_lr | 1e−4 / 3e−4 | PPO初始学习率 |
| ppo_clip_ratio / ppo_target_kl | .2 / .02 | PPO裁剪范围 / KL早停阈值；后者null可关闭 |
| ppo_value_clip | null | 默认不裁剪价值；可给正值明确启用 |
| ppo_entropy_coef / ppo_value_coef | 0 / .5 | 熵项 / 价值损失系数 |
| ppo_max_grad_norm | .5 | 梯度范数裁剪 |
| gamma / ppo_gae_lambda | 1 / .99 | 固定回合目标/优势估计 |
| eval_every / checkpoint_every | 3000 / 3000 | 总环境步间隔 |
| eval_episodes / eval_start | 3 / 20000 | 训练中选best的验证场景 |
| render_every_episodes | 10 | 每累计10个完成episode画图；0关闭自动训练绘图以测FPS |
| log_frames | false | 默认只记录episode；true另存逐帧train.csv，文件更大 |
| evaluation.episodes / episode_start | 10 / 30000 | best和last各评估相同10场景 |
| evaluation.also_sample | false | 可另加sample评估；不据此反选best |

环境步总数、eval_every、checkpoint_every、ppo_rollout_frames须能被num_envs整除。默认60000步也能被480整除，不留未更新rollout。使用自定义预算时，最好保持总步数也是rollout长度的整数倍。

两个混合改进方法配置的 `options.noise_decay` / `options.lr_decay` 可各设true/false。启用时：探索标准差乘数在总预算10%–70%按对数线性方式从1降到.1（κ=exp(a·log0.1)，a为此区间内进度）；Actor学习率按总进度线性从初值降到.2倍。后者不改变Critic学习率。调度只在新rollout开始切换，检查点保留最后一次rollout的实际尺度，不在评估时强行改成终点值。续训保留原调度总长度，不因延长预算重新升温。

## 吞吐调试

先比较num_envs=4/8/16的 `rollout_fps`、`session_fps`、`update_wall_s`；增加进程未必增益。网络小而求解器重时，GPU利用率可能低，GPU仅加速神经网络，不会自动加速SciPy/KKT或进程通信。

为测速度可临时将render_every_episodes设0，避免SVG/PDF导出干扰；每轮episodes.csv仍照常记录，实时面板也能读取。不要为了FPS降低内层资源解精度、跳过最终评估，或把恢复前的历史步数除以本次进程墙钟。
