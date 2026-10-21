# 参数与独立调试

## 1. 配置层级

`configs/environment.json`是本定稿默认环境，等于已完成停服宽场景；`configs/common.json`提供公共训练/评估设置；`configs/experiments/<ID>.json`覆盖各方法差异。每次运行保存 `effective_environment.json`、`effective_training.json`、`effective_protocol.json`，结果解释以这些实际副本为准。

可以单独改纯PPO、原混合PPO和两个改进PPO的recipe，不影响其他组。公共字段先读common，再被对应recipe的training覆盖；命令行明确覆盖项最后生效。不要为调参直接改算法Python源码，否则已有检查点源码核对会失配。

## 2. 默认物理/奖励参数

| JSON字段 | 默认值 | 单位/解释 |
|---|---:|---|
| num_users / num_uavs | 20 / 3 | 布局固定，不能只改数量字段扩规模 |
| episode_frames / frame_s | 60 / 1 | 槽数/秒 |
| area_m / user_cluster_halfwidth_m | 1000 / 250 | 地图边长/用户方形半边长，初始区500×500m |
| user_cluster_center_x_fraction / y | 0.5 / 0.5 | 地图中心 |
| altitude_m / speed_max_mps / safe_distance_m | 50 / 20 / 50 | m、m/s、m |
| local_cpu_hz / uav_cpu_hz / satellite_cpu_hz | 1e9 / 1e10 / 5e10 | 本地每用户/UAV每架/卫星CPU |
| uav_bandwidth_hz / satellite_bandwidth_hz | 1e7 / 1e8 | 每UAV/卫星带宽池，Hz |
| data_min_bit / data_max_bit | 5e5 / 2e6 | bit，0.5–2Mbit |
| cycles_per_bit | 1000 | cycles/bit |
| deadline_min_s / deadline_max_s | 0.5 / 2 | s；同时决定w=1/d |
| user_speed_alpha / user_direction_alpha | 0.6 / 0.9 | 高斯–马尔可夫记忆 |
| user_speed_noise_variance / user_direction_noise_std_rad | 2 / 2 | 速度方差/方向标准差rad |
| tx_power_w / carrier_hz | 0.2 / 2e9 | W/Hz |
| noise_reference_dbm / noise_reference_bandwidth_hz / noise_figure_db | −114 / 1e6 / 7 | 地面–UAV噪声参考 |
| los_a / los_b / los_extra_db / nlos_extra_db | 9.61 / 0.16 / 1 / 20 | UAV大尺度参数 |
| rician_k_db | 6 | UAV LoS K因子dB |
| satellite_altitude_m / satellite_min_elevation_deg | 600000 / 20 | m/度 |
| satellite_carrier_hz / satellite_gt_db_k / satellite_extra_loss_db | 2e9 / 1.1 / 3 | Hz、dB/K、dB |
| energy_budget_j / uav_joule_per_cycle | 12000 / 1e−9 | 每机J / J/cycle |
| uav_service_cutoff_enabled | true | 超预算下槽停止计算，飞行继续 |
| delay_cost_weight | 5 | 只乘奖励中的J主项，原始J不乘5 |
| deadline_penalty / energy_penalty | 1 / 2 | 超期/能量增量处罚 |
| collision_penalty / boundary_penalty / allocation_failure_penalty | 5 / 2 / 1 | 请求碰撞/越界/无效资源 |
| failure_delay_ratio | 100 | 仅非有限失败时延采用100d代理 |
| solver_max_sweeps | 5 | 有限关联扫描上限 |
| solver_epsilon / solver_certificate_tolerance / feasibility_tol | 1e−7 / 1e−6 / 1e−7 | 改进阈值/证书/可行容差 |
| greedy_direction_count / greedy_speed_levels / greedy_refine_rounds | 16 / 4 / 0 | 共49个不同候选，无细化 |

推进功率系数与卫星轨道常数也在environment.json；卫星仰角信道表在physics.py。不要为提高FPS随意降低求解精度，或把Mbit改成MB而不重新核对负载。

## 3. 公共训练和评估

| 字段 | 默认 | 含义 |
|---|---:|---|
| episodes | 1000 | 60槽共60000个环境transition |
| seed | 1 | 训练与场景主种子 |
| num_envs / use_processes | 8 / true | CPU进程环境，神经网络批量推理 |
| device / hidden | auto / 128 | 自动CUDA或CPU / 隐层宽度 |
| gamma / ppo_gae_lambda | 1 / 0.99 | 固定回合等权目标 / GAE |
| ppo_rollout_frames / ppo_minibatch_size | 480 / 128 | 全部环境合计transition/小批量 |
| ppo_epochs / ppo_critic_epochs | 10 / 10 | Actor/Critic更新轮数 |
| ppo_lr / ppo_critic_lr | 1e−4 / 3e−4 | PPO初始学习率 |
| ppo_clip_ratio / ppo_target_kl | 0.2 / 0.02 | 裁剪 / KL早停 |
| ppo_value_clip / ppo_entropy_coef / ppo_value_coef | null / 0 / 0.5 | 价值裁剪关闭/熵系数/价值系数 |
| ppo_max_grad_norm | 0.5 | 梯度裁剪 |
| actor_lr / critic_lr / tau | 1e−4 / 3e−4 / 0.005 | SAC/DDPG参数 |
| batch_size / replay_capacity | 256 / 100000 | SAC/DDPG批量/回放容量 |
| random_steps / critic_warmup_steps | 512 / 512 | 随机动作预热/额外Critic预热 |
| initial_alpha / alpha_lr | 0.01 / 1e−4 | SAC熵系数初值/学习率 |
| exploration_sigma | 0.1 | DDPG探索噪声 |
| eval_every / checkpoint_every | 3000 / 3000 | 总环境步间隔 |
| eval_episodes / eval_start | 3 / 20000 | 训练中选best的验证episode |
| selection_mode | feasible_return | 全验证集期限与电池可行优先，再回报，再J |
| render_every_episodes / log_frames | 10 / false | 每10回合刷新图/默认不记逐槽训练CSV |
| evaluation.episodes / episode_start | 10 / 30000 | 原best与last各10开发场景 |
| evaluation.policy_seed / also_sample | 1 / false | 独立采样流/默认不额外采样执行评估 |

步数与周期须能被num_envs整除，PPO总步数最好为rollout长度整数倍。两个混合改进recipe的 `options.noise_decay`、`options.lr_decay` 默认均true，其余组未启用这两项PPO调度。延长训练保留原调度长度，不能无意重新升温。

## 4. 每组独立参数示例

例如只改 `configs/experiments/hybrid_ppo_ar05.json` 中的training：

```json
"training": {
  "algorithm": "ppo",
  "control": "hybrid",
  "ppo_lr": 0.00008,
  "ppo_critic_lr": 0.0003,
  "ppo_rollout_frames": 960,
  "ppo_variant": "value_decomp_popart",
  "ppo_motion_memory": true,
  "ppo_ar_rho": 0.5,
  "ppo_ar_adaptive": false,
  "ppo_ar_rho_max": 0.9,
  "ppo_continuation_critic": true,
  "ppo_continuation_mode": "known",
  "ppo_known_delay_feature": true
}
```

这是字段写法示例，不是已经验证优于默认的推荐参数。保留该recipe原有结构字段，只加入/修改所需训练字段；使用新tag运行。改变环境参数时，复制environment.json并通过 `--environment` 指定，所有对比方法统一使用同一副本。

GPU负责网络推理/更新，SciPy与关联求解仍在CPU。测试并行度看 `session_fps`、`rollout_fps`、`update_wall_s`，不要只看GPU占用；关闭自动绘图可减少导出开销，episode日志仍保留。
