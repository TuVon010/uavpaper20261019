# 后续重新运行：命令与边界

以下命令均在定稿项目根目录执行。本页只是操作说明；本次整理和画图没有执行新的训练。

## 1. 已完成数据只重画

画图工具只读取已有CSV、JSON和轨迹，选择显示算法不会改变原始数据。隐藏AR=0.5时，改绘图配置里的展示列表即可，不删除其数据和检查点。原始seed17批次的配置、模型、结果及冻结实现继续保留。

## 2. 完全复现已有主实验或任意扫描点

新脚本`tools/reproduce_suite.py`从已有不可变`suite_plan.json`复制输入，始终从零训练，输出写到新的tag。它复用原冻结运行目录与原验证证据，不读取后来修改的工作区训练recipe。默认只打印命令，`--dry-run`明确表达这一行为；只有`--execute`才开始训练，`--prepare`只生成计划。

主场景仅重跑原PPO与自适应AR：

```powershell
python tools/reproduce_suite.py --tag rerun_main_01 --cases main --methods hybrid_ppo hybrid_ppo_ar_adaptive --dry-run
```

主场景8种方法：

```powershell
python tools/reproduce_suite.py --tag rerun_main_02 --cases main --dry-run
```

25用户场景只重跑原PPO：

```powershell
python tools/reproduce_suite.py --tag rerun_users25_01 --cases users_25 --methods hybrid_ppo --dry-run
```

144组全部复现：

```powershell
python tools/reproduce_suite.py --tag rerun_full_01 --dry-run
```

服务器Linux上把上面`python`替换为`.venv/bin/python`，确认预览范围后把`--dry-run`改为`--execute --parallel 4 --device cuda`。例如主场景8组后台运行：

```bash
cd /home/zhaojunan_25/uav_mec_final
tmux -L mec-final new-session -d -s mec-rerun-main-02 '.venv/bin/python -u tools/reproduce_suite.py --tag rerun_main_02 --cases main --execute --parallel 4 --device cuda > rerun_main_02.log 2>&1'
```

关闭SSH或本地电脑不会终止tmux中的作业。每次使用新tag与独立会话名。脚本拒绝已有结果目录，不删除失败数据、不把旧模型当新训练。已有验证文件会被检查为`passed=true`、source匹配且冻结文件hash匹配；若删除了原`ops/<原tag>/validation.json`，应从归档恢复原件，不能手写一个“passed”。可以通过`--runtime`和`--validation`显式指向保留的原冻结目录与证据。

原输入位于`results/seed17_sensitivity_20261020_01/generated/`；脚本默认从该批次复现。`--from-suite results/<其他完整批次>`可另选来源。可用case ID在来源`suite_plan.json`的`cases[].id`里查看。18个唯一case由20个扫描点复用主配置，不把用户20、CPU10GHz、带宽10MHz重复训练三次。

`--prepare`会创建新结果目录和空报告，不训练。准备后使用既有队列工具启动（此时不要再对同一tag执行`reproduce_suite.py`）：

```bash
.venv/bin/python tools/sensitivity_suite.py run --project . --runtime frozen/seed17_sensitivity_20261020_01 --root results/rerun_main_02 --validation ops/seed17_sensitivity_20261020_01/validation.json --parallel 4 --device cuda
```

新子集计划仍可用原工具`collect --root results/<新tag>`检查。仅重跑扫描点而不包含main时，无法在此新子集内部重新比较它与main的场景哈希；应连同原归档或额外main一起核验，不夸大配对范围。

## 3. 自己改变参数后进行新实验

这是一个新配置实验，不能用“完全复现”脚本。物理参数位于`configs/environment.json`，公共训练和评估配置位于`configs/common.json`，每个方法自己的覆盖项在`configs/experiments/<method>.json`。合并顺序为公共training→方法recipe training→允许的命令行覆盖，因此某字段若已在recipe内设置，只改common不会生效。

先复制环境和公共JSON再修改；已有结果和`frozen/`目录中的文件不要改。例如先保存`configs/environment_trial01.json`、`configs/common_trial01.json`，然后：

```powershell
python run.py run --experiment hybrid_ppo --environment configs/environment_trial01.json --common configs/common_trial01.json --tag trial01 --device auto
python run.py batch --experiments hybrid_ppo hybrid_ppo_ar_adaptive --environment configs/environment_trial01.json --common configs/common_trial01.json --tag trial02 --device auto
```

第一个命令运行单组，第二个按顺序运行所选组；训练完成后自动评估best和last，不需要另发evaluate。未指定`--episodes`时预算来自common；本定稿1000轮×60槽。`effective_environment.json`、`effective_training.json`、`effective_protocol.json`是该运行实际采用的配置。纯PPO用户数改变后需从头构造对应维度网络，不能直接载入原N=20检查点。

两个改进混合PPO的降噪与学习率衰减可独立关闭，且不会改JSON：

```powershell
python run.py run --experiment hybrid_ppo_ar_adaptive --tag adaptive_no_schedules01 --no-noise-decay --no-lr-decay
python run.py run --experiment hybrid_ppo_ar05 --tag ar05_lr_only01 --no-noise-decay --lr-decay
```

这两个开关仅对支持调度的改进混合recipe有效，不能传给含其他方法的全批命令。控制组和改进组使用相同物理场景与评估协议；新增试验应明确列出训练差异。

如果改变整套扫描配置：先复制`configs/suites/seed17_sensitivity.json`，修改方法列表、轴档位、预算等，再以新tag执行`tools/sensitivity_suite.py generate`。正式并行队列还需要**与新recipe和实现一致的完整冻结目录及真实验证证据**；生成计划不等于通过验证。原validation含73个文件的hash，不能拿旧proof为修改后的冻结recipe背书，也不能把未验证的新代码标为通过。原冻结runtime仅用于原recipe复现。

## 4. 中断、继续与评估完成

既有套件协调器停止时，先核对实际子进程。确认没有仍活跃子进程后才可用`tools/sensitivity_suite.py run ... --continue`恢复调度。它只跳过严格验证完成组；不会自动覆盖残缺目录，也不会把失败组静默从零重跑。

训练尚未完成、且last.pt完整时，用原冻结实现恢复到**新输出目录**，保留原训练配置和全部随机/优化器状态：

```bash
.venv/bin/python frozen/seed17_sensitivity_20261020_01/run.py run --experiment hybrid_ppo --resume results/<旧批次>/cases/main/jobs/hybrid_ppo/hybrid_ppo/last.pt --out results/manual_resume01/hybrid_ppo --tag manual_resume01 --device cuda
```

恢复不能同时传新的environment/common/调度开关。训练完成而评估未完，可在诊断半份评估目录并保留证据后，使用原实现`run.py finish --run <组结果目录> --device cuda`。有效完成需`job.done`、status、best/last summary及实际进程相互吻合，不能只看训练到60000步。

复现脚本只做新的从零实验，不自动修复原批次的失败或中断，也不覆盖原批次的完成标记。它不启动自动跟进，更不在完成后自动进入下一轮优化。
