# 智元酷拓 D1 Max 轮足版

型号 ZYCXD125M。规格书标注整机（含电池）41 kg，自由度为 12 个腿关节加 4 个轮毂关节。[智元产品资料](https://www.agibot.com.cn/support/materials/kuotuo/d1-max)

## 本地文件

| 路径 | 用途 |
| --- | --- |
| `max_description/urdf/max.urdf` | 官方 URDF，含连杆、质量与惯量、关节原点与轴、碰撞及外观网格引用 |
| `max_description/meshes/` | 29 个被 URDF 引用的 STL，以及原包中的 `base_link.m_p` |
| `max_description/launch/display_launch.py`、`package.xml`、`CMakeLists.txt` | 官方 ROS 2 描述包的可视化入口与包配置 |
| `reference/d1-max.step` | 官方整机 CAD，可供尺寸核对和碰撞建模 |
| `reference/d1-max-specifications.pdf`、`reference/d1-max-user-manual.pdf` | 官方规格书与说明书 |
| `d1_max.xml`、`scene.xml` | MuJoCo 机器人模型与平地场景 |
| `assets/base_visual.stl` | 从官方机身 STL 简化得到的外观网格 |

URDF 和网格来自[智元官方下载页](https://www.agibot.com.cn/support/software/p36)的 [URDF ZIP](https://cms-tos.agibot.com/legacy/cn/file/ueditor/php/upload/file/20260515/1778836628774244.zip)，已逐文件比对 [官方 GitHub 源码](https://github.com/AgibotTech/Agibot_D1_Max/tree/83bdebb04ae140f751e7c5e9e2c8be44aad62473/urdf/max_description)。本地保留原始文件和 ROS 路径。上游 `package.xml` 的许可证字段是占位文本。

## 仿真接入

从仓库根目录运行站立预览。

```bash
uv run python src/rl01/stand.py --robot d1-max --view
uv run python src/rl01/stand.py --robot d1-max --seconds 10
```

- MJCF 保留官方 URDF 的 29 个 link、12 个转动腿关节、4 个连续轮关节和 12 处固定连接的坐标系。活动关节保留原名、父子关系、原点、轴向和角度限位。质量合计 41.04532 kg，惯量沿用官方 URDF。浮动基座加 16 个活动关节得到 `nq=23`、`nv=22`、`nu=16`。
- URDF 前缀 `FBL`、`FAR`、`RBL`、`RAR` 分别对应左前、右前、左后、右后。每条腿的 URDF 后缀依次为 `ABAD`、`HIP`、`KNEE`、`FOOT`，SDK 对应的后缀依次为 `1_hip_roll`、`2_hip_pitch`、`3_knee_pitch`、`4_foot`。部署前还需确认真机接口中的索引、角度零点和符号。[SDK 关节命名](https://agibottech.github.io/Agibot_D1_Max/2.9%E5%85%B3%E8%8A%82%E5%91%BD%E5%90%8D%E8%A7%84%E8%8C%83.html)
- `qpos[7:]`、`qvel[6:]` 和 `ctrl` 按左前、右前、左后、右后排列，每条腿依次为髋外展、髋俯仰、膝俯仰、轮毂。`ctrl` 的腿关节目标用 rad，轮毂目标用 rad/s。`home` 站姿的四个膝盖都朝向机身中央，四轮目标转速为 0。运控 IMU 有陀螺仪、加速度和姿态传感器。
- 机身外观网格从 1,317,672 面简化为 149,998 面，其余外观使用官方 STL。碰撞使用机身盒体、腿部胶囊和半径 96 mm 的圆柱轮。接触摩擦、伺服增益和站姿是仿真参数，力矩上限取自 URDF 的 `effort`，腿关节 150 N·m、轮关节 50 N·m。官方 URDF 的 `velocity` 字段为 0，真机转速和力矩曲线仍需供应商数据或实测。
- D1 Max 有独立的平地训练入口，沿用仓库的 PPO。训练时仅去掉渲染网格，关节、惯量、碰撞、传感器和执行器与预览场景一致。

## 平地 PPO

```bash
uv run python -m rl01.train_d1_max --iterations 200 --num-envs 64
uv run python -m rl01.play_d1_max /path/to/model_200.pt
uv run python -m rl01.play_d1_max /path/to/model_200.pt --command 0.5 0.35 --view
```

训练输出包括检查点、配置和 `metrics.csv`。回放默认依次测试停车、前进、后退、左右转和带转向的前进，报告前向速度误差、偏航速度误差和摔倒数。省略检查点可跑同一组零残差基线。

- 策略以 50 Hz 更新，控制间隔内保持同一目标。直行和停车使用基础控制。转向时，腿部输出修正向内屈膝的站姿，轮部输出修正差速前馈轮速。残差随偏航指令幅值平滑放大，左右镜像由策略结构约束。动作按关节名映射，检查点记录 16 路顺序。
- Actor 的 52 维观测按机身角速度 ×0.3、机身系重力方向、前向及偏航速度指令、12 路腿角偏移、16 路关节速度、上一次裁剪后的 16 路动作排列。腿关节速度 ×0.15，轮毂速度 ×0.08。Critic 额外读取机身系线速度和机身高度。
- 任务覆盖平地前后滚动与差速转向，训练指令范围为前向 ±0.8 m/s、偏航 ±0.6 rad/s。训练对初始腿姿和地面滑动摩擦做了小范围随机化。轮胎侧滑、执行器延迟与真机编码器零点仍需通过实测校准。

## 真机资料

官方仓库包含 RobotSDK 0.1.1 的 C++ 头文件、示例和预编译库，并列出机载版本兼容表。公开接口能查到高层速度指令和关节状态订阅，逐关节位置、速度或力矩写入接口仍需向智元确认。[官方 SDK 仓库](https://github.com/AgibotTech/Agibot_D1_Max) · [SDK 开发指南](https://agibottech.github.io/Agibot_D1_Max/index.html)

## 全向与点按转向试训

全向入口是 `rl01.train_d1_max_omni`，使用原 MJCF 的 16 路执行器。前后靠滚轮，横移靠交替支撑和轮脚摆动，短转向指令触发一次低后轮脚摆动。每段保存优化器和阶段配置，以 `--max-seconds` 按实际用时截止，并通过 `tools/proc.sh` 与 `tools/supervise.py` 运行。GPU 试验可加 `--device cuda --gpu-memory-gib 10`。

```bash
uv run python tools/eval/d1_max_omni.py 测试检查点.pt --stage all --profile promotion --out 评估目录
```

评估按方向、点按和冲击分开写出指标，生成训练曲线、机身与轮脚轨迹和连续帧。点按轮脚净空按轮胎碰撞圆柱的最低点计算。阶段升档前，用相同指令和阈值对比，并将确定性与 `--sampled` 结果分别保存。
