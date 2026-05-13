# MIDI-to-Dance

基于 MIDI 文件自动生成机器人下肢舞蹈轨迹，在 MuJoCo 中可视化仿真并同步播放音频。

## 原理

### 整体流水线

```
MIDI 文件 ──→ 音乐特征提取 ──→ PCA 动作生成 ──→ 轨迹合成 ──→ CSV 输出
                                         │
                                  ┌──────┼──────────────┐
                                  │ 连续载波 (7 PC)         │  ← 不可公度正弦波叠加，永不重复
                                  │ 音乐能量包络             │  ← 音符密度/重音调制幅度
                                  │ 事件重音                 │  ← 拍子屈膝、重音侧倾、乐句呼吸
                                  │ 跨步层（可选，默认关闭）   │  ← `--enable-steps` 启用
                                  └────────────────────────┘
                                         │
                              MuJoCo 可视化 + MIDI 音频同步
```

### PCA 动作基元提取 (`pca_extractor.py`)

从 `example*.csv` 真人动捕数据中离线提取下肢 PCA 基元。**默认流水线**在去趋势与镜像增广之后再做 SVD，使主成分更少受慢漂移与左右偏置污染（详见脚本顶部文档）。

```
示例 CSV × N  ──→  读 13 个下肢关节角 (50 Hz)
        │
        ├── 高通去趋势 (Butterworth 2 阶，零相位；默认截止 0.3 Hz)，剔除低频姿势漂移后再贴回均值姿态
        ├── 丢弃低速静止帧（默认去掉速度分布最慢的 10%，基于高通后的帧速度）
        ├── 左右镜像增广：复制一份 L↔R 交换 + 横向关节符号翻转（髋滚/髋偏航、踝滚、腰偏航）
        └── 中心化 + SVD  ──→  pca_model.npz（mean_pose、components、std_scores、explained_variance_ratio…）

训练帧数约为「过滤后帧数 × 2」（镜像）；当前仓库默认模型约 5.3×10⁴ 帧。
```

**与旧版「直接居中 + SVD」相比：** 高通削弱 <0.3 Hz 漂移主导的主轴；镜像使基底在镜像变换下闭合，均值姿态左右对称；静止帧不参与协方差估计。

**`pca_motion.py` 中的 PC 序号：** 默认导出 **按解释方差排序** 的 PC1–PC7；音乐驱动按 PC 索引绑定到上表协调模式（节拍摇摆、侧倾 ADSR、onset 深蹲、yaw 扭转等，详见下文架构图）。若你希望 PC 顺序与语义原型（深蹲 / yaw / 摇摆 …）更紧对齐，可在提取时加 `--canonical`（匈牙利算法按 |cosine| 重排）。

**7 个主成分（默认模型，按方差序）解释的方差、典型负载与调高权重效果：**

| PC | 方差 | 累积 | 简要含义 | 典型负载（绝对值较大者） | 调大 `--pc-weights` / `pc_weights[i]` 时 |
|----|------|------|----------|--------------------------|----------------------------------------|
| PC1 | 54.1% | 54.1% | 左右踝俯仰相反 + 髋俯仰协同（负重前后摇摆） | R/L `ankle_pitch`，R/L `pelvic_pitch` | 前后重心 / 前后晃身更明显 |
| PC2 | 17.0% | 71.0% | 双膝反对称弯曲 + 踝滚（侧向换重心 / 迈步感） | R/L `knee_pitch`，`ankle_roll` | 侧向一屈一伸的节奏性换重心更夸张 |
| PC3 | 13.8% | 84.8% | 双膝同弯 + 踝俯仰（对称深蹲分量） | R/L `knee_pitch`，R/L `ankle_pitch` | 蹲得更深、对称起伏更大 |
| PC4 | 5.5% | 90.3% | 双侧骨盆偏航同向 + 踝滚（躯干扭转） | R/L `pelvic_yaw`，`ankle_roll` | 整体偏航扭腰 / 转身感更强 |
| PC5 | 3.5% | 93.8% | 双侧髋俯仰同向 + 踝俯仰（前后俯仰倾斜） | R/L `pelvic_pitch`，`ankle_pitch` | 上身前倾、后仰弧更大 |
| PC6 | 2.6% | 96.4% | 双侧髋滚同向 + 膝差（侧倾 / 冠状面摆动） | R/L `pelvic_roll`，`knee_pitch` | 左右冠状面晃身更明显 |
| PC7 | 1.2% | 97.7% | 踝俯仰与膝协同伸展（细小伸展模式） | R/L `ankle_pitch`，`knee_pitch` | 轻微伸展 / 近似踮脚提拉感略强 |

（百分比随数据与 CLI 参数略有浮动；上表对应当前仓库默认生成的 `pca_model.npz`。最后一列对应调大 `--pc-weights`：`pc_weights[i]` 会同时放大该 PC 的连续载波与事件重音。）

**重建公式：**

```
lower_trajs[t] = mean_pose + Σ_i activation_i[t] × component[i]
```

其中 `mean_pose` 为预处理序列的时间平均姿态（镜像增广后左右对称）；`component[i]` 为第 i 个主成分的 13 维关节协调向量。

### 动作生成架构 (`pca_motion.py`)

采用 **连续载波 + 音乐调制 + 事件重音** 三层架构，外加一个可选的跨步层（默认关闭）：

```
                   ┌─────────────────────────────────────────┐
                   │        连续载波 (_continuous_carriers)    │
                   │   每个 PC 由 3 个不可公度正弦波叠加        │
                   │   周期 3~44 秒，频比无理，永不重复          │
                   │   幅值归一化到 [-1, 1]                     │
                   └──────────────┬──────────────────────────┘
                                  │
                   ┌──────────────┴──────────────────────────┐
                   │       音乐能量包络 (features.energy)      │
                   │   0.6·note_density + 0.4·smoothed_accent  │
                   │   → γ 形 envelope = 0.05 + 0.95×E^1.5     │
                   │   carriers × envelope × amps × CARRIER_GAIN│
                   └──────────────┬──────────────────────────┘
                                  │
                   ┌──────────────┴──────────────────────────┐
                   │          事件重音（叠加到 carrier 之上）  │
                   │   PC1 ← 节拍正弦 × 起始密度  (反对称负重摇摆)│
                   │   PC2 ← 重音低音 ADSR + 符号交替 (反对称侧倾)│
                   │   PC3 ← onset 屈膝脉冲 (+正激活 = 屈膝)     │
                   │   PC4 ← 音高调制 + 小节波 + 乐句脉冲 (yaw)  │
                   │   PC5 ← 乐句级 cos 半周期呼吸弧 (前倾)      │
                   │   PC6 ← 音符密度 × ~48 拍慢正弦 (侧倾)      │
                   │   PC7 ← 音高 register tanh (微伸展)         │
                   └──────────────┬──────────────────────────┘
                                  │（以下为可选）
                   ┌──────────────┴──────────────────────────┐
                   │   跨步层 (enable_steps=False 时跳过)       │
                   │   _identify_step_events: 重音低音触发铃形   │
                   │     脉冲（峰对齐触发帧，无视觉滞后）         │
                   │   _apply_step_motion: 抬腿膝弯 + 髋屈       │
                   │   rhythm_suppression_mask: 跨步窗口里压制     │
                   │     PC 与 groove，让抬腿读得干净             │
                   └──────────────────────────────────────────┘
```

> **事件 ↔ PC 映射逻辑：** 每个音乐驱动信号绑到与其动作类型匹配的 PC 槽——节拍正弦（自然左右交替）驱动反对称摇摆 PC1；onset 脉冲（一次性正激活）驱动对称下蹲 PC3；乐句弧（缓慢呼吸）驱动对称前倾 PC5……依此类推。这样 onset、节拍、音高、乐句各特征不会互相打架，而是各自激活最贴近其形态的协调模式。

> **节奏同步设计（修复后）：** `_stride_accent` 起音从 0.3 拍降到约 40 ms 恒定，侧倾峰瞬间到位、再缓慢释放；`_identify_step_events` 铃形 `sin²(π j / N)` 的写入窗口居中于触发帧（`[ti − N/2, ti + N/2)`），脚抬到最高那一刻正好落在重拍上。在 56–86 BPM 的测试 MIDI 上，事件峰相对触发帧的偏移从 +250 ~ +800 ms 修正到 ±0 ms。

**关键设计决策：**
- **不可公度周期**：7×3=21 个不同周期，频比皆为无理数，整首乐曲内动作不重复
- **Always-on 运动**：基线振幅 5%，保证安静段落也有微动；激烈段落振幅放大约 11×
- **音乐调制幅度而非触发事件**：音符密度决定动作有多大，而非触发是否动作
- **事件重音叠加 / 提前到位**：所有事件型激活在 ≤40 ms 内达到峰值，肉眼上与拍子同步
- **跨步层默认关闭**：`_identify_step_events` + `_apply_step_motion` 不是 PCA 基元，会偶发抬腿，要时用 `--enable-steps` 显式开启

**可调权重（运行时）：**
- `scale`（CLI `--scale`）—— 全局幅度，等比缩放 7 个 PC
- `pc_weights`（CLI `--pc-weights w1 … w7`）—— 每个 PC 独立倍乘，同时作用于该 PC 的 carrier 和 event accent；短列表自动 pad 1.0、长列表截断
- `enable_steps`（CLI `--enable-steps`）—— 启用可选的跨步抬腿层（默认关闭，纯 PCA 运动）

### MIDI 解析 + 音乐特征提取

`midi_parser.py` 把 MIDI 解析成 `MidiData`，关键设计：

- **完整 tempo map** —— 每个 `set_tempo` 都记录到 `tempo_map: List[(tick, microsec/beat)]`；`tick → seconds` 用分段积分而不是「先取一个 BPM 再乘」。
- **duration 加权 dominant BPM** —— 当存在多个 tempo 段时，`bpm` 取覆盖 tick 数最多的那段（而不是简单取最后一个或最初一个），ritardando 不会污染整曲 BPM。
- **time-signature 取首个事件**（一般在 conductor track），不被后续 track 的杂项覆盖。
- 每个 `NoteEvent` 在解析阶段就基于 tempo_map 算好 `time_seconds` / `duration_seconds`，下游不再以恒定 BPM 重新推算。

`feature_extractor.py` 在 `sample_times`（50 Hz）网格上输出以下特征。所有特征均与同一时间轴对齐。

| 特征 | 计算方式 | 用途 |
|---|---|---|
| `onset_strength` | 每帧 max(`velocity/127`) | 节拍重音强度调制 |
| `note_density` | onset 计数 × 高斯平滑（~1.5 拍） | 段落能量曲线 |
| `energy` | 0.6 · density + 0.4 · 平滑 accent | `pca_motion` 的能量包络直接使用 |
| `beat_phase` | `(t · bps) % 1` | PC1 节拍正弦 |
| `is_beat` / `is_downbeat` | 向量化 nearest-beat 检测 | step/stride 触发 |
| `metric_accent` | 节拍位置 → 0–1（downbeat=1.0、其他根据 time-sig 层级） | 强弱拍权重 |
| `pitch_level` | **note_on→note_off 整段填充** 后高斯平滑 | PC4 yaw 扭转、PC7 register 抬升 |
| `pitch_contour` | `pitch_level` 的梯度 × 拍长，clip 到 ±1 | 辅助节拍调制 |
| `accent` | 0.50·metric + 0.25·velocity_dev + 0.15·duration + 0.10·pitch_leap | 综合重音（机控 MIDI 也能产生有意义信号） |
| `is_low_note` | 低音 (pitch ≤ min+40% range) **沿持续时长填充** | PC2 跨步触发条件 |
| `phrase_boundaries` | onset 间隔 ≥ 2 拍 ∪ 8 小节定时网格 ∪ 静音段恢复点 | PC5 乐句呼吸、PC4 乐句脉冲 |
| `bpm` / `time_signature` | tempo_map 加权 / 首个 time-sig 事件 | 所有周期/节拍计算的基础 |

> **关键修复（对比旧版）：** 旧 `pitch_level` 只在 onset 帧赋值（占 ~3% 帧），高斯平滑后塌缩到 ≈0.01；旧 `is_low_note` 同问题，4 个示例曲只有 0.8%–2.8% 帧标 1；旧 `accent` 只用 velocity 阈值，而机控 MIDI 的 velocity std ≈ 2 rad，重音几乎是噪声。重写后 4 个曲子 step 触发数从 2–6 提升到 8–14（详见下表）。

| 示例曲 | BPM | 旧 pitch_level max | 新 pitch_level max | 旧 step 候选 | 新 step 候选 |
|---|---|---|---|---|---|
| yellow_bass | 86 | 0.029 | **1.000** | 6 | **14** |
| 为你着迷 | 116 | 0.034 | **0.930** | 2 | **13** |
| 光辉岁月 | 70 | 0.019 | **0.800** | 2 | **10** |
| 难忘今宵 | 56 | 0.013 | **0.857** | 6 | **8** |

### 安全约束

- 所有关节 clamp 到 URDF 定义的角度限制
- 高斯平滑 (σ=2 samples ≈ 40ms) 去除突变
- 运动学模式下通过迭代 IK 实现脚底贴地 + ZMP/CoM 平衡

## 使用方法

### 环境依赖

```bash
pip install mido numpy scipy mujoco matplotlib
```

### 步骤 1（一次性）：从动作捕捉数据提取 PCA 模型

```bash
python -m midi_to_dance.pca_extractor csv/example1.csv csv/example2.csv -o pca_model.npz
```

常用选项：
- `-o` — 输出 `.npz` 路径（默认 `pca_model.npz`）
- `-n` — 保留主成分个数（默认 7）
- `--fs` — 动捕采样率 Hz（默认 50）
- `--hp-cutoff` — 高通截止频率 Hz（默认 0.3；设为 `0` 关闭去趋势）
- `--static-quantile` — 丢弃全局速度分布最低的分位比例（默认 `0.10`；设为 `0` 保留全部帧）
- `--no-mirror` — 不做左右镜像增广
- `--canonical` — 按语义原型重排 PC 顺序（与 `pca_motion.py` 注释槽位对齐；默认关闭，保留按方差排序）

`pca_model.npz` 已预生成；更换 `csv/example*.csv` 或调整上述预处理时请重新运行本命令。

### 步骤 2：从 MIDI 生成舞蹈轨迹

```bash
# 基本用法（纯 PCA，不抬腿）
python -m midi_to_dance.main mid/yellow_bass.mid -o csv/output.csv

# 完整选项
python -m midi_to_dance.main mid/yellow_bass.mid \
    -o csv/output.csv \
    --dt 0.02 \                            # 采样周期 (默认 0.02s = 50Hz)
    --scale 1.0 \                          # 全局动作幅度缩放 (默认 1.0)
    --pc-weights 1.0 1.0 1.8 0.4 1.0 1.0 1.0 \  # 每个 PC 独立权重
    --enable-steps \                       # 可选：启用重拍抬腿（默认关闭）
    --stats \                              # 打印关节运动统计
    --plot                                 # 生成 matplotlib 可视化图表
```

`--pc-weights` 的 7 维顺序对应 PC1–PC7：1=前后摇摆，2=侧向膝伸缩，3=深蹲，4=偏航扭腰，5=前倾，6=侧倾，7=微伸展（详见 [动作生成架构](#动作生成架构-pca_motionpy)）。短列表右侧自动 pad 1.0，长列表截断。

`--enable-steps` 开启时会在 PCA 之上额外叠加一层重拍抬腿事件（钟形铃 + 节奏抑制掩码），并在 CSV 中追加 `left_foot_step` / `right_foot_step` 两列；默认关闭以保证机器人双脚全程锚定。

### 步骤 3：MuJoCo 可视化仿真

```bash
# 运动学仿真（默认）：直接设定关节角度 + 脚底贴地 + CoM 平衡
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid

# 动力学仿真：PD 位置执行器 + 物理解算
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid --dynamics

# 选项
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid \
    --slow 0.5 \           # 半速播放
    --no-audio \           # 禁用音频
    --dynamics \           # 动力学模式（默认运动学）
    --audio-offset 0.0     # 动作相对音频 Popen 时刻的延迟（秒）；正值延后动作
```

**仿真模式**：
- **运动学 (kinematic)**：每帧直接设置 `qpos` + 迭代 IK（脚底贴地 + ZMP 平衡）+ XY 锚定防滑步
- **动力学 (dynamics)**：27 个 position 执行器 (`kp=60, kv=4` 腿 / `kp=40, kv=3` 臂腰) + `mj_step`，含重力、接触力等物理效应。ZMP 反馈叠加到髋俯仰执行器

**音频/动作同步：** `sim_start_time` 在 `subprocess.Popen` 返回的瞬间记录（不再等待 0.5 s 验证），动作时钟以此为原点。`--audio-offset` 用于补偿不同音频后端的 buffer 延迟（paplay/aplay 典型 50–100 ms）：正值再延后动作、负值再提前。默认 0 在多数 PulseAudio 系统上已足够精准。

**仿真交互**：
- 鼠标拖拽：旋转视角
- 滚轮：缩放
- Esc：退出
- 空格：播放/暂停（视窗聚焦后）

### 输出格式

CSV 格式，首行为列名，每行一个时间帧（50 Hz）：

```
timestamp,left_leg_pelvic_pitch,...,left_shoulder_pitch,...,right_wrist_roll
0.000000,0.138518,...,0.170745,...,1.073393
0.020000,0.144098,...,0.170745,...,1.073393
...
```

至少 28 列：`timestamp` + 27 个关节角；当且仅当使用 `--enable-steps` 启用跨步层时，`trajectory_writer.py` 才会追加 `left_foot_step`、`right_foot_step` 两列。`simulate.py` 在缺失这两列时自动降级为「双脚全程锚定」。
- 下肢 13 关节：6 左腿 + 6 右腿 + 腰 → **PCA 生成**
- 左臂 7 关节 + 右臂 7 关节 → **中性姿态常量**（来自示例动捕数据的弹奏姿态）

## 项目结构

```
midi_to_dance/
├── __init__.py
├── midi_parser.py           # mido 解析 MIDI → NoteEvent/MidiData（全轨道扫描）
├── feature_extractor.py     # 音乐特征提取 (MusicalFeatures dataclass)
├── pca_extractor.py         # 高通去趋势 + 静止帧过滤 + 镜像增广 + SVD → pca_model.npz
├── pca_motion.py            # PCA 动作生成：连续载波 + 音乐调制 + 事件重音
├── trajectory_generator.py  # 调用 pca_motion + 关节限位 clamp + 高斯平滑
├── trajectory_writer.py     # CSV 输出
├── main.py                  # CLI 入口 + matplotlib 可视化
├── motion_primitives.py     # 旧版手调基元 (已弃用，保留参考)
└── simulate.py              # MuJoCo 仿真 (运动学/动力学) + MIDI 音频合成播放

pca_model.npz                # PCA 模型 (预生成)

csv/
├── example1.csv             # 动作捕捉示例 1 (~327s, 50Hz)
├── example2.csv             # 动作捕捉示例 2 (~265s, 50Hz)
├── output.csv               # 生成的轨迹
└── test_output.csv          # 测试输出

mid/
└── yellow_bass.mid          # 示例 MIDI 文件

casbot_band_urdf/
├── meshes/                  # STL 模型网格
├── urdf/                    # URDF 机器人定义
└── xml/                     # MJCF 机器人场景定义
```

## 自定义

### 调整动作幅度

```bash
# 全局等比缩放
python -m midi_to_dance.main mid/xxx.mid -o csv/xxx.csv --scale 1.5

# 每个 PC 独立调权（顺序：weight rock / lateral knee / squat / yaw / lean / sway / extension）
python -m midi_to_dance.main mid/xxx.mid -o csv/xxx.csv \
    --pc-weights 1.0 1.0 1.8 0.4 1.0 1.0 1.0   # 加强深蹲 PC3、收敛偏航 PC4
python -m midi_to_dance.main mid/xxx.mid -o csv/xxx.csv \
    --pc-weights 1.0 1.0 0.0                   # 消融实验：关掉 PC3，PC4–7 自动保 1.0

# 或编辑 pca_motion.py 中 generate_pca_motion() 的常量：
#   base_level     — 安静段最小振幅 (默认 0.05)
#   dynamic_range  — 能量调制范围 (默认 0.95)
#   envelope_gamma — 包络 γ 形指数 (默认 1.5)
#   CARRIER_GAIN   — 载波增益 (默认 2.4)
#   amps[i] * X    — 每个 PC 事件重音乘子 (PC1:1.5、PC2:1.5、PC3:2.0、PC4:1.5、PC5:1.4、PC6:1.4、PC7:1.2)
```

> `pc_weights` 同时作用于该 PC 的 carrier 和 event accent，权重直接乘进 `amps[i]`；与 `scale` 复合（`amps[i] = std_scores[i] · scale · pc_weights[i]`）。不影响 `_apply_step_motion` 与 `_groove_patterns` 里的绝对量。

### 启用 / 关闭重拍抬腿层

```bash
# 默认：纯 PCA，双脚全程锚定，CSV 不含 left_foot_step / right_foot_step
python -m midi_to_dance.main mid/xxx.mid -o csv/xxx.csv

# 显式启用：在 PCA 之上叠加单腿抬步事件
python -m midi_to_dance.main mid/xxx.mid -o csv/xxx.csv --enable-steps
```

### 替换动作捕捉数据

```bash
python -m midi_to_dance.pca_extractor csv/new_example1.csv csv/new_example2.csv -o pca_model.npz
```

重新生成后自动生效，无需修改代码。

### 调整主成分数量

```bash
python -m midi_to_dance.pca_extractor csv/example1.csv csv/example2.csv -n 5 -o pca_model.npz
```

### 替换手臂姿态

修改 `trajectory_generator.py` 中 `NEUTRAL_STANCE` 的手臂关节值。
