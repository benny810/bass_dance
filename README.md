# MIDI-to-Dance

基于 MIDI 文件自动生成机器人下肢舞蹈轨迹，在 MuJoCo 中可视化仿真并同步播放音频。

## 原理

### 整体流水线

```
MIDI 文件 ──→ 音乐特征提取 ──→ PCA 动作生成 ──→ 轨迹合成 ──→ CSV 输出
                                         │
                                  ┌──────┼──────┐
                                  │ 连续载波 (7 PC)  │  ← 不可公度正弦波叠加，永不重复
                                  │ 音乐能量包络      │  ← 音符密度/重音调制幅度
                                  │ 事件重音          │  ← 拍子屈膝、重音跨步、乐句呼吸
                                  └──────────────────┘
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

**`pca_motion.py` 中的 PC 序号：** 默认导出 **按解释方差排序** 的 PC1–PC7；事件重音仍按索引绑定（PC1 屈膝脉冲、PC4 跨步包络等）。若你希望 PC 顺序与注释里的语义槽（深蹲 / yaw / 摇摆 / 跨步 …）对齐，可在提取时加 `--canonical`（内部用匈牙利算法最大化原型与各 PC 的 |cosine|）。

**7 个主成分（默认模型，按方差序）解释的方差与典型负载：**

| PC | 方差 | 累积 | 简要含义 | 典型负载（绝对值较大者） |
|----|------|------|----------|--------------------------|
| PC1 | 54.1% | 54.1% | 左右踝俯仰相反 + 髋俯仰协同（负重前后摇摆） | R/L `ankle_pitch`，R/L `pelvic_pitch` |
| PC2 | 17.0% | 71.0% | 双膝反对称弯曲 + 踝滚（侧向换重心 / 迈步感） | R/L `knee_pitch`，`ankle_roll` |
| PC3 | 13.8% | 84.8% | 双膝同弯 + 踝俯仰（对称深蹲分量） | R/L `knee_pitch`，R/L `ankle_pitch` |
| PC4 | 5.5% | 90.3% | 双侧骨盆偏航同向 + 踝滚（躯干扭转） | R/L `pelvic_yaw`，`ankle_roll` |
| PC5 | 3.5% | 93.8% | 双侧髋俯仰同向 + 踝俯仰（前后俯仰倾斜） | R/L `pelvic_pitch`，`ankle_pitch` |
| PC6 | 2.6% | 96.4% | 双侧髋滚同向 + 膝差（侧倾 / 冠状面摆动） | R/L `pelvic_roll`，`knee_pitch` |
| PC7 | 1.2% | 97.7% | 踝俯仰与膝协同伸展（细小伸展模式） | R/L `ankle_pitch`，`knee_pitch` |

（百分比随数据与 CLI 参数略有浮动；上表对应当前仓库默认生成的 `pca_model.npz`。）

**重建公式：**

```
lower_trajs[t] = mean_pose + Σ_i activation_i[t] × component[i]
```

其中 `mean_pose` 为预处理序列的时间平均姿态（镜像增广后左右对称）；`component[i]` 为第 i 个主成分的 13 维关节协调向量。

### 动作生成架构 (`pca_motion.py`)

采用 **连续载波 + 音乐调制 + 事件重音** 三层架构：

```
                   ┌─────────────────────────────────────────┐
                   │        连续载波 (_continuous_carriers)    │
                   │   每个 PC 由 3 个不可公度正弦波叠加        │
                   │   周期 3~44 秒，频比无理，永不重复          │
                   │   幅值归一化到 [-1, 1]                     │
                   └──────────────┬──────────────────────────┘
                                  │
                   ┌──────────────┴──────────────────────────┐
                   │       音乐能量包络 (_musical_energy)      │
                   │   音符密度 + 重音强度 → 高斯平滑 (2拍)     │
                   │   energy ∈ [0, 1]                         │
                   │   调制公式: envelope = 0.05 + 0.95×E^1.5  │
                   │   安静段落振幅 ~0.1×，激烈段落 ~1.8×       │
                   │   段落间能量对比度约 11:1                    │
                   └──────────────┬──────────────────────────┘
                                  │
                   ┌──────────────┴──────────────────────────┐
                   │          事件重音 (~30% 能量)             │
                   │   PC1: 拍子屈膝脉冲 (指数衰减 τ=0.15s)     │
                   │   PC2: 音高调制 + 小节波 + 乐句脉冲         │
                   │   PC3: 节拍正弦 × 起始强度包络              │
                   │   PC4: 重音低音 ADSR 跨步                  │
                   │   PC5: 音符密度 × 慢速符号振荡              │
                   │   PC6: 音高范围 → 重心偏移 (轨内归一化)     │
                   │   PC7: 乐句级 cos 半周期呼吸弧线             │
                   └──────────────────────────────────────────┘
```

**关键设计决策：**
- **不可公度周期**：7×3=21 个不同周期，频比皆为无理数，整首乐曲内动作不重复
- **Always-on 运动**：基线振幅 5%，保证安静段落也有微动；激烈段落振幅放大 18 倍
- **音乐调制幅度而非触发事件**：音符密度决定动作有多大，而非触发是否动作
- **事件重音叠加**：拍子屈膝、重音跨步等仍在连续运动之上叠加，保持音乐响应性

### MIDI 特征提取

从 MIDI 文件中提取以下音乐特征作为舞蹈动作的驱动信号：

| 特征 | 来源 | 用途 |
|------|------|------|
| **onset_strength** | velocity / 127 | PC1 屈膝幅度, PC3/PC5 调制 |
| **beat_phase** | 拍内位置 (0=downbeat, 0.5=upbeat) | PC1 downbeat 深度区分, PC3 节拍正弦 |
| **accent** | velocity > 均值+0.5σ | PC1/PC4 触发条件, 能量包络 |
| **pitch_level** | 音高归一化 (0~1), 高斯平滑 | PC2/PC6 连续调制 (轨内中心化) |
| **pitch_contour** | pitch_level 的梯度 | 音乐能量调制的辅助信号 |
| **is_low_note** | 低音音符 (≤ 最低音+40%间距) | PC4 跨步触发条件之一 |
| **is_downbeat** | 每小节第一拍 | PC4 跨步触发条件之一 |
| **phrase_boundaries** | onset 间隔 ≥ 2 拍 | PC2 乐句脉冲, PC7 乐句呼吸弧线 |
| **bpm / time_signature** | MIDI tempo 轨道 | 所有周期/节拍计算的基础 |

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
# 基本用法
python -m midi_to_dance.main mid/yellow_bass.mid -o csv/output.csv

# 完整选项
python -m midi_to_dance.main mid/yellow_bass.mid \
    -o csv/output.csv \
    --dt 0.02 \      # 采样周期 (默认 0.02s = 50Hz)
    --scale 1.0 \    # 动作幅度缩放 (默认 1.0, >1 夸张, <1 收敛)
    --stats \        # 打印关节运动统计
    --plot           # 生成 matplotlib 可视化图表
```

### 步骤 3：MuJoCo 可视化仿真

```bash
# 运动学仿真（默认）：直接设定关节角度 + 脚底贴地 + CoM 平衡
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid

# 动力学仿真：PD 位置执行器 + 物理解算
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid --dynamics

# 选项
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid \
    --slow 0.5 \     # 半速播放
    --no-audio \     # 禁用音频
    --dynamics       # 动力学模式 (默认运动学)
```

**仿真模式**：
- **运动学 (kinematic)**：每帧直接设置 `qpos` + 迭代 IK（脚底贴地 + ZMP 平衡）+ XY 锚定防滑步
- **动力学 (dynamics)**：27 个 position 执行器 (`kp=60, kv=4` 腿 / `kp=40, kv=3` 臂腰) + `mj_step`，含重力、接触力等物理效应。ZMP 反馈叠加到髋俯仰执行器

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

至少 28 列：`timestamp` + 27 个关节角；若生成轨迹包含跨步相位，还会追加 `left_foot_step`、`right_foot_step`（由 `trajectory_writer.py` 自动写出）。
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
# 全局缩放
python -m midi_to_dance.main mid/xxx.mid -o csv/xxx.csv --scale 1.5

# 或编辑 pca_motion.py 中 generate_pca_motion() 的参数：
#   base_level    — 安静段最小振幅 (默认 0.05)
#   dynamic_range — 能量调制范围 (默认 0.95)
#   carrier_gain  — 载波增益 (默认 2.0)
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
