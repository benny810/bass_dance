# MIDI-to-Dance

基于 MIDI 文件自动生成机器人下肢舞蹈轨迹，在 MuJoCo 中可视化仿真并同步播放音频。

## 原理

### 整体流水线

```
MIDI 文件 ──→ 音乐特征提取 ──→ 舞蹈基元生成 ──→ 轨迹合成 ──→ CSV 输出
                                         │
                                         ├── bounce  (身体律动)
                                         ├── sway    (左右摇摆)
                                         ├── step    (前后挪步)
                                         └── squat   (半蹲马步)
                                              │
                                    MuJoCo 可视化 + MIDI 音频同步
```

### MIDI 特征提取

从 MIDI 文件中提取以下音乐特征作为舞蹈动作的驱动信号：

| 特征 | 来源 | 用途 |
|------|------|------|
| **onset_grid** | 音符触发时刻 | bounce 的时机 |
| **beat_phase** | 节拍位置 (0.0=downbeat, 0.5=upbeat) | bounce 幅度 (downbeat > upbeat) |
| **accent** | velocity 高于均值+0.5σ | squat 触发条件之一 |
| **pitch_contour** | 音高变化趋势 | sway 方向 (音高上升→右倾，下降→左倾) |
| **is_low_note** | 低音音符 (F2/A2) | squat 触发条件之一 |
| **phrase_boundaries** | onset 间隔 ≥ 2 拍 | step 触发点 |
| **measure_wave** | 小节周期正弦波 | sway 周期摆动 |

### 舞蹈基元

每种基元输出各关节的角度偏移量，最终叠加到中性姿态上。

#### Bounce (身体律动)

```
触发: 每个音符 onset
关节: 双侧 knee_pitch + pelvic_pitch
规则:
  - onset 时刻膝盖弯曲 0.1~0.25 rad，指数衰减 (τ=0.12s)
  - downbeat 幅度 > upbeat 幅度 (×0.6)
  - velocity 调制: 力度越大弯曲越深
  - pelvic_pitch = 0.3 × knee_pitch (前倾补偿)
```

#### Sway (左右摇摆)

```
触发: pitch_contour + 小节周期
关节: 双侧 pelvic_roll + ankle_roll + waist_yaw
规则:
  - 音高上升 → sway 向右 (right pelvic_roll ↑)
  - 音高下降 → sway 向左 (left pelvic_roll ↑)
  - 幅度 ~0.08~0.15 rad
  - 小节正弦波叠加周期性摆动
  - ankle_roll 反向补偿保持脚底贴地
  - waist_yaw 反向扭转增强上半身表现力
```

#### Step (前后挪步)

```
触发: 乐句边界 + downbeat，间隔 ≥ 4 拍
关节: 单侧 pelvic_pitch, pelvic_yaw, knee_pitch, ankle_pitch
规则:
  - 主导腿 knee_pitch +0.18 rad, pelvic_pitch +0.09 rad
  - 左右腿交替迈步
  - 半正弦脉冲 (duration = 0.5 拍)
  - 支撑腿微调以保持平衡
```

#### Squat (半蹲马步)

```
触发: downbeat + accent + low_note 三者同时满足，间隔 ≥ 8 拍
关节: 双侧 knee_pitch + pelvic_pitch
规则:
  - 膝盖深屈 0.4 rad
  - Attack (0.3 拍, 快下) → Hold (3 拍, 保持) → Release (0.5 拍, 缓起)
  - pelvic_pitch 前倾 25% 以维持 COM 平衡
```

### 中性姿态

所有基元叠加在一个预设的中性姿态之上：

| 关节 | 偏移 (rad) | 说明 |
|------|-----------|------|
| knee_pitch | 0.35 | 基线膝盖微曲 (20°) |
| pelvic_roll L/R | ±0.20 | 髋外展，加宽站姿 (11.5°) |
| pelvic_pitch | 0.12 | 身体微前倾补偿膝弯 |
| ankle_roll L/R | ∓0.12 | 脚踝外翻保持脚底贴地 |
| 手臂 (14关节) | 来自 example.csv | Bass 弹奏姿态 |

### 安全约束

- 所有关节 clamp 到 URDF 定义的角度限制
- 高斯平滑 (σ=2 samples ≈ 40ms) 去除突变
- 中性姿态经关节限位校验

## 使用方法

### 环境依赖

```bash
pip install mido numpy scipy mujoco matplotlib
```

### 生成轨迹

```bash
# 基本用法
python -m midi_to_dance.main mid/yellow贝斯.mid -o csv/output.csv

# 完整选项
python -m midi_to_dance.main mid/yellow贝斯.mid \
    -o csv/output.csv \
    --dt 0.02 \      # 采样周期 (默认 0.02s = 50Hz)
    --scale 0.8 \    # 动作幅度缩放 (默认 1.0)
    --stats \        # 打印关节运动统计
    --plot           # 生成 matplotlib 可视化图表
```

### MuJoCo 可视化仿真

```bash
# 基本用法 (同步播放 MIDI 音频)
python -m midi_to_dance.simulate csv/output.csv mid/yellow贝斯.mid

# 选项
python -m midi_to_dance.simulate csv/output.csv mid/yellow贝斯.mid \
    --slow 0.5 \     # 半速播放
    --no-audio       # 禁用音频
```

**仿真交互**：
- 鼠标拖拽：旋转视角
- 滚轮：缩放
- Esc：退出

### 输出格式

CSV 格式，首行为列名，每行一个时间帧（50 Hz）：

```
timestamp,left_leg_pelvic_pitch,...,left_shoulder_pitch,...,right_wrist_roll
0.000000,0.138518,...,0.170745,...,1.073393
0.020000,0.144098,...,0.170745,...,1.073393
...
```

共 28 列（timestamp + 27 joints）：
- 下肢 13 关节：6 左腿 + 6 右腿 + 腰
- 左臂 7 关节：shoulder pitch/roll/yaw, elbow, wrist yaw/pitch/roll
- 右臂 7 关节：同上

## 项目结构

```
midi_to_dance/
├── __init__.py
├── midi_parser.py           # mido 解析 MIDI → NoteEvent/MidiData
├── feature_extractor.py     # 音乐特征提取
├── motion_primitives.py     # 4 种舞蹈基元
├── trajectory_generator.py  # 基元叠加 + 关节限位 + 平滑 + 中性姿态
├── trajectory_writer.py     # CSV 输出
├── main.py                  # CLI 入口 + matplotlib 可视化
└── simulate.py              # MuJoCo 仿真 + MIDI 音频合成播放

csv/
├── output.csv               # 生成的轨迹
└── example.csv              # 参考动捕数据 (提供手臂姿态)

mid/
└── yellow贝斯.mid           # 示例 MIDI 文件

casbot_band_urdf/urdf/
└── CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.urdf  # 机器人模型
```

## 自定义

### 调整动作幅度

```bash
# 全局缩放
python -m midi_to_dance.main mid/xxx.mid -o csv/xxx.csv --scale 1.5

# 或直接编辑 trajectory_generator.py 中的 NEUTRAL_STANCE 字典调整具体参数
```

### 调整舞蹈风格

编辑 `motion_primitives.py` 中各基元的参数：
- `depth`: bounce 深度
- `amplitude`: sway 幅度
- `step_depth`: 迈步幅度
- `squat_depth`: 下蹲深度
- `hold_beats`: 下蹲保持时间

### 替换手臂姿态

修改 `trajectory_generator.py` 中 `NEUTRAL_STANCE` 的手臂关节值，可从 `example.csv` 的其他帧提取。
