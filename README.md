# Bandit MOPD

Bandit MOPD 是一个面向大语言模型的自适应多教师在线策略蒸馏框架，可视为 MOPD 方法向 on-policy 强化学习训练范式的扩展。框架基于学生模型实时生成的 rollout 评估候选教师，并通过上下文组合 Bandit，在预测效用、探索不确定性、教师多样性与推理成本之间进行权衡，动态选择并加权教师组合。

选定教师的概率分布经融合后形成蒸馏目标，并通过反向 KL 优化持续更新学生模型，Domain覆盖 Math、Code、Science 和 IF 等能力领域。

# 数据集来源

Bandit MOPD 的训练提示统一取自 NVIDIA 发布的 [Nemotron-3-Nano-RL-Training-Blend](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend/tree/ffd169f2b74bb492ec607d64bd56f7435054972b)，并固定在 revision `ffd169f2b74bb492ec607d64bd56f7435054972b`。

| 领域 | 原始数据来源 | 训练集样本 |
| --- | --- | ---: |
| 数学 | [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/tree/65877096c24ffa7abc4e4fa5edb95cf3413a5674)、[Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data/tree/1cdedc52e0e2db85fdf252f9be682e63a5a38c33) | 14,579 |
| 代码 | [Nemotron-RL-coding-competitive_coding](https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding) | 8,552 |
| 科学 | [Nemotron-RL-knowledge-mcqa](https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa) | 15,735 |
| 指令遵循 | [Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following) | 13,261 |
| **合计** | — | **52,127** |

从固定版本中选取上述四个文本领域，对提示进行格式统一和全局去重，使用 seed `42` 按领域划分为约 80% 训练集、10% probe 集和 10% 开发集；最终得到 52,127 条训练样本，以及各 6,513 条 probe 和开发样本。训练仅使用训练集，并进一步过滤渲染后超过 1,024 token 的提示；冻结评测集不参与训练。

数据处理实现见 [`scripts/experiments/prepare_nemotron.py`](scripts/experiments/prepare_nemotron.py) 与 [`bandit_mopd/data.py`](bandit_mopd/data.py)。
# 开源模型选取

本项目使用 [M2RL 开源模型集合](https://huggingface.co/collections/Jackwang111/m2rl) 中的 4B 参数 Qwen3 系列权重。为保持学生模型与领域教师在模型结构和输出空间上的一致性，选择 SFT 模型作为学生初始化，并选取与训练数据四个领域对应的 RL 专家模型组成候选教师池。

| 角色 | 模型 | 用途 |
| --- | --- | --- |
| 学生模型 | [M2RL-SFT](https://huggingface.co/Jackwang111/M2RL-SFT) | 作为 Bandit MOPD 的初始化模型，并在蒸馏过程中更新参数 |
| 数学教师 | [M2RL-RL_Math](https://huggingface.co/Jackwang111/M2RL-RL_Math) | 提供数学领域的策略分布 |
| 代码教师 | [M2RL-RL_Coding](https://huggingface.co/Jackwang111/M2RL-RL_Coding) | 提供代码生成领域的策略分布 |
| 科学教师 | [M2RL-RL_Science](https://huggingface.co/Jackwang111/M2RL-RL_Science) | 提供科学问答领域的策略分布 |
| 指令遵循教师 | [M2RL-RL_IF](https://huggingface.co/Jackwang111/M2RL-RL_IF) | 提供指令遵循领域的策略分布 |

训练期间四个教师模型保持冻结，仅用于评估学生 rollout 并构造加权蒸馏目标；Bandit 根据当前上下文动态选择教师组合，学生模型则通过反向 KL 目标进行更新。
# Benchmark

本项目围绕 Math、Code、Science 和 IF 四个 Domain 选择评测基准。所有 benchmark 均作为冻结的最终评测集，仅用于比较训练前后的模型能力，不参与学生更新、Bandit 反馈或超参数选择。

| Domain | Benchmark | 评测规模 | 评测重点 | 使用范围 |
| --- | --- | ---: | --- | --- |
| Math | [AIME 2024](https://huggingface.co/datasets/math-ai/aime24/tree/83a7f387baaa524a8bda0022eac0541582297103) | 30 | 竞赛数学推理与答案正确率`avg@8` | 当前主评测 |
| Math | [AIME 2025](https://huggingface.co/datasets/math-ai/aime25/tree/563bb8404243c5f09de6ec262f2db674fe5bce9b) | 30 | 竞赛数学推理与答案正确率`avg@8` | 当前主评测 |
| Code | [LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/0fe84c3912ea0c4d4a78037083943e8f0c4dd505) release v5 | 880 | 代码生成及公开、隐藏测试用例通过率`pass@1` | 当前主评测 |
| Code | [LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/0fe84c3912ea0c4d4a78037083943e8f0c4dd505) release v6 | 1,055 | 更完整的累计代码生成评测`pass@1` | 冻结扩展评测 |
| Science | [GPQA-Diamond](https://github.com/idavidrein/gpqa/tree/56686c06f5e19865c153de0fdb11be3890014df7) | 198 | 研究生级物理、化学和生物学选择题推理 | 当前主评测 |
| Science | [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle/tree/5a81a4c7271a2a2a312b9a690f0c2fde837e4c29) | 2,158 | 跨学科高难度知识与推理；使用无图片文本子集 | 冻结扩展评测 |
| IF | [IFEval](https://huggingface.co/datasets/google/IFEval/tree/966cd89545d6b6acfd7638bc708b98261ca58e84) | 541 | 可验证指令约束的严格与宽松遵循率 | 当前主评测 |
| IF | [IFBench](https://huggingface.co/datasets/allenai/IFBench_test/tree/2e8a48de45ff3bf41242f927254ca81b59ca3ae2) | 300 | 多约束指令遵循能力 | 当前主评测 |

# Baseline

当前已有结果如下，所有数值均为百分比：

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">Math</th>
      <th colspan="2">Instruction Following</th>
      <th>Code</th>
      <th>Science</th>
    </tr>
    <tr>
      <th>AIME 2024<br><sub>avg@8</sub></th>
      <th>AIME 2025<br><sub>avg@8</sub></th>
      <th>IFBench<br><sub>Prompt Strict</sub></th>
      <th>IFEval<br><sub>Prompt Strict</sub></th>
      <th>LiveCodeBench v5<br><sub>pass@1</sub></th>
      <th>GPQA-Diamond<br><sub>Accuracy</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Student (SFT-only)</td>
      <td>47.92</td>
      <td>38.33</td>
      <td>48.00</td>
      <td>79.48</td>
      <td>64.66</td>
      <td>39.90</td>
    </tr>
    <tr>
      <td>RL Teacher</td>
      <td><strong>67.50</strong></td>
      <td><strong>64.58</strong></td>
      <td><strong>69.00</strong></td>
      <td><strong>92.24</strong></td>
      <td>71.93</td>
      <td>45.45</td>
    </tr>
    <tr>
      <td>Bandit-MOPD without task_gain(β=1, α=0.5, checkpoint-100)</td>
      <td>57.08</td>
      <td>53.75</td>
      <td>56.33</td>
      <td>82.26</td>
      <td>71.36</td>
      <td>45.45</td>
    </tr>
    <tr>
      <td>Bandit-MOPD with task_gain (β=10, α=0.5; checkpoint-90)</td>
      <td>62.92</td>
      <td>57.50</td>
      <td>54.33</td>
      <td>82.62</td>
      <td>70.34</td>
      <td>48.48</td>
    </tr>
    <tr>
      <td>Bandit-MOPD with task_gain (β=1, α=0.75; checkpoint-100)</td>
      <td>60.42</td>
      <td>48.75</td>
      <td>56.67</td>
      <td>83.36</td>
      <td>67.61</td>
      <td>42.93</td>
    </tr>
    <tr>
      <td>raw-MOPD</td>
      <td>62.50</td>
      <td>59.17</td>
      <td>59.33</td>
      <td>83.73</td>
      <td><strong>73.75</strong></td>
      <td><strong>50.00</strong></td>
    </tr>
    <tr>
      <td>MIX-RL</td>
      <td colspan="6"><em>TODO</em></td>
    </tr>
  </tbody>
</table>

`RL Teacher` 表示每个 Domain 对应的单领域教师：Math 使用 `M2RL-RL_Math`，Code 使用 `M2RL-RL_Coding`，Science 使用 `M2RL-RL_Science`，Instruction Following 使用 `M2RL-RL_IF`，并非同一个教师跨所有 benchmark 的结果。加粗表示当前已完成实验中的最佳成绩。

实验设计了两种reward形式，一种是没有task_gain,一种是加入了task_gain。
当前最接近raw-bandit测评结果的是Bandit-MOPD with task_gain (β=10, α=0.5; checkpoint-90)，但是这个实验的参数β=10，使得参数一边倒，几乎退化成为了单教师。

上述结果采用统一的 16K concise 推理设置：`max_new_tokens=16384`、`temperature=0.6`、`top_p=0.95`、`top_k=20`。

# 主教师＋辅助教师局部修正实验（2026-10-25）

本节记录 `playground` 中最新完成的两轮 200 步实验，区别于前文的四领域组合教师实验。本次只使用 **IF、Math、Code 三个领域**，不训练 Science，也没有评测 GPQA。三组对照均从 `M2RL-SFT` 初始化，采用相同的共同训练设置，并测评各自的 `checkpoint-000200`。

## 实验目的与最新参数

本实验检查：在保留领域主教师监督的基础上，能否选择一位有帮助的辅助教师，通过小幅、局部的分布修正获得互补监督。reward 不使用 `task_gain`、标准答案、任务验证器或独立参考教师。

| 参数 | Naive MOPD k=64 | 上一轮 Bandit | 本轮 Bandit |
| --- | --- | --- | --- |
| 主教师 | 按领域固定 | 按领域固定 | 按领域固定 |
| 辅助教师选择 | 无 | 分领域 LinUCB，辅助教师或不用辅助 | 分领域 LinUCB，辅助教师或不用辅助 |
| 探索系数 `bandit.alpha` | — | 0.3 | **0.03** |
| 冲突惩罚系数 `conflict_weight` | — | 0.1 | **0.02** |
| 学生初始化 | M2RL-SFT | M2RL-SFT | M2RL-SFT |
| 教师池 | M2RL-IF / M2RL-Math / M2RL-Coding | 同左 | 同左 |
| 训练步数 / 有效 batch size | 200 / 2 | 200 / 2 | 200 / 2 |
| 学习率 / weight decay | 2e-6 / 0 | 2e-6 / 0 | 2e-6 / 0 |
| 梯度裁剪 / 精度 | 1.0 / BF16 | 1.0 / BF16 | 1.0 / BF16 |
| Prompt / 回答长度上限 | 1024 / 8192 | 1024 / 8192 | 1024 / 8192 |
| 训练采样 temperature / top-p / top-k | 1.0 / 1.0 / 0 | 1.0 / 1.0 / 0 | 1.0 / 1.0 / 0 |
| 训练 thinking | 关闭 | 关闭 | 关闭 |
| 领域调度 | IF → Math → Code 轮换 | 同左 | 同左 |
| 蒸馏词表支持 | 主教师 top-64 | 主、辅助各 top-64 的去重并集，最多 128 | 同上一轮 |
| 学生 / 教师计算 | GPU / CPU | GPU / CPU | 物理 GPU 1 / CPU |

本轮只相对上一轮修改了探索系数和冲突惩罚系数；聚合规则、reward 其余部分和共同训练参数保持不变。GPU 0 上的首次尝试因其他进程占用显存，在完成 9 步后失败且没有 checkpoint；最终报告使用的是在物理 GPU 1 上从初始模型重新训练的完整 200 步结果，不拼接失败尝试。

## 技术路线（简述）

1. **学生先作答。** 每步取同一领域的两个 prompt，由当前学生生成回答。教师在学生生成的上下文上提供 next-token 分布，构成 on-policy 蒸馏。
2. **主教师固定，Bandit 选择辅助。** Math 数据始终使用 Math 主教师，再选择 IF、Code 或不用辅助。context 包括领域标记、平均回答长度、学生预测熵和训练进度。每个领域、每个动作分别学习线性收益模型，选择分数等于“预测 reward＋探索奖励”。当前 context 不读取题目语义，独立领域模型中的领域 one-hot 也是冗余特征。
3. **每个 token 只允许有限修正。** 监督目标为 `q=(1-alpha)*主教师分布+alpha*辅助教师分布`。alpha 取三个上限的最小值：主教师越不确定，初始预算越大；TV 预算限制整个分布变化；优势保护限制对主教师明确偏好的破坏。alpha 是固定规则计算的，不由 Bandit 直接学习。
4. **更新学生。** 在两位教师 top-64 的并集上计算广义 reverse-KL，使用完整词表归一化后的原始概率，不对子集重新归一化。两条回答各贡献一半 loss，最后只做一次优化器更新。选择不用辅助时，使用主教师 top-64。
5. **测量辅助的增量作用。** 从同一学生与 Adam 状态出发，比较“仅主教师的虚拟一步更新”和“主＋辅助的实际一步更新”。在历史诊断前缀上，检查联合更新是否额外沿辅助独有的分布方向移动；虚拟分支评估后恢复参数，不推进真实优化器状态。
6. **反馈给 Bandit。** reward 为 `B - lambda_C * C - cost`：B 是互补机会乘以实际吸收比例，C 是辅助修正反向于主教师教学方向的程度。本轮 lambda_C=0.02。不用辅助时 reward=0。非测量步使用历史吸收比例估计，再更新所选动作的收益模型。

## 最新最终结果

Macro avg 为 AIME24、AIME25、IFBench、IFEval、LiveCodeBench v5 五项分数的等权算术平均，单位为百分比。

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">Math</th>
      <th colspan="2">Instruction Following</th>
      <th>Code</th>
      <th rowspan="2">Macro avg</th>
    </tr>
    <tr>
      <th>AIME 2024<br><sub>avg@8</sub></th>
      <th>AIME 2025<br><sub>avg@8</sub></th>
      <th>IFBench<br><sub>Prompt Strict</sub></th>
      <th>IFEval<br><sub>Prompt Strict</sub></th>
      <th>LiveCodeBench v5<br><sub>pass@1</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Naive MOPD (fixed top-k=64, batch=2, checkpoint-200)</td>
      <td>66.25</td>
      <td>57.92</td>
      <td><strong>59.67</strong></td>
      <td>85.03</td>
      <td><strong>74.32</strong></td>
      <td>68.64</td>
    </tr>
    <tr>
      <td>Auxiliary Bandit＋Local Correction (探索=0.3, 冲突惩罚=0.1, checkpoint-200)</td>
      <td><strong>67.92</strong></td>
      <td>58.75</td>
      <td>59.00</td>
      <td><strong>85.40</strong></td>
      <td>73.18</td>
      <td><strong>68.85</strong></td>
    </tr>
    <tr>
      <td>Auxiliary Bandit＋Local Correction (探索=0.03, 冲突惩罚=0.02, checkpoint-200；本轮)</td>
      <td>65.42</td>
      <td><strong>62.08</strong></td>
      <td>59.00</td>
      <td>84.84</td>
      <td>71.93</td>
      <td>68.65</td>
    </tr>
  </tbody>
</table>


