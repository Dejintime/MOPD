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
      <td>Bandit-MOPD (checkpoint-100)</td>
      <td>57.08</td>
      <td>53.75</td>
      <td>56.33</td>
      <td>82.26</td>
      <td>71.36</td>
      <td>45.45</td>
    </tr>
    <tr>
      <td>task_gain (β=10, α=0.5; checkpoint-90)</td>
      <td>62.92</td>
      <td>57.50</td>
      <td>54.33</td>
      <td>82.62</td>
      <td>70.34</td>
      <td>48.48</td>
    </tr>
    <tr>
      <td>MOPD</td>
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

Bandit-MOPD `checkpoint-100` 的计数结果为：LiveCodeBench v5 `628/880`、GPQA-Diamond `90/198`、IFEval Prompt Strict `445/541`、IFBench Prompt Strict `169/300`。

上述结果采用统一的 16K concise 推理设置：`max_new_tokens=16384`、`temperature=0.6`、`top_p=0.95`、`top_k=20`。完整结果与来源记录见 [benchmark report](analysis/2026-09-20-benchmark-loss-report/report.md)。
