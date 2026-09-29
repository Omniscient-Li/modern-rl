# modern-rl

个人学习 **动手学强化学习**（[hands-on-modern-rl](https://github.com/walkinglabs/hands-on-modern-rl)）的记录仓库。
每学完一章，把该章的可运行代码和讲解文档归档到这里，方便复习和对照。

## 当前进度

| 章节 | 主题 | 状态 |
| --- | --- | --- |
| Chapter 3 | 马尔可夫决策过程与价值函数 | 已归档 |

从 Chapter 3 开始记录。Chapter 1 是环境准备与 CartPole 入门，不属于本次学习笔记的范围，未收录。

## 目录结构

```
code/chapter03_mdp/        该章的可运行实验
docs/chapter03_mdp/        该章的讲解文档与配图
```

### code/chapter03_mdp

| 文件 | 内容 |
| --- | --- |
| `two_armed_bandit.py` | 两臂老虎机：epsilon-greedy、UCB 等探索策略的比较 |
| `bellman_equation_verify.py` | 用数值方法验证贝尔曼期望方程与最优方程 |
| `gridworld_q_learning.py` | GridWorld 上的价值迭代与 Q-learning |
| `requirements.txt` | 依赖清单 |

### docs/chapter03_mdp

| 文件 | 内容 |
| --- | --- |
| `bandit.md` | 多臂老虎机 |
| `mdp.md` | 马尔可夫决策过程 |
| `policy-value.md` | 策略与价值函数 |
| `value-bellman.md` | 贝尔曼方程 |
| `value-q.md` | 动作价值函数 |
| `value-experiment.md` | 价值函数实验 |
| `dp-mc-td.md` | 动态规划、蒙特卡洛与时序差分 |
| `algorithm-taxonomy.md` | 算法分类 |
| `reward-design.md` | 奖励设计 |

## 运行代码

```bash
pip install -r code/chapter03_mdp/requirements.txt
python code/chapter03_mdp/two_armed_bandit.py
```

`two_armed_bandit.py` 与 `gridworld_q_learning.py` 只依赖 numpy 和 matplotlib，CPU 上即可运行；
`bellman_equation_verify.py` 会用到 torch，装 CPU 版即可。

## 来源与许可

`code/` 与 `docs/` 下的内容整理自 [walkinglabs/hands-on-modern-rl](https://github.com/walkinglabs/hands-on-modern-rl)，
原项目以 **CC BY-NC-SA 4.0** 许可发布，本仓库沿用同一许可，见 [LICENSE](LICENSE)。
转载与再分发请保留署名，并遵守非商业与相同方式共享条款。

文档中的跨章引用（指向 Chapter 7、8、9 等）在本仓库内暂时无法跳转，对应章节归档后会自然生效。