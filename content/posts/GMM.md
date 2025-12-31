---
date: "2025-12-30T11:25:27+08:00"
draft: false
title: "结构估计──GMM法"
toc: true
math: true
slug: GMM
categories:
    - 结构方程模型
tags:
    - 结构方程
    - 前沿方法
---

广义矩估计与模拟矩方法详解

<!--more-->

# 广义矩估计与模拟矩方法详解

---

## 第一章：从 MLE 到 GMM 的转变

### 1.1 为什么需要超越 MLE

前面讲了 NFXP 方法的计算挑战。让我们重新审视这个问题。

**NFXP 面临的核心困境：**

```
参数 θ → 求解动态规划 → 价值函数V(x;θ) → 理论CCP → 似然函数 → 优化θ
                    ↑                                           ↓
                    └─────────────────────────────────────────┘
                             循环嵌套：内层完全求解！
```

每次参数更新，都要**完全求解贝尔曼方程**到收敛。这是计算的瓶颈。

**一个激进的想法：**

何必用似然函数这种"硬约束"？能不能用**更宽松的约束**，比如说"模型应该匹配数据的某些矩（moments）"？

这就是**矩估计**的思想。

### 1.2 矩估计的基本思想

**核心原理很简单：**

模型有参数$\theta$，模型暗示数据应该有某些统计特征（矩）。

如果我们能写出：

$$\mathbb{E}_\theta[\text{某个函数}] = \text{数据中的对应统计量}$$

那么，使等式成立的$\theta$，就是我们要的估计量。

**最简单的例子：**

数据$\{x_1, x_2, \ldots, x_N\}$来自分布$N(\mu, \sigma^2)$。

模型说：$\mathbb{E}_\theta[X] = \mu$。

数据说：样本均值是$\bar{x} = \frac{1}{N}\sum x_i$。

令两者相等：$\mu = \bar{x}$，得到参数估计。

这就是**矩估计**。

### 1.3 从简单矩估计到 GMM

**局限：** 如果参数数量$k$大于矩条件数，就无法精确匹配所有矩。

**扩展：** 使用**多个矩条件**，比如均值、方差、偏度等。

如果有$m$个矩条件和$k$个参数，其中$m > k$（超识别），则：

**无法同时精确满足所有条件**。

**GMM 的解决方案：**

找参数$\theta$，使所有矩条件的**加权平方偏差最小**：

$$\hat{\theta}_{GMM} = \arg\min_\theta \left[ \bar{g}_N(\theta) \right]' W_N \left[ \bar{g}_N(\theta) \right]$$

其中：

-   $\bar{g}_N(\theta) = \frac{1}{N} \sum_{i=1}^N g_i(\theta)$ = 样本矩条件（"矩差距"）
-   $g_i(\theta)$ = 个体$i$的矩条件
-   $W_N$ = 权重矩阵（通常是某个协方差矩阵的逆）

这就是**广义矩估计（GMM）**。

---

## 第二章：GMM 的数学框架

### 2.1 矩条件的定义

设模型由参数$\theta \in \mathbb{R}^k$描述。

定义**矩条件函数**：

$$g(Z_i; \theta): \mathbb{R}^d \times \mathbb{R}^k \to \mathbb{R}^m$$

其中：

-   $Z_i$ = 个体$i$的观测数据（可以是多个变量）
-   $\theta$ = 参数向量
-   $g(\cdot;\theta)$ = 向量函数，有$m$个分量（$m$个矩条件）

**关键性质：** 在真实参数$\theta_0$下，矩条件的期望为零：

$$\mathbb{E}_{\theta_0}[g(Z_i;\theta_0)] = 0_{m \times 1}$$

这是矩条件的**定义性质**。

### 2.2 样本矩条件

给定数据样本$\{Z_1, Z_2, \ldots, Z_N\}$，定义：

$$\bar{g}_N(\theta) = \frac{1}{N} \sum_{i=1}^N g(Z_i;\theta)$$

这是矩条件的**样本对应物**。

**在真实参数处：**

由大数律，当$N \to \infty$时：

$$\bar{g}_N(\theta_0) \xrightarrow{p} \mathbb{E}[g(Z;\theta_0)] = 0$$

即样本矩条件应该接近零。

**在错误参数处：**

如果$\theta \neq \theta_0$，则$\bar{g}_N(\theta)$通常**不接近零**。

### 2.3 GMM 估计量的定义

> 详见[GMM法](/post/GMM/)

**GMM 估计量：**

$$\hat{\theta}_{GMM} = \arg\min_{\theta} J_N(\theta)$$

其中**目标函数**为：

$$J_N(\theta) = \left[ \bar{g}_N(\theta) \right]' W_N \left[ \bar{g}_N(\theta) \right]$$

这是一个二次型，$W_N$是$m \times m$的**权重矩阵**。

**权重矩阵的选择：**

最优的权重矩阵是矩条件的**协方差矩阵的逆**：

$$W_N = \left[ \widehat{Var}(g(Z_i;\theta_0)) \right]^{-1} = S_0^{-1}$$

其中：

$$S_0 = \mathbb{E}[g(Z_i;\theta_0) g(Z_i;\theta_0)']$$

（假设$\mathbb{E}[g(Z_i;\theta_0)] = 0$，则这就是协方差）

**为什么这个权重最优？** 因为它使得不同的矩条件**按其精度比例**被加权——方差大的条件权重小，方差小的条件权重大。

**实践中的权重选择：**

通常分两步：

1. **第一步：** 用恒等权重$W_N^{(1)} = I_m$，得到初步估计$\hat{\theta}_1$
2. **第二步：** 用$\hat{\theta}_1$估计$S_0$，设$W_N^{(2)} = \hat{S}_0^{-1}$，重新优化得到$\hat{\theta}_{GMM}$

这叫**两步 GMM（Two-Step GMM）**。

### 2.4 GMM 的渐近分布

**定理：** 在标准正则性条件下，

$$\sqrt{N}(\hat{\theta}_{GMM} - \theta_0) \xrightarrow{d} N(0, V_{GMM})$$

其中渐近方差为：

$$V_{GMM} = (G' S_0^{-1} G)^{-1}$$

其中：

$$G = \frac{\partial \mathbb{E}[g(Z_i;\theta)]}{\partial \theta'}\Big|_{\theta=\theta_0} = \mathbb{E}\left[\frac{\partial g(Z_i;\theta_0)}{\partial \theta'}\right]$$

是矩条件关于参数的导数矩阵。

**对比 MLE：**

MLE 的渐近方差是：$V_{MLE} = \mathcal{I}(\theta_0)^{-1}$

当矩条件来自似然函数的 score 时，$G = \mathcal{I}(\theta_0)$，所以$V_{GMM} = V_{MLE}$。

但当矩条件不来自似然函数时，$V_{GMM}$通常**比$V_{MLE}$大**（效率更低）。

这是 GMM 相比 MLE 的"成本"——在交换计算便利性的同时，损失了点估计的效率。

### 2.5 过度识别与 J 检验

**定义：**

-   **恰当识别（Just Identified）**：$m = k$（矩条件数 = 参数数）
-   **过度识别（Over-Identified）**：$m > k$
-   **识别不足（Under-Identified）**：$m < k$

**恰当识别的情形：**

如果$m = k$且$\text{rank}(G) = k$，则 GMM 估计量由以下线性系统唯一确定：

$$\bar{g}_N(\hat{\theta}) = 0_m$$

即：使所有矩条件**精确满足**。

**过度识别的情形：**

无法同时精确满足所有矩条件，但可以最小化加权偏差。

此时，$\bar{g}_N(\hat{\theta})$通常**不为零**。

**J 检验（Sargan-Hansen Test）：**

检验模型是否正确的直观想法：

如果模型正确，应有$\bar{g}_N(\hat{\theta}) \approx 0$。

定义**检验统计量**：

$$J_N = N \left[ \bar{g}_N(\hat{\theta}_{GMM}) \right]' W_N \left[ \bar{g}_N(\hat{\theta}_{GMM}) \right]$$

**在模型正确的原假设下**，

$$J_N \xrightarrow{d} \chi^2_{m-k}$$

即自由度为$m-k$（超额识别约束数）的卡方分布。

**使用：** 如果$J_N$的$p$值很小，拒绝原假设，说明模型不太合适。

---

## 第三章：动态模型中的 GMM 应用

### 3.1 动态模型中的矩条件

回到动态离散选择模型。

状态$x_t$，决策$d_t$，由贝尔曼方程驱动：

$$V(x_t) = \max_d \{ u(x_t, d; \theta) + \beta \mathbb{E}_t[V(x_{t+1})] \}$$

**问题：** 如何写出矩条件？

**思路：** 利用模型的**理论含义**。

例如，如果模型正确，那么：

-   特定状态下的平均选择频率应该与模型预测一致
-   状态转移的过程应该与模型的转移函数一致
-   等等

### 3.2 具体矩条件的构造

#### 矩条件 1：选择概率的矩条件

定义：对于状态$x_j$和选择$d_k$，

$$g_{jk}(Z_i; \theta) = \mathbb{1}(x_i = x_j, d_i = d_k) - P_\theta(d_k | x_j)$$

其期望是：

$$\mathbb{E}[g_{jk}(Z_i; \theta_0)] = Pr(x_i = x_j, d_i = d_k) - Pr(d_k = 1 | x_j) \cdot Pr(x_i = x_j) = 0$$

只有在模型正确时才成立。

**释义：** 这说的是，在每个状态，选择的比例应该与模型的 CCP 一致。

#### 矩条件 2：状态转移的矩条件

定义：

$$g_{trans}(Z_i; \theta) = [x_{i,t+1} - \mathbb{E}_\theta(x_{i,t+1} | x_{i,t}, d_{i,t})]$$

期望为零（按定义，条件期望的误差）。

**释义：** 实际的状态转移应该围绕模型预测的转移均值波动。

#### 矩条件 3：贝尔曼差距（Euler Equation 型）

更巧妙的方法：利用贝尔曼方程本身。

定义：

$$g_{bellman}(Z_i; \theta) = d_{i,t} - P_\theta(x_{i,t})$$

**这就是说：** 在每个观测，实际决策与模型预测的选择概率的差，期望为零。

### 3.3 矩条件的直观理解

这些矩条件都有一个共同的思想：

**"模型预测的某个量与数据中的对应量应该匹配"**

比如：

-   预测的 CCP 与实际选择频率匹配
-   预测的状态转移与实际转移匹配
-   预测的决策与实际决策匹配

这与 MLE 的思想完全不同：

-   MLE 说："参数应该最大化观测到这个数据的概率"
-   GMM 说："参数应该使模型的预测与数据的统计特征匹配"

**比喻：**

-   MLE = "我的模型应该'解释'这个数据"（概率意义）
-   GMM = "我的模型应该'拟合'这个数据"（统计特征意义）

---

## 第四章：模拟矩方法（SMM）

### 4.1 为什么需要模拟？

前面讨论的 GMM 中，矩条件$g(Z_i;\theta)$有**闭式表达**。

但在复杂的动态模型中，经常无法写出闭式的矩条件。

**例子：** 假设我们有一个包含两个连续状态变量的模型，状态转移方程很复杂。

我们想说："模型中，给定今年状态$(x_1, x_2)$，明年的期望状态应该是某个值"。

但期望$\mathbb{E}[x'|x]$**没有闭式**，只能通过贝尔曼方程的求解、再加上转移过程的蒙特卡洛抽样才能计算。

**SMM 的思路：** 用**模拟数据**来计算无法用闭式表达的矩。

### 4.2 SMM 的基本框架

**一般思路：**

```
给定参数θ：
  1. 模拟N_sim条虚拟的代理人序列（每个T期）
  2. 对这些虚拟数据计算样本矩
  3. 与实际数据的样本矩比较

优化θ，使模拟矩与实际矩尽可能接近
```

**具体步骤：**

#### 步骤 1：模拟数据生成

给定参数$\theta$和初始状态分布：

```
for s = 1 to N_sim:  （模拟N_sim个代理人）
  for t = 1 to T:    （每个T期）
    x_s,t ~ 初始分布或上期转移分布
    计算CCP: P_s,t = P_θ(x_s,t)
    从伯努利抽样: d_s,t ~ Bernoulli(P_s,t)
    根据转移函数: x_s,t+1 = f(x_s,t, d_s,t, ε_s,t)
           其中ε_s,t ~ 状态转移的随机项
```

结果：得到虚拟的数据集$\{(x^{sim}_s,t, d^{sim}_s,t)\}$

#### 步骤 2：计算模拟矩

对虚拟数据计算样本矩：

$$\bar{g}^{sim}_N(\theta) = \frac{1}{N_{sim} \cdot T} \sum_s \sum_t g(Z^{sim}_{s,t}; \theta)$$

比如，选择概率的矩条件的模拟值是：

$$\bar{g}^{sim}_{jk}(\theta) = \frac{1}{N_{sim} \cdot T} \sum_s \sum_t \mathbb{1}(x^{sim}_{s,t} = x_j, d^{sim}_{s,t} = d_k) - P_\theta(d_k|x_j)$$

#### 步骤 3：计算实际矩

对真实数据计算样本矩：

$$\bar{g}^{real}_N(\theta) = \frac{1}{N} \sum_{i=1}^N g(Z_i; \theta)$$

#### 步骤 4：定义目标函数

$$J_N(\theta) = \left[ \bar{g}^{sim}_N(\theta) - \bar{g}^{real}_N(\theta) \right]' W_N \left[ \bar{g}^{sim}_N(\theta) - \bar{g}^{real}_N(\theta) \right]$$

**SMM 估计量：**

$$\hat{\theta}_{SMM} = \arg\min_\theta J_N(\theta)$$

### 4.3 SMM 与标准 GMM 的关系

**标准 GMM 的目标函数：**

$$J_N^{GMM}(\theta) = \left[ \bar{g}^{real}_N(\theta) \right]' W_N \left[ \bar{g}^{real}_N(\theta) \right]$$

（令样本矩条件尽可能接近零）

**SMM 的目标函数：**

$$J_N^{SMM}(\theta) = \left[ \bar{g}^{sim}_N(\theta) - \bar{g}^{real}_N(\theta) \right]' W_N \left[ \bar{g}^{sim}_N(\theta) - \bar{g}^{real}_N(\theta) \right]$$

（令模拟矩与实际矩相等）

**关键差异：**

在标准 GMM 中，矩条件$g(Z_i;\theta)$的期望在$\theta_0$处为零。

在 SMM 中，**没有这个性质**！矩条件是$(g^{sim} - g^{real})$，这不一定有任何特殊的期望。

这意味着 SMM 的理论**更复杂**，因为目标函数的结构不同。

### 4.4 SMM 的渐近分布

**定理（在适当条件下）：**

$$\sqrt{N}(\hat{\theta}_{SMM} - \theta_0) \xrightarrow{d} N(0, V_{SMM})$$

其中渐近方差涉及：

1. 实际数据矩的方差
2. 模拟误差（来自有限的模拟样本$N_{sim}$）

**关键发现：** SMM 的渐近方差包含两部分：

$$V_{SMM} = V_{\text{sampling}} + V_{\text{simulation}}$$

-   $V_{\text{sampling}}$：来自真实数据的采样误差
-   $V_{\text{simulation}}$：来自模拟数据的蒙特卡洛误差

**权衡：**

如果增加$N_{sim}$（更多模拟），$V_{\text{simulation}}$减小，但计算成本增加。

在实践中，通常选择$N_{sim}$使得$V_{\text{simulation}} \approx V_{\text{sampling}}$（平衡两个误差）。

### 4.5 模拟样本规模的选择

**规则 1：平衡误差**

设真实数据样本$N$，模拟样本$N_{sim}$。

对于矩条件的精度：

-   实际矩的标准误：$O(1/\sqrt{N})$
-   模拟矩的标准误：$O(1/\sqrt{N_{sim}})$

为了平衡，选择$N_{sim} \approx N$。

**规则 2：计算考虑**

如果$N_{sim}$太大，模拟成本会压倒优化成本。

实际中，常见的选择是$N_{sim} = N/10$到$N$。

**规则 3：经验法则**

-   小项目（$N \sim 1000$）：$N_{sim} \approx 500-1000$
-   中等项目（$N \sim 10000$）：$N_{sim} \approx 5000-10000$
-   大项目（$N \sim 100000$）：$N_{sim}$可以相对小一些（如$N_{sim} = N/10$）

---

## 第五章：GMM 与 SMM 的实际应用

### 5.1 矩条件的选择与构造

**关键问题：** 选择哪些矩条件？

**一般原则：**

1. **参数的识别**：矩条件应该能够识别（唯一确定）要估计的参数
2. **数据可计**：矩条件的样本对应物应该能从数据容易计算
3. **信息量**：矩条件应该包含关于参数的足够信息

**常见的矩条件来源：**

#### 来源 1：模型的一阶条件

如果模型来自代理人的最优化（价值函数最大化），则代理人的决策规则（CCP）本身就提供了矩条件。

例如，假设我们能观测到的矩是：

$$m_j = Pr(\text{在状态}x_j \text{选择} d=1 | x_j)$$

模型预测这应该是 CCP：$P_\theta(x_j)$。

矩条件：$m_j = P_\theta(x_j)$

#### 来源 2：结构假设

如果模型对过程进行了特定假设（如状态转移的函数形式），这些假设本身就给出矩条件。

例如，如果模型说状态转移是：

$$x_{t+1} = \rho x_t + \sigma \epsilon_{t+1}, \quad \epsilon \sim N(0,1)$$

则矩条件可以是：

$$\mathbb{E}[x_{t+1} - \rho x_t] = 0$$ $$\mathbb{E}[(x_{t+1} - \rho x_t)^2] = \sigma^2$$

等等。

#### 来源 3：Euler 方程（金融与宏观）

在很多经济学模型中，代理人在时间上优化，导致**Euler 方程**：

$$\mathbb{E}_t\left[\frac{\partial u(c_t)}{\partial c_t} = \beta (1+r) \mathbb{E}_t\left[\frac{\partial u(c_{t+1})}{\partial c_{t+1}}\right]\right] = 0$$

这本身就是一个矩条件。

### 5.2 矩条件数量与参数识别

**规则：**

要估计$k$个参数，至少需要$k$个**线性独立**的矩条件（使得矩条件的 Jacobian 秩为$k$）。

**计数例子：**

假设汽车维修模型中：

-   参数：$\theta = (\text{维护成本系数}, \text{维修成本})$，共 2 个参数

选择矩条件：

1. "里程在 100-110 万的车，维修率应该是...（模型预测的维修率）"
2. "里程在 150-160 万的车，维修率应该是...（模型预测的维修率）"

这两个矩条件可以唯一识别两个参数。

**更好的做法：** 使用多个里程区间（如 5 个）得到 5 个矩条件，形成过度识别。

### 5.3 权重矩阵与两步估计

**第一步：恒等权重**

$$W^{(1)} = I_m$$

得到初步估计：$\hat{\theta}_1$

**第二步：最优权重**

用$\hat{\theta}_1$估计矩条件的协方差：

$$\hat{S} = \frac{1}{N}\sum_i g(Z_i;\hat{\theta}_1) g(Z_i;\hat{\theta}_1)'$$

设权重为：$W^{(2)} = \hat{S}^{-1}$

重新优化得到$\hat{\theta}_{GMM}$

**为什么两步？**

-   第一步得到一个$\sqrt{N}$-相容的估计
-   第二步使用最优权重，提高渐近效率
-   如果直接用最优权重（但没有初步估计），权重矩阵在真实参数处也依赖于$\theta_0$，形成循环

---

## 第六章：GMM 与 SMM 的优势与局限

### 6.1 相比 NFXP/MLE 的优势

| 方面 | NFXP/MLE | GMM/SMM |
| --- | --- | --- |
| **计算复杂性** | 高（需完全求解 DP） | 相对低（无需优化贝尔曼方程） |
| **维数诅咒** | 严重（状态数$N_x^3$） | 缓解（依赖矩条件的选择） |
| **稳健性** | 对模型假设敏感 | 相对稳健（矩条件不用完全刻画分布） |
| **反事实** | 自然支持 | 需要额外工作 |
| **效率** | 最优（充分利用似然信息） | 次优（可能低效） |
| **模型检验** | 间接（收敛性等） | 直接（J 检验） |

### 6.2 GMM 相比 NFXP 的主要优势

#### 优势 1：计算成本显著降低

NFXP 每次评估似然，都需$O(I_{bellman} \times N_x)$的计算来求解贝尔曼方程。

GMM/SMM 不需要完全求解贝尔曼方程！

只需：

-   计算 CCP（从某个来源，可以是非参数估计的）
-   计算矩条件（$O(N_x)$或更低）

**成本对比：**

假设参数维数$k=5$，状态数$N_x=100$，观测$N=10,000$。

-   NFXP：$O(k \times N_x^3 \times N_{iter})$ = $O(5 \times 10^6 \times 500)$ ≈ 25 亿次操作
-   GMM：$O(N \times m + k \times m^2)$ = $O(10,000 \times 10 + 5 \times 100)$ ≈ 100,000 次操作

**差距：** 5 个数量级！

#### 优势 2：维数诅咒缓解

如果模型有 3 个连续状态变量：

-   NFXP：$N_x = 50^3 = 125,000$，梯度计算成本$O(k N_x^3)$不可行
-   GMM：选择$m=20$个矩条件（比$N_x$小得多），成本$O(N \times 20 + k \times 20^2)$仍然可行

#### 优势 3：稳健性更高

NFXP 假设：

-   离散化精度足够
-   数值优化精确
-   私人冲击分布完全正确（通常假设 logit）

GMM 假设：

-   矩条件的期望为零（通常更弱）
-   不需要假设完整的分布

因此，当模型有误设时，GMM 的偏差通常**更小**。

### 6.3 GMM/SMM 的主要局限

#### 局限 1：效率损失

当模型完全正确时，MLE 是**最优渐近效率**的估计量。

GMM 使用的矩条件通常不是似然函数的 score，所以**方差更大**。

**数值例子：**

假设 MLE 的标准误是$\hat{\theta}$的 0.1，GMM 可能是 0.15。

这意味着置信区间更宽，假设检验功效更低。

#### 局限 2：反事实分析困难

GMM 估计的是参数，但没有直接给出完整的模型结构。

如果要做反事实分析（"如果维修成本降低 50%会怎样"），需要：

1. 估计参数
2. 指定完整的模型（成本函数、状态转移等）
3. 求解动态规划

第 2 步不总是直接的。

相比之下，NFXP 估计过程已经完全求解了动态规划，反事实分析是"免费的"。

#### 局限 3：矩条件的选择与检验

**问题 1：** 选择哪些矩条件？

通常有多种可能的选择（里程区间的粗细程度、是否包含二阶矩等），没有唯一的"正确答案"。

结果可能对这个选择敏感。

**问题 2：** 即使用 J 检验，也不知道模型的哪个部分有问题。

J 检验只说"模型不匹配"，但不说是成本函数有误、状态转移有误，还是其他。

#### 局限 4：SMM 的额外复杂性

SMM 引入了模拟误差。如果模拟样本$N_{sim}$太小，模拟不精确，导致目标函数有噪声，优化困难。

需要仔细选择$N_{sim}$来平衡计算成本和精度。

---

## 第七章：GMM 与 SMM 的详细对比

### 7.1 矩条件的三种类型

#### 类型 1：显式矩条件（Explicit Moments）

矩条件有闭式表达，不需要模拟。

**例子：**

$$g_1(Z_i;\theta) = d_i - P_\theta(x_i)$$ $$g_2(Z_i;\theta) = x_{i,t+1} - \mathbb{E}_\theta(x_{i,t+1}|x_{i,t}, d_{i,t})$$

**特点：**

-   计算快（直接）
-   精确（无模拟误差）
-   需要 CCP 或期望值的闭式形式

#### 类型 2：可以部分模拟的矩条件（Partially Simulated）

矩条件的形式明确，但某些期望值需要模拟。

**例子：** 假设成本函数包含一个难以直接计算的项。

用蒙特卡洛近似这个期望，然后代入矩条件。

**特点：**

-   中等复杂度
-   可控制的模拟误差
-   实际应用中常见

#### 类型 3：完全模拟的矩条件（Fully Simulated，SMM）

整个矩条件都通过模拟虚拟数据来计算。

**例子：** 对虚拟数据计算平均维修率，与实际平均维修率比较。

**特点：**

-   非常灵活
-   可处理极复杂的模型
-   计算成本大，且含模拟误差

### 7.2 何时用 GMM vs SMM

**优先使用 GMM 的情形：**

1. 矩条件有闭式表达（至少大部分）
2. CCP 可以从非参数方法直接估计
3. 计算资源有限
4. 模型不太复杂

**必须用 SMM 的情形：**

1. 矩条件没有闭式表达
2. 需要通过完整的动态规划解来定义矩条件
3. 状态维数高（$\geq 3$），离散化不可行
4. 有计算资源（或能并行化）

**混合使用：** 在许多现代应用中，既不是"纯 GMM"也不是"纯 SMM"，而是混合：

-   用非参数 CCP 的矩条件（GMM 的一部分）
-   加上某些需要模拟的补充条件（SMM 的一部分）

---

## 第八章：实际应用案例

### 8.1 案例 1：汽车维修的简化 GMM

**模型：** 简化的一维状态模型（里程）

**参数：** $\theta = (\rho, RC)$

-   $\rho$：维护成本系数
-   $RC$：维修成本

**矩条件：**

将里程区间分成 5 档：

| 里程区间 | 平均维修率（数据） | 模型预测的 CCP |
| --- | --- | --- |
| 0-80 万 | $\bar{d}_1 = 0.05$ | $P_\theta(x_1)$ |
| 80-120 万 | $\bar{d}_2 = 0.15$ | $P_\theta(x_2)$ |
| 120-160 万 | $\bar{d}_3 = 0.30$ | $P_\theta(x_3)$ |
| 160-200 万 | $\bar{d}_4 = 0.50$ | $P_\theta(x_4)$ |
| 200+万 | $\bar{d}_5 = 0.70$ | $P_\theta(x_5)$ |

**矩条件：**

$$g_j(\theta) = \bar{d}_j - P_\theta(x_j), \quad j=1,\ldots,5$$

**GMM 目标函数：**

$$\hat{\theta}_{GMM} = \arg\min_\theta \sum_{j=1}^5 g_j(\theta)^2$$

（用恒等权重，简化表达）

**为什么这行得通？**

-   如果$\rho$太大（维护成本太贵），模型预测所有维修率都会高
-   如果$RC$太小（维修成本太便宜），模型预测即使在低里程也会维修
-   ==调整$\rho$和$RC$，使 5 个预测的维修率都与数据匹配==

**计算成本：**

与 NFXP 相比：

-   不需要求解贝尔曼方程（计算 CCP 的方式待定）
-   每次评估目标函数，只需计算 5 个矩条件
-   可能优化只需几十次迭代

**总成本：** 秒级（对比 NFXP 的分钟级）

### 8.2 案例 2：多维状态的 SMM

**模型：** 设备替换决策，有两个状态变量：

-   $x_1$：主设备年龄
-   $x_2$：附加设备年龄

**参数：** $\theta = (\theta_1, \theta_2, \theta_3)$

**矩条件（需要模拟）：**

1. "不同$(x_1, x_2)$的替换率应该与模型预测一致"
2. "主设备年龄的自相关系数应该是..."（如果有转移过程）
3. "设备年龄的联合分布应该...（某些分位数）"

**为什么需要 SMM？**

-   完全离散化：$50 \times 50$状态，贝尔曼方程很大
-   NFXP 计算成本过高
-   SMM 中，只需模拟虚拟数据，计算对应矩，无需完全求解 DP

**步骤：**

```
对于每个参数候选θ：
  模拟10000个虚拟企业，每个运营20年
  对虚拟数据计算：
    - 各(x1,x2)组合的替换率
    - 年龄的自相关
    - 其他统计量
  与真实数据比较
  计算目标函数J(θ)

优化θ最小化J(θ)
```

---

## 第九章：GMM 与 SMM 的计算实现

### 9.1 矩条件的精确计算

假设已经估计了 CCP（从非参数方法或其他）。

**计算矩条件的步骤：**

#### 对于显式矩条件（GMM）

**矩条件：** $g_j(\theta) = \bar{d}_j - P_\theta(x_j)$

**计算：**

1. 从数据中统计：在状态$x_j$附近，维修的比例$\bar{d}_j$
2. 根据参数$\theta$，计算模型预测的 CCP：$P_\theta(x_j)$
3. 差值即为矩条件

**代码框架（伪代码）：**

```
function compute_moments_GMM(theta, data, CCP)
  moments = []
  for j in 1:num_states
    # 数据中的维修率
    data_mean = mean(data[state==j].repair)

    # 模型预测的CCP
    model_pred = CCP[j](theta)

    # 矩条件
    moment = data_mean - model_pred
    moments.append(moment)

  return moments
```

**计算复杂度：** O($N + N_x$)，非常快。

#### 对于模拟矩条件（SMM）

**步骤 1：参数确定下的模拟**

```
function simulate_data(theta, N_sim, T_periods)
  for i in 1:N_sim
    x[i,1] = 初始状态
    for t in 1:T_periods
      P = CCP(x[i,t], theta)  # 根据θ计算CCP
      d[i,t] ~ Bernoulli(P)
      x[i,t+1] = transition(x[i,t], d[i,t], noise)

  return {x, d}
```

**步骤 2：计算模拟矩**

对虚拟数据$\{x^{sim}, d^{sim}\}$计算样本矩，与实际数据的样本矩比较。

```
function compute_moments_SMM(theta, real_data, N_sim)
  sim_data = simulate_data(theta, N_sim, T)

  # 实际矩
  real_moments = compute_statistics(real_data)

  # 模拟矩
  sim_moments = compute_statistics(sim_data)

  # 差异
  diff = sim_moments - real_moments

  return diff
```

**计算复杂度：** O($N_{sim} \times T + N \times m$)

如果$N_{sim} \times T$很大，这可能成为瓶颈。

### 9.2 目标函数的优化

**目标函数：**

$$J(\theta) = [g(\theta)]' W [g(\theta)]$$

其中$g(\theta)$是矩条件（向量），$W$是权重矩阵。

**一阶条件：**

$$\frac{\partial J}{\partial \theta} = 2 G(\theta)' W [g(\theta)] = 0$$

其中$G(\theta) = \partial g / \partial \theta'$。

**优化算法：**

#### 方法 1：梯度下降

```
theta_0 = 初始值
for iter in 1:max_iter
  g = compute_moments(theta_0)
  G = numerical_gradient(compute_moments, theta_0)

  grad = 2 * G' * W * g

  # 线搜索
  step_size = line_search(J, theta_0, -grad)

  theta_1 = theta_0 - step_size * grad

  if ||theta_1 - theta_0|| < tol:
    break

  theta_0 = theta_1

return theta_0
```

**优点：** 简单

**缺点：** 收敛慢，特别是在接近最优时

#### 方法 2：拟牛顿法（BFGS）

```
theta_0 = 初始值
B_0 = I  # 初始的Hessian近似

for iter in 1:max_iter
  g = compute_moments(theta_0)
  G = numerical_gradient(compute_moments, theta_0)

  grad = 2 * G' * W * g

  # 搜索方向
  direction = -inv(B) * grad

  # 线搜索
  step_size = line_search(J, theta_0, direction)

  theta_1 = theta_0 + step_size * direction

  # 更新Hessian近似
  B_1 = update_B_BFGS(B_0, theta_1 - theta_0, grad_1 - grad_0)

  if ||theta_1 - theta_0|| < tol:
    break

  theta_0 = theta_1
  B_0 = B_1

return theta_0
```

**优点：** 快速收敛，适度计算

**缺点：** 需要更新 Hessian 近似

#### 方法 3：数值梯度的计算

由于矩条件的形式复杂，有限差分通常是可行的：

$$\frac{\partial g_j}{\partial \theta_k} \approx \frac{g_j(\theta + \epsilon e_k) - g_j(\theta - \epsilon e_k)}{2\epsilon}$$

对$m$个矩条件和$k$个参数，需要$2k$次矩条件评估。

**成本权衡：**

-   如果矩条件评估很快（GMM），数值梯度可行
-   如果矩条件评估很慢（SMM），需要考虑解析梯度或其他技巧

### 9.3 收敛诊断

**诊断 1：梯度的大小**

在最优点处，梯度应该接近零。

```
grad_norm = ||∂J/∂θ||
if grad_norm > 1e-4:
  警告：可能未收敛
```

**诊断 2：参数变化**

连续迭代的参数变化应该减小：

```
param_change = ||theta_k - theta_{k-1}||
if param_change 在振荡：
  可能步长太大，或函数有多个局部最小值
```

**诊断 3：目标函数值**

```
if J 在减少但减速：
  正常的收敛行为
if J 突然增加：
  可能数值问题，或步长过大
```

**诊断 4：J 检验**

在最优$\hat{\theta}$处，计算：

$$J_N = N [g(\hat{\theta})]' W [g(\hat{\theta})]$$

检查其是否来自$\chi^2_{m-k}$分布。

```
p_value = 1 - cdf_chi2(J_N, m-k)
if p_value < 0.05:
  模型可能有误设
```

---

## 第十章：GMM 与 SMM 的高级话题

### 10.1 Continuous Updating GMM

前面讨论的两步 GMM 有个缺陷：

第一步用恒等权重$I_m$估计的$\hat{\theta}_1$可能不在目标函数$J^{(2)}(\theta)$的临界点上。

**连续更新 GMM（Continuous Updating Estimator，CUE）：**

直接优化：

$$\hat{\theta}_{CUE} = \arg\min_\theta [g_N(\theta)]' [\widehat{Var}(g_N(\theta))]^{-1} [g_N(\theta)]$$

其中权重矩阵$W(\theta) = [\widehat{Var}(g_N(\theta))]^{-1}$也随$\theta$变化。

**优点：** 渐近一阶有效，不需要两步

**缺点：** 优化困难（目标函数非光滑），计算成本高

### 10.2 多目标函数的处理

有时候有多个"竞争的"目标，比如：

-   拟合 CCP
-   拟合状态分布
-   拟合某个转移过程

可以构造多个矩条件集合，组合成一个总的目标函数：

$$J(\theta) = \sum_k \lambda_k J_k(\theta)$$

其中$\lambda_k$是权重（可以反映不同目标的相对重要性）。

**关键问题：** $\lambda_k$如何选择？

通常：

-   从先验知识或文献
-   敏感性分析：显示结果对$\lambda$的依赖程度
-   后处理的不确定性量化

### 10.3 蒙特卡洛误差的修正

在 SMM 中，模拟引入了额外的噪声。

这导致估计量的方差比标准 GMM 大。

**修正方法 1：增加模拟样本**

选择$N_{sim}$足够大使$V_{simulation} \ll V_{sampling}$。

**修正方法 2：使用近似方差公式**

GMM 方差公式适应改为：

$$V_{SMM} = V_{sampling} \times \left(1 + \frac{N}{N_{sim} \times T}\right)$$

这个"修正因子"说明了模拟如何增加方差。

**修正方法 3：Antithetic Variates 与其他方差减少技巧**

蒙特卡洛模拟中，有多种技巧可减少噪声：

-   对偶变量（Antithetic Variates）
-   控制变量（Control Variates）
-   分层抽样（Stratified Sampling）

使用这些可显著减少$N_{sim}$的需求。

### 10.4 动态面板模型的 GMM（Arellano-Bond）

在时间序列数据上应用 GMM 时，常面临**动态内生性**问题。

**Arellano-Bond 估计器：** 使用滞后变量作为工具变量

$$g_i(Z_i; \theta) = [\text{模型残差}] \times [\text{滞后值}]$$

这利用了"滞后值与当期误差正交"的假设。

虽然经典动态面板文献不直接关于动态离散选择，但相同思想可应用：用过去的决策状态作为矩条件。

---

## 第十一章：GMM 与 NFXP 的深入对比

### 11.1 识别论的视角

两种方法在**参数识别**上的差异。

**NFXP（完全信息最大似然）：**

-   依赖完整的模型 specification
-   CCP 的形状完全由参数$\theta$确定
-   参数识别通过似然函数的形状（一般很强）

**GMM（矩方程）：**

-   只依赖选定的矩条件
-   参数识别通过矩条件的独立性（可能较弱）
-   可能存在多个$\theta$给出相同的矩

**识别强度对比：**

| 情形 | NFXP 识别 | GMM 识别 |
| --- | --- | --- |
| 矩条件数 < 参数数 | 无（无法估计） | 无（识别不足） |
| 矩条件数 = 参数数 | 通常强 | 取决于条件的独立性 |
| 矩条件数 > 参数数 | 非常强 | 中等（取决于冗余性） |

### 11.2 误设（Misspecification）下的表现

**场景：** 假设真实模型有某个特征（如不同的成本函数形式），但估计时用错了假设。

**NFXP 的反应：**

-   所有参数估计都会有**系统性偏差**
-   偏差的方向和大小难以预测
-   反事实分析会继承这个偏差

**GMM 的反应：**

-   只有参与矩条件的参数会有明显偏差
-   其他参数可能"幸存"
-   通过 J 检验可以（有时）检测误设

**例子：**

假设维修成本实际上不是常数$RC$，而是随里程变化的：$RC(x) = RC_0 + RC_1 \times x$。

-   NFXP 用常数 RC 估计，会给出有偏的参数（无法检测到这个误设）
-   GMM 如果矩条件包含"不同里程的维修率"，J 检验会显示不拟合，可能提示有问题

### 11.3 样本量的影响

**小样本表现：**

NFXP：

-   最大似然估计在小样本中仍然相对稳健
-   但数值优化可能困难（噪声大）

GMM：

-   样本矩条件的波动大，目标函数有噪声
-   但仍可用（矩估计的性质）

**大样本效率：**

NFXP：

-   渐近有效，标准误随$N^{-1/2}$缩放（最优）

GMM：

-   非完全有效，标准误可能更大
-   但在维数高时，渐近优于 NFXP（计算考虑）

---

## 第十二章：实用指南与决策树

### 12.1 选择方法的决策流程

```
问题分析：参数数，状态维数，计算资源

    ├─ 计算资源充足？计算成本不是瓶颈？
    │  ├─ 是 → NFXP（最优效率）
    │  └─ 否 → 考虑GMM/SMM
    │
    ├─ 完整的模型specification已知且相信？
    │  ├─ 是 → NFXP or GMM都可
    │  └─ 否 → GMM（更稳健）
    │
    ├─ 需要反事实分析？
    │  ├─ 是 → NFXP（自然支持）
    │  └─ 否 → GMM足够
    │
    ├─ 状态维数？
    │  ├─ 1-2维 → NFXP or GMM都可
    │  ├─ 3维 → NFXP困难，用GMM/SMM
    │  └─ 4+维 → SMM（或特殊结构）
    │
    └─ 是否需要进行假设检验（J检验）？
       ├─ 是 → GMM（直接）
       └─ 否 → NFXP也可
```

### 12.2 GMM/SMM 的实施检查清单

在正式估计前，检查以下事项：

-   [ ] **矩条件清晰** - 能否用一句话描述每个矩条件的经济含义？
-   [ ] **参数识别** - 每个参数是否由某个矩条件唯一识别？
-   [ ] **数据可计** - 是否能从数据容易计算所有矩条件的样本对应物？
-   [ ] **初值合理** - 初始参数值是否在可信范围内？
-   [ ] **样本大小** - $N$是否足够使矩条件的方差可估计？
-   [ ] **模拟规模**（如用 SMM）- $N_{sim}$与$N$的比例是否合理？
-   [ ] **权重矩阵** - 是否考虑两步估计或 CUE？
-   [ ] **收敛诊断** - 是否检查梯度、参数变化、目标函数趋势？
-   [ ] **敏感性分析** - 结果对矩条件选择、初值、权重的依赖？
-   [ ] **外部验证** - 估计的参数是否与现有文献、行业知识一致？

### 12.3 实际代码框架（概念伪代码）

```python
# GMM估计的整体框架

class GMMEstimator:
  def __init__(self, data, moment_functions, param_dim):
    self.data = data
    self.g = moment_functions  # 矩条件函数列表
    self.k = param_dim

  def compute_moments(self, theta):
    """计算样本矩条件"""
    moments = []
    for g_i in self.g:
      m = mean([g_i(z, theta) for z in self.data])
      moments.append(m)
    return np.array(moments)

  def compute_var_matrix(self, theta):
    """估计矩条件的协方差矩阵"""
    gvals = np.array([
      [g_i(z, theta) for g_i in self.g]
      for z in self.data
    ])
    return gvals.T @ gvals / len(self.data)

  def objective(self, theta, W=None):
    """GMM目标函数"""
    g = self.compute_moments(theta)
    if W is None:
      W = np.eye(len(g))
    return g @ W @ g

  def estimate_twostep(self, theta_init):
    """两步GMM估计"""
    # 第一步：恒等权重
    res1 = minimize(
      lambda t: self.objective(t, W=np.eye(len(self.g))),
      theta_init
    )
    theta1 = res1.x

    # 第二步：最优权重
    S = self.compute_var_matrix(theta1)
    W2 = np.linalg.inv(S)

    res2 = minimize(
      lambda t: self.objective(t, W=W2),
      theta1
    )
    theta_hat = res2.x

    return theta_hat

  def j_test(self, theta_hat):
    """Sargan-Hansen J检验"""
    g = self.compute_moments(theta_hat)
    S = self.compute_var_matrix(theta_hat)
    W = np.linalg.inv(S)

    J = len(self.data) * (g @ W @ g)
    df = len(self.g) - self.k  # 超额识别自由度
    p_value = 1 - chi2.cdf(J, df)

    return J, df, p_value

  def bootstrap_se(self, theta_hat, nboot=100):
    """Bootstrap标准误"""
    ses = np.zeros(self.k)
    thetas = []

    for b in range(nboot):
      idx = np.random.choice(len(self.data), len(self.data))
      boot_data = [self.data[i] for i in idx]

      # 用bootstrap样本重新估计
      ... # 用self.data = boot_data重新估计

      thetas.append(theta_b)

    return np.std(thetas, axis=0)
```

---

## 第十三章：GMM 与 SMM 的文献进展

### 13.1 重要里程碑

**GMM 的起源：**

-   1982: Hansen - "GMM 理论与应用"（Econometrica）
-   成为计量经济学标准方法

**动态离散选择中的应用：**

-   1994: Pakes & Olley - 用 GMM 估计动态模型
-   2000s: Bajari, Benkard, Levin 等 - SMM 在复杂模型中的应用

**与 NFXP 的对比：**

-   2000: Aguirregabiria & Mira - CCP 方法，介于 GMM 和 NFXP 之间
-   2010: 综述性论文，总结三种方法的优劣

### 13.2 当代应用的趋势

**趋势 1：混合方法**

-   不是"纯粹"的 GMM 或 SMM
-   而是结合两者的优势：用非参数 CCP 的简单矩条件+少量模拟矩条件

**趋势 2：高维动态模型**

-   面对 3 维或更高的状态空间
-   SMM 成为必选（NFXP 不可行）

**趋势 3：贝叶斯方法**

-   不仅是频率派的 GMM
-   贝叶斯 GMM（用 MCMC）也越来越常见

**趋势 4：机器学习融合**

-   用神经网络等学习 CCP 或 policy 函数
-   再用 GMM 估计参数（避免求解 DP）

---

## 第十四章：总结与对比表

### 14.1 三种方法的综合对比

| 方面 | NFXP | CCP 半参数 | GMM/SMM |
| --- | --- | --- | --- |
| **计算难度** | 高 | 中 | 低 |
| **所需的 DP 求解** | 完整 | 部分反演 | 无 |
| **维数诅咒影响** | 严重 | 中等 | 轻微 |
| **参数效率** | 最优 | 次优 | 次优 |
| **误设下表现** | 有偏 | 相对稳健 | 相对稳健 |
| **反事实支持** | 自然 | 需额外工作 | 需额外工作 |
| **模型检验** | 间接 | 通过 CCP 比较 | 直接（J 检验） |
| **适用问题范围** | 中等 | 广 | 最广 |
| **学习曲线** | 陡 | 中 | 平缓 |

### 14.2 选择建议速查表

```
低维（状态≤2维），小参数（k≤5）
└─ 计算充足 → NFXP
└─ 计算有限 → CCP or GMM

中维（状态3维），中等参数（5≤k≤10）
└─ 需反事实 → NFXP（如可行）or CCP
└─ 只需参数 → GMM/SMM

高维（状态≥4维），多参数（k>10）
└─ SMM is necessary (NFXP infeasible)

模型不确定
└─ GMM/SMM （稳健性）

需要假设检验
└─ GMM/SMM （J检验）

时间紧张
└─ GMM/SMM （快速粗略估计）
     或 CCP（折衷）
```

### 14.3 各方法的计算复杂性（Big-O 记号）

假设参数$k$个，状态$N_x$个，观测$N$个。

| 方法 | 单次评估成本 | 优化迭代数 | 总成本 |
| --- | --- | --- | --- |
| NFXP | $O(k N_x^3)$或$O(k N_x^2)$ | 500-1000 | $O(k N_x^3 \times 1000)$ |
| CCP | $O(N + N_x^2)$ | 300-500 | $O(k N_x^2 \times 500)$ |
| GMM | $O(N \times m)$ | 50-200 | $O(N \times m \times 200)$ |
| SMM | $O(N_{sim} \times T + N \times m)$ | 100-500 | 取决于$N_{sim}$ |

其中$m$是矩条件数（通常$m \ll N_x$）。

---

## 第十五章：高级应用与拓展

### 15.1 不完全信息下的 GMM

真实经济中，代理人的信息可能不完全。比如，一个公司不知道竞争对手未来的策略。

**处理方法：**

-   用**信念函数**（belief function）代替完全前瞻的 CCP
-   GMM 中的矩条件变成"给定其信念的最优决策应该与观测一致"
-   同时估计参数和信念的演化

这导致更复杂的 GMM 问题，但计算上仍比 NFXP 可行。

### 15.2 异质性与混合分布

如果代理人有未观测的异质性（比如，不同驾驶员对可靠性的价值评估不同）。

**处理方法：**

-   用混合分布模型（Mixture Models）
-   GMM 中增加矩条件来刻画分布的特征（如二阶矩）
-   SMM 中在模拟时对不同类型的代理人分别抽样

### 15.3 多代理人博弈（产业组织）

如果决策不仅由自己的状态驱动，还由竞争对手的决策驱动（Markov Perfect Equilibrium）。

**计算挑战：** 求解均衡远比单代理问题复杂

**GMM/SMM 的优势：**

-   可以避免显式求解均衡
-   直接用矩条件"均衡 action 应该与 CCP 一致"

**例子：** 进退市决策受竞争对手进退决策影响

-   NFXP 需要求解所有可能的均衡
-   GMM 可直接用矩条件估计

---

## 第十六章：最终总结

### 16.1 三种估计方法的精髓

**NFXP（嵌套固定点）** $$\text{核心：} \max_\theta \ell(\theta) = \max_\theta \sum_i \log P_\theta(d_i | x_i)$$ $$\text{其中} P_\theta = \text{由完整求解的动态规划确定}$$

-   充分利用模型结构
-   渐近最优效率
-   计算密集

**CCP 半参数方法** $$\text{核心：} \text{非参数估计CCP} \to \text{反演价值函数} \to \text{估计参数}$$

-   折衷方案
-   相对快速且稳健
-   理论相对复杂

**GMM/SMM（矩方法）** $$\text{核心：} \min_\theta ||g^{sim}(\theta) - g^{real}(\theta)||_W^2$$ $$\text{其中} g = \text{选定的统计矩}$$

-   最灵活，适用范围最广
-   计算相对快（特别高维）
-   效率可能次优

### 16.2 实践指导原则

1. **从简单开始**

    - 先用非参数方法理解数据
    - 然后逐步增加模型复杂度

2. **多方法验证**

    - 不要仅用一种方法
    - 如果条件允许，用多种方法估计，比较结果
    - 差异很大时，说明有问题（模型、实施还是数据）

3. **充分诊断**

    - 检验模型假设（用 J 检验等）
    - 敏感性分析（对初值、矩条件选择、权重等）
    - Bootstrap 或其他方法估计标准误

4. **透明报告**

    - 清晰说明矩条件定义
    - 报告诊断统计量
    - 讨论局限与假设

5. **质重于速**
    - 花时间在问题理解和模型设定上
    - 不要盲目追求最快的估计方法
    - 一个可信的参数估计比快速的粗糙估计值钱得多

### 16.3 学习路径与参考资源

**基础阶段：** 理论与基本应用

-   教科书：Cameron & Trivedi《Microeconometrics》（GMM 章节）
-   论文：Hansen 1982（开创性）

**进阶阶段：** 动态模型

-   论文：Rust 1987（NFXP）
-   论文：Aguirregabiria & Mira 2010（综述）

**高级阶段：** 应用与拓展

-   产业组织应用：Pakes & McGuire 系列
-   复杂模型：Nevo & Whinston 等

**实现阶段：** 编程与实验

-   从简单的玩具模型开始
-   逐步增加复杂度
-   多用蒙特卡洛模拟验证（你的估计能否恢复已知参数）

---

## 最后：快速参考卡

```
┌─────────────────────────────────────────────┐
│        方法选择的快速决策树                 │
├─────────────────────────────────────────────┤
│                                             │
│  状态维数？                                 │
│  ├─ 1维:  NFXP or GMM                      │
│  ├─ 2维:  NFXP or GMM or CCP               │
│  ├─ 3维:  GMM/SMM (NFXP hard)              │
│  └─ 4+:   SMM only                         │
│                                             │
│  计算资源？                                 │
│  ├─ 充足:   NFXP (best efficiency)        │
│  ├─ 中等:   CCP (compromise)              │
│  └─ 有限:   GMM/SMM (fast)                │
│                                             │
│  反事实需求？                              │
│  ├─ 是:     prefer NFXP                    │
│  └─ 否:     GMM/SMM 足够                   │
│                                             │
│  模型确定性？                              │
│  ├─ 高:     NFXP (gain efficiency)        │
│  └─ 低:     GMM/SMM (more robust)         │
│                                             │
└─────────────────────────────────────────────┘
```

**三种方法的"黄金法则"**：

1. **如果能用 NFXP，就用它**（最优效率）
2. **如果 NFXP 太慢，用 CCP**（折衷）
3. **如果 CCP 都不行，用 GMM/SMM**（最后手段，但往往足够好）
