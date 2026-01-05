---
date: "2026-01-04"
draft: false
title: "CO2 emission regulation and generation allocation with heterogeneous coal-fired generators"
toc: true
math: true
weight:
categories:
    - 论文解析
tags:
    - 结构方程估计
---

<!--more-->

# 模型深度解析

**论文：** CO2 emission regulation and generation allocation with heterogeneous coal-fired generators **领域：** Environmental Economics / Structural IO (Production Function Estimation)

## 1. 核心变量定义 (Notation)

在进入推导前，我们要明确下标和变量的含义，这对于理解面板数据结构至关重要。

-   **下标：**
    -   $j$: 发电机组 (Generator)，这是本文的最小分析单位（区别于常见的 Plant-level）。
    -   $i$: 电厂 (Plant)，一个电厂可能包含多个机组。
    -   $t$: 年份 (Year)。
    -   $g(j)$: 机组 $j$ 的容量类型 (Capacity type)，分为 $S$ (Small), $M$ (Medium), $L$ (Large)。
-   **核心变量：**
    -   $Q_{ijt}$: 发电量 (Output, electricity generation)。
    -   $H_{ijt}$: 热能输入 (Heat input)，主要来自于煤炭燃烧。
    -   $K_j$: 装机容量 (Capacity)，这是资本存量，短期内固定。
    -   $\beta_{h, g(j)}$: 规模报酬系数 (Returns to scale)，允许不同类型的机组有不同的规模报酬。
    -   $\omega_{ijt}$: **不可观测的生产率 (Unobserved Productivity)**。这是结构化估计的核心，代表技术效率，厂商知道但计量经济学家不知道。
    -   $\epsilon_{ijt}$: 测量误差或由于不可预见冲击导致的误差。

## 2. 结构原语与模型构建 (Structural Primitives)

### A. 生产函数设定 (Leontief Technology)

作者没有使用常见的 Cobb-Douglas，而是使用了 **Leontief (定比)** 生产函数形式，这在电力行业文献中很常见（如 Fabrizio et al., 2007），因为资本（发电机组）和燃料（热能）之间的替代性极低。

$$Q_{ijt} = \min \{ \underbrace{e^{\beta_{0g(j)} + \omega_{ijt}} H_{ijt}^{\beta_{h, g(j)}}}_{\text{燃料约束部分}}, \quad \underbrace{t_0 K_j}_{\text{物理容量约束}} \}$$

-   **经济学直觉：** 发电量取决于热能输入转换效率，但上限不能超过物理装机容量（一年 8760 小时 $t_0$ 满负荷运转）。
-   **理性厂商假设：** 理性的电厂不会投入超过容量限制的热能（那是浪费）。因此，厂商会选择 $H_{ijt}$ 使得燃料部分正好等于产出（在未达到容量上限时）。

### B. 逆向推导：输入需求函数

根据上述假设，我们可以写出有效率的产出方程：

$$Q_{ijt} = e^{\beta_{0g(j)} + \omega_{ijt}} H_{ijt}^{\beta_{h, g(j)}}$$

对上述公式取对数（Log-linearization）：

$$q_{ijt} = \beta_{0g(j)} + \beta_{h, g(j)} h_{ijt} + \omega_{ijt}$$

_(注：小写字母代表对数值)_

这就是我们需要估计的基础方程（Equation 3）。但这里存在严重的计量问题。

## 3. 识别策略 (Identification Strategy)

### A. 内生性问题 (The Simultaneity Bias)

如果我们直接对 $q_{ijt} = \beta_0 + \beta_h h_{ijt} + \omega_{ijt} + \epsilon_{ijt}$ 进行 OLS 回归，结果是有偏的。

-   **原因：** $\omega_{ijt}$ (生产率) 对厂商是已知的。当 $\omega_{ijt}$ 较高时（效率高），厂商的最优决策往往是增加投入 $h_{ijt}$（或者根据需求调整）。
-   **后果：** $Corr(h_{ijt}, \omega_{ijt}) \neq 0$。OLS 假设解释变量与误差项不相关，这一假设被打破。通常会导致 $\beta_h$ 的估计值**向上偏误 (Upward Bias)**。

### B. 控制函数法 (Control Function Approach)

作者采用了 **Proxy Variable (代理变量)** 方法来解决这个问题。这属于 Olley-Pakes (1996), Levinsohn-Petrin (2003), Ackerberg-Caves-Frazer (2015) 的文献流派。

-   **代理变量：** 厂用电量 (Auxiliary electricity consumption, $e_{ijt}$)。

-   **单调性假设 (Monotonicity Assumption)：** 假设对于给定的热能输入 $h$，生产率 $\omega$ 越高，所需的辅助用电可能越低（效率更高），或者存在某种单调函数关系：

    $$e_{ijt} = f(h_{ijt}, \omega_{ijt})$$

-   **反函数技巧 (Inversion)：** 只要 $f$ 关于 $\omega$ 是严格单调的，我们要以把 $\omega$ 反解出来：

    $$\omega_{ijt} = f^{-1}(e_{ijt}, h_{ijt})$$

这一步非常关键，它把不可观测的 $\omega_{ijt}$ 转化为了两个可观测变量 $e_{ijt}$ and $h_{ijt}$ 的非参数函数。

## 4. 估计过程 (Estimation Routine)

估计分为两个阶段 (Two-Stage Estimation)：

### 第一阶段：剔除测量误差

将反函数代入生产函数：

$$q_{ijt} = \beta_0 + \beta_h h_{ijt} + \underbrace{f^{-1}(e_{ijt}, h_{ijt})}_{\omega_{ijt}} + \epsilon_{ijt}$$

令 $\phi(e_{ijt}, h_{ijt}) = \beta_0 + \beta_h h_{ijt} + f^{-1}(e_{ijt}, h_{ijt})$。

$$q_{ijt} = \phi(e_{ijt}, h_{ijt}) + \epsilon_{ijt}$$

-   **操作：** 使用非参数方法（如多项式展开）将 $e_{ijt}$ 和 $h_{ijt}$ 对 $q_{ijt}$ 进行回归。
-   **目的：** 得到 $\hat{\phi}_{ijt}$（预测产出）和 $\hat{\epsilon}_{ijt}$（纯误差）。此时还不能识别 $\beta_h$，因为 $h_{ijt}$ 既在线性部分里，也在非参数函数 $f^{-1}$ 里，存在共线性。

### 第二阶段：GMM 估计结构参数

这一步利用 $\omega_{ijt}$ 的演化规律来识别参数。

-   **马尔可夫假设 (Markov Assumption)：** 生产率遵循一阶马尔可夫过程：

    $$\omega_{ijt} = \rho \omega_{ijt-1} + \xi_{ijt}$$

    其中 $\xi_{ijt}$ 是当期的生产率冲击 (Innovation/Shock)。

-   **构造** $\omega$**：** 对于任意给定的参数候选值 $(\beta_h^*, \beta_0^*)$，我们可以计算出对应的生产率：

    $$\omega_{ijt}(\beta^*) = \hat{\phi}_{ijt} - \beta_0^* - \beta_h^* h_{ijt}$$

-   **构造残差：** 将 $\omega$ 代入马尔可夫过程，得到残差 $\xi_{ijt} + \epsilon_{ijt}$。

-   **矩条件 (Moment Conditions) - 核心识别：** 我们需要找到工具变量 $Z_{ijt}$，使得 $E[(\xi_{ijt} + \epsilon_{ijt}) \otimes Z_{ijt}] = 0$。

    作者使用的工具变量 $Z_{ijt} = \{1, p_{it}^{coal}, p_{it-1}^{coal}, h_{ijt-1}\}$。

    -   $h_{ijt-1}$ **(滞后一期投入)：** 有效。因为 $t-1$ 期的投入是在 $t$ 期冲击 $\xi_{ijt}$ 发生之前决定的，所以不相关。
    -   $p_{it}^{coal}$ **(煤价)：** 有效。煤价由市场或政府决定，对于单个机组是外生的（Exogenous），且价格会影响投入 $h_{ijt}$ 的选择（相关性），因此是很好的 IV。

    **GMM 目标函数：**

    $$\min_{\beta} \left( \frac{1}{N} Z' \eta(\beta) \right)' W \left( \frac{1}{N} Z' \eta(\beta) \right)$$

## 5. 反事实分析框架 (Counterfactual Analysis)

估计出 $\hat{\beta}_h$ 和 $\hat{\omega}_{ijt}$ 后，作者并没有止步，而是通过**求解优化问题**来模拟不同政策情景。这是结构化论文的亮点。

**核心逻辑：** 重新分配 (Reallocation)。 如果政府不再按电厂分配指标，而是允许在省内或区域内自由交易配额，为了最小化总成本，边际成本低的机组应该多发电。

**省一级优化问题 (Equation 14-16)：**

$$\min_{\{Q_{ijt}\}} \sum_{i \in \text{Province}} \sum_{j} \text{Cost}_{ijt}(Q_{ijt})$$

$$\text{s.t.} \sum Q_{ijt} \ge Q_{\text{target}}, \quad Q_{ijt} \le \text{Capacity}$$

-   **变量成本函数：** 作者利用估计出的参数，倒推出了成本函数：

    $$C_{ijt} = P_{it} \cdot M_{ijt} = P_{it} \cdot \frac{1}{\theta_{it}H_0} \left( \frac{Q_{ijt}}{e^{\beta_0 + \omega_{ijt}}} \right)^{\frac{1}{\beta_h}}$$

    _注意看分母中的_ $\omega_{ijt}$_：生产率越高的机组，生成同样_ $Q$ _所需的成本越低。_

**结论逻辑：** 这种优化本质上是让 $\omega_{ijt}$ 大的机组多发电，$\omega_{ijt}$ 小的机组少发电，从而在总发电量不变的情况下，降低总煤耗和总排放。

## 小白也能懂

你需要明确哪些是你的**输入 (Data)**，哪些是你要解出的**未知数 (Parameters)**，以及哪些是**外生给定 (Calibrated)** 的。

### 6.1 参数清单 (Parameter Space)

你需要估计的参数集合 $\Theta$。注意，这篇论文不仅估计一套参数，而是对三类机组（Large, Medium, Small）分别估计，所以参数空间是 $3 \times$。

| 参数符号 | 含义 | 来源 | 为什么需要结构估计？ |
| --- | --- | --- | --- |
| $\beta_{h, g}$ | 热能输入的规模报酬 | **结构估计 (GMM)** | 决定了边际成本曲线的形状。OLS 估计会有偏，导致分配效率计算错误。 |
| $\beta_{0, g}$ | 基准生产率常数 | **结构估计 (GMM)** | 决定了该类型机组的平均技术水平。 |
| $\rho_g$ | 生产率持续性 (Persistence) | **结构估计 (GMM)** | 决定了生产率随时间的演变，用于在 GMM 中分离出随机冲击 $\xi$。 |
| $\omega_{ijt}$ | **机组级生产率 (Latent)** | **推导算出 (Derived)** | **这是最重要的副产品**。反事实分析全是基于这个分布做的。 |
| $\gamma$ | CO2 排放系数 | **校准 (Calibrated)** | 工程常数，不需要估计 (0.0838 tons/GJ)。 |
| $H_0$ | 标准煤热值 | **校准 (Calibrated)** | 工程标准，不需要估计。 |

### 6.2 数据需求 (Data Space)

要运行代码，你需要构建一个 Panel Dataset，包含以下列：

| 变量符号 | 实际数据列名 (Example) | 处理方式 | 在模型中的角色 |
| --- | --- | --- | --- |
| $j, i, t$ | `unit_id`, `plant_id`, `year` | 索引 | 面板数据 ID |
| $Q_{ijt}$ | `generation_output` | 取对数 $\to q_{ijt}$ | 生产函数左侧变量 (Dep Var) |
| $H_{ijt}$ | `coal_consumption` \* `heat_value` | 取对数 $\to h_{ijt}$ | 内生解释变量 (Endogenous Regressor) |
| $E_{ijt}$ | `auxiliary_power` | 取对数 $\to e_{ijt}$ | **代理变量 (Proxy)**，用于反解 $\omega$ |
| $P_{it}^{coal}$ | `coal_price` | 取对数 | **工具变量 (IV)**，用于构建矩条件 |
| $K_j$ | `capacity_mw` | 保持原值 | 反事实分析中的物理约束 |
| $Type_j$ | `capacity_type` (L/M/S) | 分组依据 | 决定用哪一组参数进行估计 |

### 6.3 映射逻辑：从数据到参数的算法流 (The Algorithm)

这就是你要写的代码逻辑（比如用 Python 或 Stata 编写）：

**Step 0: 数据清洗** 按 `capacity_type` 把数据分成三个子样本（Large, Medium, Small）。对每个子样本分别执行以下步骤。

**Step 1: 剔除测量误差 (First Stage)**

-   **目标：** 得到纯净的产出预测值 $\hat{\phi}_{ijt}$。

-   **方法：** 运行非参数回归。

    $$q_{ijt} = c + \beta_h h_{ijt} + \text{Poly}(h_{ijt}, e_{ijt}) + \epsilon_{ijt}$$

    _(注意：实际操作中，通常把_ $\beta_h h$ _也并入多项式中一起估，因为这一步识别不出_ $\beta_h$_)_。

-   **输出：** 得到 $\hat{\phi}_{ijt}$ （即 $q_{ijt} - \hat{\epsilon}_{ijt}$）。

**Step 2: GMM 寻优 (Second Stage)** 这是求解器（Solver，如 `scipy.optimize.minimize`）的工作流程：

1. **Guess:** 求解器猜一组参数值 $\Theta^{guess} = \{\beta_h^*, \beta_0^*, \rho^*\}$。

2. **Imply** $\omega$**:** 利用猜测的 $\beta$，从 Step 1 的结果中算出隐含的生产率：

    $$\omega_{ijt}(\Theta^{guess}) = \hat{\phi}_{ijt} - \beta_0^* - \beta_h^* h_{ijt}$$

3. **Recover Shock:** 利用猜测的 $\rho$，算出当前的生产率冲击：

    $$\xi_{ijt}(\Theta^{guess}) = \omega_{ijt} - \rho^* \omega_{ijt-1}$$

4. **Moment Condition:** 检查冲击 $\xi$ 是否与工具变量 $Z$ 正交。计算目标函数值 $J$：

    $$J = \left\| \frac{1}{N} \sum (\xi_{ijt} \cdot Z_{ijt}) \right\|^2$$

5. **Iterate:** 求解器不断调整 $\Theta^{guess}$，直到 $J$ 最小（接近 0）。

**Step 3: 收敛与后续**

-   当 $J$ 最小化时，得到的 $\hat{\beta}_h$ 就是你的结构参数估计值。
-   **最后一步（关键）：** 将最优参数代回 Step 2 的公式，计算出最终的每个机组每年的生产率 $\hat{\omega}_{ijt}$。
-   有了 $\hat{\beta}_h$ 和 $\hat{\omega}_{ijt}$，你就可以画出那张边际成本曲线图，并开始做反事实模拟（比如：把所有机组的 $\omega$ 拿出来，按效率高低重新分配 $Q$）。

### 6.4 为什么是这个方法？(Methodology Choice)

-   **为什么不直接用 OLS?** 数据空间中，$h_{ijt}$ (煤耗) 和 $\omega_{ijt}$ (不可观测效率) 高度正相关。OLS 无法区分“这是因为投入多产出多”还是“这是因为效率高产出多”，导致 $\beta_h$ 估高了。
-   **为什么不用固定效应 (Fixed Effects)?** FE 假设 $\omega_{ij}$ 是不随时间变化的常数。但在电力行业，机组的老化、维护状态会让效率逐年波动。FE 无法处理 Time-varying unobservables。
-   **为什么是 Proxy 方法?** 因为电力行业有一个完美的物理代理变量——**厂用电 (Auxiliary Power)**。物理定律决定了机组运行状态越好，辅机耗电通常有特定规律。这满足了单调性假设，使得反解 $\omega$ 在工程上非常合理。

## 总结：你作为博士生需要掌握的 "Takeaway"

1. **Modeling:** 使用 Leontief 生产函数描述电力行业是恰当的，因为它捕捉了物理约束。
2. **Estimation:** 只要看到生产函数估计，第一反应必须是“内生性”，第二反应是“代理变量法（Control Function）”。本文使用了厂用电作为代理变量。
3. **Application:** 结构化估计的最终目的不仅仅是得到 $\beta$，而是为了获得 $\omega$（异质性）。有了 $\omega$，就能计算影子成本，进而做反事实的资源错配（Misallocation）分析。
