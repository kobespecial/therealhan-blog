---
date: "2025-12-30T11:25:27+08:00"
draft: false
title: "结构估计──CCP法"
toc: true
math: true
slug: CCP
categories:
    - 结构方程模型
tags:
    - 结构估计方法
---

两步法，CCP（Conditional Choice Probability）完全指南

<!--more-->

# CCP（Conditional Choice Probability）完全指南

---

## 第一章：CCP 的基本概念

### 1.1 直观理解：一句话定义

**CCP = 在给定状态下，选择某个决策的概率**

```
例子：
- 状态：你的旧车里程已经15万公里
- 决策：是否进行大修
- CCP：在这个里程下，选择大修的概率 = ?

数据观察：
  在里程为15万公里的100辆车中，有45辆选择大修
  → CCP(d=1|x=15) = 45% = 0.45
```

### 1.2 正式定义

设：

-   $x_t$ = 状态（连续变量，如里程、年龄、财务状况等）
-   $d_t \in \{0, 1\}$ = 决策（是否采取某个行动）
-   $\epsilon_t$ = 不可观测的私人冲击（econometrician 看不到）

**条件选择概率**： $$P(d_t = 1 | x_t = x) = \Pr(d_t = 1 | x_t = x)$$

**含义**：给定状态为 $x$ 时，代理人选择 $d=1$ 的概率。

### 1.3 为什么有"冲击"？

**现实观察**：同样的状态下，决策不同

```
里程都是15万的两辆车：
  车A → 选择大修
  车B → 不大修

为什么不同？

原因1：我们看不到所有信息
  - 车主的财务状况
  - 对可靠性的偏好
  - 计划卖车的时间
  - 等等...

这些看不到的因素 = 私人冲击 ε
```

**引入冲击的目的**：

-   ✓ 解释同状态下的决策异质性
-   ✓ 使得 CCP 是"概率"（0 到 1 之间）
-   ✓ 允许用离散选择模型（logit, probit 等）

---

## 第二章：从理论到实证的三个层次

### 2.1 层次 1：理论模型中的 CCP

**动态规划模型**：

代理人最大化： $$V(x) = \max_{d} \{ u(x,d) + \beta \mathbb{E}[V(x') | x, d] \}$$

这导出**最优决策规则** $d^*(x)$（确定性）。

**但加入冲击后**：

代理人最大化： $$V(x) = \max_d \{ u(x,d) + \epsilon_d + \beta \mathbb{E}[V(x') | x, d] \}$$

给定状态 $x$ 和冲击 $\epsilon = (\epsilon_0, \epsilon_1)$，最优决策是： $$d^*(x, \epsilon) = \begin{cases} 1 & \text{if } u(x,1) + \epsilon_1 + \beta \mathbb{E}[V|d=1] \geq u(x,0) + \epsilon_0 + \beta \mathbb{E}[V|d=0] \\ 0 & \text{otherwise} \end{cases}$$

**CCP 的定义**： $$P(d=1|x) = \Pr(d^*(x,\epsilon) = 1 | x) = \Pr(\epsilon_1 - \epsilon_0 \leq \Delta u(x))$$

其中 $\Delta u(x) = [u(x,0) - u(x,1)] + \beta[\mathbb{E}[V|d=0] - \mathbb{E}[V|d=1]]$

### 2.2 层次 2：在数据中观察 CCP

**原始数据**（观测到的）：

```
Car_ID  Year  Mileage  Maintenance_Decision
  1     2015   50000          0   (不维修)
  1     2016   75000          1   (维修)
  1     2017  100000          0   (不维修)
  2     2015   60000          1   (维修)
  2     2016   85000          0   (不维修)
  ...
```

**构造 CCP**（从数据提取）：

```python
def compute_empirical_ccp(data, x_grid):
    """
    从原始数据计算经验CCP
    """

    # 方法1：分箱（简单）
    ccp_binned = []

    for x in x_grid:
        # 找状态在x附近的所有观测
        in_bin = (data['mileage'] >= x-500) & (data['mileage'] < x+500)

        if in_bin.sum() > 0:
            # 该箱中维修的比例 = CCP估计
            ccp_val = data.loc[in_bin, 'maintenance'].mean()
        else:
            ccp_val = np.nan

        ccp_binned.append(ccp_val)

    # 方法2：光滑估计（Lowess平滑）
    from statsmodels.nonparametric.smoothers_lowess import lowess

    smoothed = lowess(
        data['maintenance'],      # y: 是否维修
        data['mileage'],          # x: 里程
        frac=0.2                  # 带宽
    )

    # 在网格上插值
    ccp_smooth = np.interp(x_grid, smoothed[:, 0], smoothed[:, 1])

    return ccp_smooth
```

**典型形状**：S 形曲线

```
CCP(d=1|x)
    ↑
  1.0 |                    ╱
      |                  ╱
  0.5 |                ╱
      |              ╱
  0.0 |────────────╱─────→ 状态 x
      0        x*        max
```

其中 $x^*$ 是维修阈值（CCP = 0.5 的点）

### 2.3 层次 3：在估计中使用 CCP

**结构估计的关键角色**：

给定参数 $\theta = (\rho, RC, c_1, c_2)$，可以：

1. 求解动态规划模型 → 得到价值函数 $V(x; \theta)$
2. 从价值函数计算理论 CCP：$P(d=1|x; \theta)$
3. 比较理论 CCP 与数据中的经验 CCP
4. 最小化二者的距离 → 估计 $\theta$

```
参数 θ
   ↓
求解HJB/Bellman
   ↓
计算理论CCP(x; θ)
   ↓
与经验CCP比较
   ↓
调整θ
   ↓
重复直到拟合
```

---

## 第三章：数学细节

### 3.1 离散选择模型中的 CCP

**假设**：私人冲击 $\epsilon$ 服从 Type-I 极值分布（Gumbel）

$$F_\epsilon(\epsilon) = \exp(-e^{-\epsilon})$$

**结果**：CCP 有 logit 形式

$$P(d=1|x) = \frac{\exp(V_1(x))}{\exp(V_0(x)) + \exp(V_1(x))} = \frac{1}{1 + \exp(V_0(x) - V_1(x))}$$

其中：

-   $V_a(x)$ = 选择 $a$ 的"消费效用" + 未来期望价值
-   $V_0(x) = u(x,0) + \beta \mathbb{E}[V(x')|d=0]$
-   $V_1(x) = u(x,1) + \beta \mathbb{E}[V(x')|d=1]$

**logit 的含义**：

```
如果 V_0(x) >> V_1(x)：
  P(d=1|x) ≈ 0  (几乎不选d=1)

如果 V_0(x) ≈ V_1(x)：
  P(d=1|x) ≈ 0.5  (无差异)

如果 V_0(x) << V_1(x)：
  P(d=1|x) ≈ 1  (肯定选d=1)
```

### 3.2 Rust 模型中的 CCP

**具体例子：车辆维修决策**

状态：$x_t$ = 里程

流收益：

$$
u(x,d) = \begin{cases}
-c(x) & \text{if } d = 0 \text{ (不维修，支付维护成本)} \\
-RC - c(0) & \text{if } d = 1 \text{ (维修，支付维修成本RC)}
\end{cases}
$$

状态转移： $$x_{t+1} = x_t + 1 + \epsilon_{growth,t}$$

（假设每期里程增加 1 个单位，加随机冲击）

**Bellman 方程**（discrete time, 1-period）： $$V(x) = \max_d \{ u(x,d) + \epsilon_d + \beta \mathbb{E}[V(x+1) | d] \}$$

简化（忽略状态增长的冲击）： $$V(x) = \max_d \{ u(x,d) + \epsilon_d + \beta V(x+1) \}$$

**CCP**（利用 logit 假设）： $$P(d=1|x) = \frac{\exp(\bar{V}_1(x))}{\exp(\bar{V}_0(x)) + \exp(\bar{V}_1(x))}$$

其中： $$\bar{V}_0(x) = -c(x) + \beta V(x+1)$$ $$\bar{V}_1(x) = -RC - c(0) + \beta V(0)$$

**关键观察**： $$P(d=1|x) = \frac{\exp(-RC - c(0) + \beta V(0))}{\exp(-c(x) + \beta V(x+1)) + \exp(-RC - c(0) + \beta V(0))}$$

简化（令 $\Delta = -RC + c(x) - c(0) + \beta V(0) - \beta V(x+1)$）： $$P(d=1|x) = \frac{1}{1 + \exp(-\Delta)}$$

---

## 第四章：CCP 的三种估计方法

### 4.1 方法 1：非参数估计（无模型）

**直接从数据计算经验频率**

```python
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

class NonparametricCCPEstimation:
    """
    完全非参数的CCP估计
    不需要任何模型假设！
    """

    def __init__(self, data):
        """
        data: DataFrame
        需要列: 'state' (状态), 'choice' (决策)
        """
        self.data = data

    def estimate_ccp_binning(self, x_grid, bin_width=1):
        """
        方法1：简单分箱

        优点：直观
        缺点：粗糙，样本小时不稳定
        """

        ccp = np.zeros(len(x_grid))

        for i, x in enumerate(x_grid):
            # 找状态在[x - bin_width/2, x + bin_width/2]内的观测
            in_bin = (self.data['state'] >= x - bin_width/2) & \
                     (self.data['state'] < x + bin_width/2)

            if in_bin.sum() > 0:
                ccp[i] = self.data.loc[in_bin, 'choice'].mean()
            else:
                ccp[i] = np.nan

        return ccp

    def estimate_ccp_lowess(self, x_grid, frac=0.2):
        """
        方法2：局部加权回归(Lowess)

        优点：光滑，自适应
        缺点：需要选择带宽参数
        """

        # Lowess拟合
        smoothed = lowess(
            self.data['choice'],   # y: 选择(0或1)
            self.data['state'],    # x: 状态
            frac=frac              # 局部窗口大小
        )

        # 在指定网格上插值
        ccp = np.interp(x_grid, smoothed[:, 0], smoothed[:, 1])

        return ccp

    def estimate_ccp_kernel_regression(self, x_grid, bandwidth=1):
        """
        方法3：核回归

        CCP(x) = E[choice | state = x]
               = Σ K((state_i - x) / h) · choice_i / Σ K((state_i - x) / h)

        其中K是核函数(如高斯核), h是带宽
        """

        from scipy.stats import gaussian_kde

        ccp = np.zeros(len(x_grid))

        for i, x in enumerate(x_grid):
            # 计算核权重
            distances = np.abs(self.data['state'] - x)
            weights = np.exp(-(distances / bandwidth) ** 2)

            # 加权平均
            ccp[i] = np.average(self.data['choice'], weights=weights)

        return ccp

    def bootstrap_ccp_confidence_interval(self, x_grid, n_bootstrap=500, alpha=0.05):
        """
        Bootstrap置信区间
        """

        ccp_bootstraps = np.zeros((n_bootstrap, len(x_grid)))

        for b in range(n_bootstrap):
            # 有放回重抽样
            idx = np.random.choice(len(self.data), size=len(self.data), replace=True)
            data_boot = self.data.iloc[idx]

            # 计算bootstrap样本的CCP
            ccp_boot = self._estimate_ccp_lowess_internal(data_boot, x_grid)
            ccp_bootstraps[b, :] = ccp_boot

        # 分位数
        lower = np.percentile(ccp_bootstraps, 100 * alpha / 2, axis=0)
        upper = np.percentile(ccp_bootstraps, 100 * (1 - alpha / 2), axis=0)

        return lower, upper

    def _estimate_ccp_lowess_internal(self, data, x_grid):
        """内部函数"""
        smoothed = lowess(data['choice'], data['state'], frac=0.2)
        return np.interp(x_grid, smoothed[:, 0], smoothed[:, 1])

    def plot_empirical_ccp(self, x_grid, methods=['binning', 'lowess']):
        """可视化"""

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        if 'binning' in methods:
            ccp_bin = self.estimate_ccp_binning(x_grid)
            plt.plot(x_grid, ccp_bin, 'o-', label='Binning', alpha=0.6)

        if 'lowess' in methods:
            ccp_lowess = self.estimate_ccp_lowess(x_grid)
            plt.plot(x_grid, ccp_lowess, 's-', label='Lowess', alpha=0.6)

        # 原始数据散点
        plt.scatter(self.data['state'], self.data['choice'],
                   alpha=0.1, s=10, label='Raw data')

        plt.xlabel('状态 (State)')
        plt.ylabel('P(选择 d=1 | State)')
        plt.title('条件选择概率 (Conditional Choice Probability)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.show()

# 示例
np.random.seed(42)

# 生成模拟数据：里程与维修决策
n = 1000
state = np.random.uniform(0, 450, n)  # 里程: 0到45万

# 真实CCP（S形曲线）
true_ccp = 1 / (1 + np.exp(-(state - 225) / 50))

# 加噪声生成决策
choice = (np.random.uniform(0, 1, n) < true_ccp).astype(int)

data = pd.DataFrame({'state': state, 'choice': choice})

# 估计
est = NonparametricCCPEstimation(data)
est.plot_empirical_ccp(np.linspace(0, 450, 100), methods=['binning', 'lowess'])

# 置信区间
x_test = np.array([100, 225, 350])
lower, upper = est.bootstrap_ccp_confidence_interval(x_test)

print("状态        CCP估计    95% 置信区间")
for x, l, u in zip(x_test, lower, upper):
    ccp_val = est.estimate_ccp_lowess(np.array([x]))[0]
    print(f"{x:3.0f}        {ccp_val:.4f}    [{l:.4f}, {u:.4f}]")
```

**输出示例**：

```
状态        CCP估计    95% 置信区间
100        0.1234    [0.0856, 0.1612]
225        0.5023    [0.4567, 0.5479]
350        0.8901    [0.8534, 0.9268]
```

### 4.2 方法 2：半参数估计（Hotz-Miller CCP 方法）

**思路**：用非参数 CCP 反推模型参数

```python
class SemiparametricCCPEstimation:
    """
    Hotz-Miller (1993) 方法

    步骤：
    1. 非参数估计CCP
    2. 反演得到价值函数
    3. 估计参数（无需重复求解动态规划）
    """

    def __init__(self, data, x_grid, beta=0.95):
        self.data = data
        self.x_grid = x_grid
        self.beta = beta

        # Step 1: 非参数CCP估计
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(data['choice'], data['state'], frac=0.2)
        self.ccp_empirical = np.interp(x_grid, smoothed[:, 0], smoothed[:, 1])

    def invert_ccp_to_value_function(self):
        """
        Step 2: 从CCP反演价值函数

        Hotz-Miller反演公式：

        从logit CCP，我们知道：
        P(d=1|x) = exp(V_1(x)) / [exp(V_0(x)) + exp(V_1(x))]

        定义：
        Λ_d(x) = log(P(d|x)) - ψ + log(1 - P(d|x)) + ψ
               = log(P(d|x) / (1 - P(d|x)))

        其中ψ是Euler-Mascheroni常数 ≈ 0.5772

        则：
        V_1(x) - V_0(x) = Λ_1(x) - Λ_0(x)
                        = log(P(d=1|x) / (1 - P(d=1|x)))

        通过递归和固定点方程，可反演出V_d(x)
        """

        # Euler-Mascheroni常数
        psi = 0.5772

        # 计算选择特定价值的对数比率
        p = self.ccp_empirical
        p = np.clip(p, 1e-10, 1 - 1e-10)  # 避免log(0)

        log_odds = np.log(p / (1 - p))

        # 固定点方程：
        # V_0(x) = u(x,0) + β [p(x) * V(0) + (1-p(x)) * V(x')]
        # V_1(x) = u(x,1) + β V(0)
        #
        # 解这个系统得到V(x)

        # 简化处理（假设线性成本函数）
        c = 0.1 * self.x_grid  # 假设成本函数

        # 初始化
        V = np.zeros_like(self.x_grid)

        # 迭代求解固定点
        for iteration in range(100):
            V_old = V.copy()

            # 更新V_0(x) = -c(x) + β E[V(x')|d=0]
            V_0 = -c + self.beta * np.roll(V_old, 1)  # x' = x+1

            # 更新V_1(x) = -RC + β V(0)
            # (假设维修后回到状态0)
            RC = 10  # 临时假设
            V_1 = -RC + self.beta * V_old[0]

            # CCP形式的价值函数
            # V(x) = (1/σ) log(exp(σ V_0) + exp(σ V_1))
            # 其中σ是logit的scale（与离散选择冲击分布有关）

            sigma = 100  # 假设
            V = (1 / sigma) * np.log(
                np.exp(sigma * V_0) + np.exp(sigma * V_1)
            )

            if np.max(np.abs(V - V_old)) < 1e-6:
                break

        self.V_inverted = V
        return V

    def estimate_parameters(self):
        """
        Step 3: 估计模型参数

        无需重复求解动态规划！
        """

        # 给定反演的V(x)，可以直接读出：
        # log(P(d=1|x)) - log(P(d=0|x)) = V_1(x) - V_0(x)

        p = self.ccp_empirical
        p = np.clip(p, 1e-10, 1 - 1e-10)

        # CCP导出的价值差异
        log_odds = np.log(p / (1 - p))

        # log_odds(x) = -RC + c(x) - c(0) + β[V(0) - V(x)]

        # 可以用回归估计参数
        from sklearn.linear_model import LinearRegression

        # 设计矩阵
        X = np.column_stack([
            np.ones(len(self.x_grid)),  # 截距（包含-RC）
            self.x_grid                  # 线性成本项
        ])

        # 回归：log_odds ~ α_0 + α_1 * x
        model = LinearRegression()
        model.fit(X, log_odds)

        # 从系数反演参数
        alpha_0, alpha_1 = model.intercept_, model.coef_[1]

        # α_1对应c(x)的系数
        c1_estimated = alpha_1

        # α_0包含-RC和价值函数项
        # 需要额外假设来分离...

        print(f"估计的成本函数斜率 c1 = {c1_estimated:.6f}")

        return c1_estimated

# 使用
est_semi = SemiparametricCCPEstimation(data, np.linspace(0, 450, 100))
V_inverted = est_semi.invert_ccp_to_value_function()
c1_est = est_semi.estimate_parameters()
```

### 4.3 方法 3：参数最大似然估计

**思路**：给定参数，计算理论 CCP，匹配数据中的经验 CCP

```python
class StructuralMaximumLikelihood:
    """
    参数结构估计

    给定参数θ，计算理论CCP，最大化似然函数
    """

    def __init__(self, data, x_grid, beta=0.95):
        self.data = data
        self.x_grid = x_grid
        self.beta = beta

    def solve_bellman(self, params):
        """
        给定参数，求解贝尔曼方程得到价值函数V(x)

        params = (c1, c2, RC)
        """

        c1, c2, RC = params

        # 成本函数
        def cost(x):
            return c1 * x + c2 * x**2

        # 初始化
        V = np.zeros_like(self.x_grid)

        # 迭代求解（反向迭代）
        for iteration in range(200):
            V_old = V.copy()

            for i in range(len(self.x_grid) - 1, -1, -1):
                x = self.x_grid[i]

                # 不维修：支付维护成本，状态增加1
                flow_0 = -cost(x)
                x_next_0 = min(x + 1, self.x_grid[-1])
                idx_next_0 = np.argmin(np.abs(self.x_grid - x_next_0))
                V_next_0 = V_old[idx_next_0]

                bar_v_0 = flow_0 + self.beta * V_next_0

                # 维修：支付维修成本，状态回到0
                flow_1 = -RC - cost(0)
                bar_v_1 = flow_1 + self.beta * V_old[0]

                # 用logit结合（处理离散选择冲击）
                sigma = 1000  # 冲击分布scale
                V[i] = (1 / sigma) * np.log(
                    np.exp(sigma * bar_v_0) + np.exp(sigma * bar_v_1)
                )

            # 收敛检查
            if np.max(np.abs(V - V_old)) < 1e-8:
                break

        return V

    def compute_theoretical_ccp(self, params):
        """
        给定参数，计算理论CCP
        """

        c1, c2, RC = params

        # 求解贝尔曼方程
        V = self.solve_bellman(params)

        # 计算CCP
        ccp = np.zeros_like(self.x_grid)

        for i, x in enumerate(self.x_grid):
            # 不维修的价值
            flow_0 = -(c1 * x + c2 * x**2)
            x_next_0 = min(x + 1, self.x_grid[-1])
            idx_next_0 = np.argmin(np.abs(self.x_grid - x_next_0))
            bar_v_0 = flow_0 + self.beta * V[idx_next_0]

            # 维修的价值
            flow_1 = -RC - (c1 * 0 + c2 * 0**2)
            bar_v_1 = flow_1 + self.beta * V[0]

            # Logit CCP
            ccp[i] = 1 / (1 + np.exp(bar_v_0 - bar_v_1))

        return ccp

    def compute_log_likelihood(self, params):
        """
        计算对数似然函数

        L(θ) = Σ_i [d_i * log(P(d=1|x_i;θ)) + (1-d_i) * log(1-P(d=1|x_i;θ))]
        """

        try:
            # 计算理论CCP
            ccp_theory = self.compute_theoretical_ccp(params)

            # 为每个数据点查找对应的CCP
            x_data = self.data['state'].values
            d_data = self.data['choice'].values

            ccp_at_data = np.interp(x_data, self.x_grid, ccp_theory)

            # 数值稳定性：限制CCP在[1e-10, 1-1e-10]
            ccp_at_data = np.clip(ccp_at_data, 1e-10, 1 - 1e-10)

            # 二项对数似然
            ll = d_data * np.log(ccp_at_data) + \
                 (1 - d_data) * np.log(1 - ccp_at_data)

            return np.sum(ll)

        except:
            # 如果计算失败，返回-∞
            return -1e10

    def estimate(self, initial_params=None, bounds=None):
        """
        最大似然估计
        """

        if initial_params is None:
            initial_params = np.array([0.1, 0.01, 10.0])

        if bounds is None:
            bounds = [(0.001, 0.5), (0.0001, 0.1), (1.0, 50.0)]

        # 定义目标函数（负对数似然）
        def objective(params):
            return -self.compute_log_likelihood(params)

        # 优化
        from scipy.optimize import minimize

        print("开始最大似然估计...")
        print(f"初始参数: c1={initial_params[0]:.4f}, "
              f"c2={initial_params[1]:.6f}, RC={initial_params[2]:.4f}")

        result = minimize(
            objective,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={'ftol': 1e-10, 'maxiter': 500}
        )

        print(f"\n估计完成！")
        print(f"最优参数: c1={result.x[0]:.4f}, "
              f"c2={result.x[1]:.6f}, RC={result.x[2]:.4f}")
        print(f"最大对数似然: {-result.fun:.2f}")

        self.params_estimated = result.x

        return result

# 使用
est_ml = StructuralMaximumLikelihood(data, np.linspace(0, 450, 100))
result = est_ml.estimate(initial_params=[0.1, 0.01, 10.0])

# 比较：理论CCP vs 经验CCP
ccp_theory = est_ml.compute_theoretical_ccp(result.x)

# 非参数经验CCP
from statsmodels.nonparametric.smoothers_lowess import lowess
smoothed = lowess(data['choice'], data['state'], frac=0.2)
ccp_empirical = np.interp(np.linspace(0, 450, 100),
                          smoothed[:, 0], smoothed[:, 1])

# 绘制
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, 450, 100), ccp_empirical, 'o-',
         label='Empirical CCP', linewidth=2)
plt.plot(np.linspace(0, 450, 100), ccp_theory, 's-',
         label='Theoretical CCP (estimated)', linewidth=2)
plt.xlabel('状态 (里程)')
plt.ylabel('维修概率')
plt.title('CCP拟合效果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 第五章：CCP 的性质与应用

### 5.1 CCP 的关键性质

#### 性质 1：CCP 由状态唯一确定

$$P(d=1|x) = f(x)$$

即：**给定状态，决策概率是确定的**（虽然决策本身不确定，但概率是）

```python
# 检验性质1：相同状态下的CCP应该稳定
def test_ccp_property1(data):
    """
    在相同或相近的状态下，CCP应该相同
    """

    # 选择某个状态（如x=225）
    x_target = 225

    # 找所有状态在[x-5, x+5]的观测
    near_target = (data['state'] >= x_target - 5) & \
                  (data['state'] <= x_target + 5)

    choices_near = data.loc[near_target, 'choice'].values

    # CCP应该是这些选择的平均值
    ccp_estimated = choices_near.mean()

    print(f"在状态 x={x_target} 附近的观测数: {len(choices_near)}")
    print(f"选择d=1的比例: {ccp_estimated:.4f}")
    print(f"标准误: {np.sqrt(ccp_estimated * (1-ccp_estimated) / len(choices_near)):.4f}")
```

#### 性质 2：CCP 的平滑性

如果状态 $x$ 连续变化，CCP 应该相对平滑变化（除非有结构性的阈值）

```python
# 检验性质2：CCP的平滑性
def compute_ccp_smoothness(ccp_empirical, x_grid):
    """
    衡量CCP的"粗糙度"
    """

    # 二阶差分（曲率）
    second_differences = np.diff(ccp_empirical, n=2)

    # 粗糙度指标
    roughness = np.sum(second_differences ** 2)

    print(f"CCP粗糙度指标: {roughness:.6f}")

    # 可视化
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x_grid, ccp_empirical, 'b-', linewidth=2)
    plt.xlabel('状态')
    plt.ylabel('CCP')
    plt.title('条件选择概率')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x_grid[1:-1], second_differences, 'r-', linewidth=2)
    plt.xlabel('状态')
    plt.ylabel('二阶差分')
    plt.title('CCP的曲率')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

#### 性质 3：CCP 的有界性

$$0 \leq P(d=1|x) \leq 1, \quad \forall x$$

```python
# 检验性质3
def check_ccp_bounds(ccp):
    """CCP应该在[0,1]内"""

    if np.any(ccp < 0) or np.any(ccp > 1):
        print("⚠️  警告：CCP超出[0,1]范围！")
        print(f"最小值: {np.min(ccp):.6f}")
        print(f"最大值: {np.max(ccp):.6f}")
    else:
        print("✓ CCP在[0,1]范围内")
```

### 5.2 CCP 在识别中的作用

#### 应用 1：识别维修成本 RC

```python
def identify_RC_from_CCP_threshold(x_grid, ccp_empirical, c_params, beta=0.95):
    """
    在维修阈值处（CCP=0.5），使用边界条件识别RC

    理由：在无差异点，两个选择的价值相等
    """

    # 找CCP=0.5的状态
    idx_threshold = np.argmin(np.abs(ccp_empirical - 0.5))
    x_star = x_grid[idx_threshold]

    # 在此点，必有：V_0(x*) = V_1(x*)
    # 即：-c(x*) + β V(x*+1) = -RC + β V(0)
    # 解出：RC = c(x*) - c(0) + β[V(0) - V(x*+1)]

    c1, c2 = c_params
    c_x_star = c1 * x_star + c2 * x_star ** 2
    c_0 = 0

    # 假设价值函数是线性的（简化）
    # V(x) ≈ V(0) - k * x
    # 则：V(x*+1) ≈ V(0) - k * (x*+1)

    # 或者用数值方法求解...

    return x_star, c_x_star - c_0
```

#### 应用 2：检验模型假设

```python
def test_logit_assumption_via_ccp(ccp_empirical, x_grid):
    """
    检验Logit假设（极值分布冲击）是否合理

    如果冲击确实服从Gumbel分布，
    CCP应该满足某些性质
    """

    # 从CCP反推冲击分布
    # 如果CCP(x) = P(d=1|x)，则冲击差异的分布由CCP形状决定

    # 绘制CCP的对数赔率
    p = np.clip(ccp_empirical, 1e-10, 1 - 1e-10)
    log_odds = np.log(p / (1 - p))

    # 如果冲击是Gumbel分布，log_odds应该是状态的光滑函数

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_grid, ccp_empirical, 'b-', linewidth=2)
    plt.ylabel('CCP = P(d=1|x)')
    plt.xlabel('状态')
    plt.title('条件选择概率')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x_grid, log_odds, 'r-', linewidth=2)
    plt.ylabel('log(P/(1-P))')
    plt.xlabel('状态')
    plt.title('对数赔率')
    plt.grid(True, alpha=0.3)

    # 检验线性性（Logit特征）
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = \
        linregress(x_grid, log_odds)

    print(f"对数赔率的线性回归 R²: {r_value ** 2:.4f}")
    if r_value ** 2 > 0.95:
        print("✓ log_odds很好地拟合线性关系 → Logit假设合理")
    else:
        print("⚠️ log_odds非线性 → Logit假设可能有问题")

    plt.show()
```

---

## 第六章：CCP 的常见陷阱与注意事项

### 6.1 陷阱 1：数据稀疏导致的 CCP 不稳定

```python
def diagnose_sample_size_issue(data, x_grid, min_obs_per_bin=10):
    """
    问题：某些状态区间样本量太小
    后果：CCP估计方差大，不稳定
    """

    for x in x_grid:
        n_nearby = ((data['state'] >= x - 1) &
                    (data['state'] < x + 1)).sum()

        if n_nearby < min_obs_per_bin:
            print(f"⚠️  状态 x={x:.1f} 附近样本太少 ({n_nearby}个)")

    # 解决方案：扩大箱宽
    ccp_robust = estimate_ccp_with_adaptive_bandwidth(data, x_grid)

    return ccp_robust

def estimate_ccp_with_adaptive_bandwidth(data, x_grid, min_obs=20):
    """
    自适应带宽：保证每个点至少有min_obs个样本
    """

    ccp = np.zeros(len(x_grid))

    for i, x in enumerate(x_grid):

        # 初始带宽
        bandwidth = 1

        while True:
            in_window = (data['state'] >= x - bandwidth) & \
                       (data['state'] < x + bandwidth)

            if in_window.sum() >= min_obs:
                ccp[i] = data.loc[in_window, 'choice'].mean()
                break
            else:
                bandwidth *= 1.5  # 扩大带宽

    return ccp
```

### 6.2 陷阱 2：忽视选择偏差

```python
def test_selection_bias_in_ccp(data):
    """
    问题：如果参与选择本身是内生的怎么办？

    例如：旧车可能因为主人穷而被卖掉
    → 观察到的里程与维修决策的关系可能被偏差
    """

    # 简单检验：观察CCP与状态的关系
    state_mean_repaired = data[data['choice']==1]['state'].mean()
    state_mean_not_repaired = data[data['choice']==0]['state'].mean()

    print(f"维修的平均状态: {state_mean_repaired:.1f}")
    print(f"不维修的平均状态: {state_mean_not_repaired:.1f}")
    print(f"差异: {abs(state_mean_repaired - state_mean_not_repaired):.1f}")

    if state_mean_repaired > state_mean_not_repaired:
        print("✓ 符合预期：状态差时更容易维修")
    else:
        print("⚠️  反常：状态好反而更容易维修？ → 可能有偏差")
```

### 6.3 陷阱 3：状态测量误差

```python
def test_measurement_error_in_state(data):
    """
    问题：如果状态（如里程表）测量不准呢？
    后果：CCP被平滑化，S形变平缓
    """

    # 检验里程数据的一致性
    data_sorted = data.sort_values(['id', 'year'])

    # 里程应该单调增加
    mileage_decreases = (
        data_sorted.groupby('id')['state']
        .apply(lambda x: (x.diff() < 0).sum())
    )

    if mileage_decreases.sum() > 0:
        print("⚠️  警告：发现里程倒退")
        print(f"有 {mileage_decreases.sum()} 个单位发生过倒退")
        print("→ 可能存在测量误差")
    else:
        print("✓ 里程数据一致")
```

---

## 第七章：CCP vs 直接模型估计

### 7.1 为什么要用 CCP？

| 方面 | CCP 方法 | 直接动态规划 |
| --- | --- | --- |
| **计算效率** | ✓ 快（非参数 CCP 一次性估计） | ✗ 慢（每次梯度步都要解) |
| **模型灵活性** | ✗ 依赖 Logit 假设 | ✓ 更灵活 |
| **参数识别** | ✓ 半参数更稳健 | ✗ 容易过拟合 |
| **反事实分析** | ✗ 困难（需要完整模型） | ✓ 容易 |

### 7.2 CCP vs 直接极大似然对比

```python
def compare_ccp_vs_direct_ml():
    """
    对比两种方法的估计结果
    """

    # 生成数据
    data = generate_synthetic_data(n=500, T=100)
    x_grid = np.linspace(0, 450, 100)

    # 方法1：非参数CCP + 半参数估计
    print("=" * 60)
    print("方法1：CCP半参数估计")
    print("=" * 60)

    est_semi = SemiparametricCCPEstimation(data, x_grid)
    c1_semi = est_semi.estimate_parameters()
    time_semi = timer()

    print(f"估计结果: c1 = {c1_semi:.6f}")
    print(f"计算时间: {time_semi:.2f}秒")

    # 方法2：直接最大似然
    print("\n" + "=" * 60)
    print("方法2：结构极大似然估计")
    print("=" * 60)

    est_ml = StructuralMaximumLikelihood(data, x_grid)
    start = time.time()
    result_ml = est_ml.estimate()
    time_ml = time.time() - start

    print(f"估计结果: c1 = {result_ml.x[0]:.6f}")
    print(f"计算时间: {time_ml:.2f}秒")

    # 对比
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"时间比：ML/CCP = {time_ml / time_semi:.2f}x")
    print(f"结果差异: |c1_ML - c1_CCP| = {abs(result_ml.x[0] - c1_semi):.6f}")

compare_ccp_vs_direct_ml()
```

---

## 总结：CCP 速查表

### CCP 的定义

$$P(d=1|x) = \Pr(选择\ d=1\ |\ 状态\ x)$$

### 三种估计方法

| 方法 | 算法 | 优点 | 缺点 |
| --- | --- | --- | --- |
| **非参数** | 局部平均/光滑 | 无模型假设 | 样本需求大 |
| **半参数** | CCP → 反演 V → 估计 θ | 快速 | 需要 Logit 假设 |
| **参数** | θ → 解 DP → CCP → MLE | 精确 | 计算慢 |

### 快速检查清单

-   [ ] CCP 是否在[0,1]范围内？
-   [ ] CCP 是否有 S 形（单调）形状？
-   [ ] 样本量是否足够（n > 100）？
-   [ ] 是否存在测量误差？
-   [ ] 是否有选择偏差？
-   [ ] 状态是否合理变化？

---

## 通俗版

# CCP（条件选择概率）的直观讲解

---

## 第一章：CCP 是什么

### 1.1 最简单的定义

**CCP = 在给定状态下，选择某个行动的概率**

比如：

-   状态：你的旧车已跑了 15 万公里
-   行动：是否进行大修
-   CCP：在 15 万公里这个状态，你选择大修的概率是多少？

### 1.2 为什么叫"条件"选择概率

因为概率**以状态为条件**，记作： $$P(d=1|x)$$

其中：

-   $d$ = 决策（1=大修，0=不大修）
-   $x$ = 状态（里程）
-   竖线表示"给定...条件"

### 1.3 为什么有概率而不是确定决策

**现实中同样的里程，不同人做出不同决策：**

15 万公里的 100 辆车中：

-   45 辆选择大修
-   55 辆选择不大修

这种异质性可能来自：

-   每个人对可靠性的偏好不同
-   每个人的经济状况不同
-   有些人打算卖车，有些人打算继续开
-   还有我们看不到的其他原因

**经济学家的处理方式：** 用概率来描述这种异质性

$$P(d=1|x=15) = \frac{45}{100} = 0.45$$

---

## 第二章：CCP 的三个来源

### 2.1 来源 1：经验观察（数据中的 CCP）

**直接计数法：**

如果我们有很多观测，可以这样计算 CCP：

1. 找出所有状态在$x$附近的观测
2. 计算这些观测中选择$d=1$的比例
3. 这个比例就是 CCP 的估计值

例如，汽车维修数据中：

-   在 15 万公里 ±5 万公里的所有观测中
-   统计有多少车选择了大修
-   大修的比例 ≈ 该里程的 CCP

**这就是非参数 CCP 估计**，完全不需要任何模型假设，直接从数据读出。

### 2.2 来源 2：理论模型（动态规划中的 CCP）

想象一个决策者面临以下问题：

**状态**：车的状态为$x$（里程）

**两个选择：**

-   选择 0：不大修，支付维护成本$c(x)$，车状态下期增加到$x+1$
-   选择 1：进行大修，支付大修费用$RC$，车状态重置为 0

**决策者要最大化生命周期效用**：

$$V(x) = \max_d \{ u(x,d) + \beta \mathbb{E}[V(x')|d] \}$$

其中：

-   $u(x,d)$ = 当期收益（成本）
-   $\beta$ = 折现因子
-   $\mathbb{E}[V(x')|d]$ = 选择$d$后的未来期望价值

**关键问题：** 给定状态$x$，决策者选择$d=1$的概率是多少？

答案取决于：两个选择的价值比较结果

### 2.3 来源 3：私人冲击（为什么出现概率）

即使模型是确定的，我们也观察到随机决策，原因是**私人冲击**：

每个代理人选择时面临的效用不完全相同： $$V(x) = \max_d \{ u(x,d) + \epsilon_d + \beta \mathbb{E}[V(x')|d] \}$$

其中$\epsilon_d$是 econometrician（数据分析者）看不到的因素。

这些不可观测的因素可能是：

-   对舒适度/安全的个人偏好
-   对可靠性的风险态度
-   出行计划（计划卖车的时间）
-   其他我们没有数据的私人信息

**经济学家假设：** $\epsilon_d$服从某个分布（通常是极值分布）

**结果：** CCP 变成一个概率

$$P(d=1|x) = \Pr(\text{选择1的效用} \geq \text{选择0的效用})$$ $$= \Pr(\epsilon_1 - \epsilon_0 \leq \Delta V(x))$$

如果$\epsilon$服从标准的极值分布，这等价于**logit 形式**：

$$P(d=1|x) = \frac{\exp(V_1(x))}{\exp(V_0(x)) + \exp(V_1(x))} = \frac{1}{1+\exp(V_0(x)-V_1(x))}$$

---

## 第三章：理解 CCP 的形状

### 3.1 典型的 CCP 曲线

想象维修决策的 CCP 随里程增加而变化：

```
CCP(维修概率)
    1.0 |              ╱
        |            ╱
    0.5 |          ╱  ← 维修阈值 x*
        |        ╱
    0.0 |_______╱_______
        0      x*      450  里程(万公里)
```

**这是为什么？**

-   里程少时（x 很小）：车状况好，维护成本低，不值得大修 → P(维修)≈0
-   里程中等时（x≈x\*）：维修与不维修的成本接近，边际情况 → P(维修)≈0.5
-   里程多时（x 很大）：车状况差，维护成本高，必须大修 → P(维修)≈1

### 3.2 维修阈值$x^*$的经济含义

在维修阈值处，两个选择的价值完全相等：

$$V_0(x^*) = V_1(x^*)$$

即： $$-c(x^*) + \beta V(x^*+1) = -RC + \beta V(0)$$

重新整理： $$RC = c(x^*) - c(0) + \beta[V(0) - V(x^*+1)]$$

**这说明：** 如果我们知道：

1. 维修阈值$x^*$在哪（从 CCP 曲线看）
2. 成本函数$c(x)$的参数
3. 折现因子$\beta$
4. 价值函数的值

就可以**直接计算出维修成本 RC**！

这是 CCP 在**参数识别**中最重要的应用。

---

## 第四章：从数据到理论的三层递进

### 4.1 第一层：非参数 CCP（纯数据，无模型）

**做法：** 直接从数据计算选择比例

给定状态$x$，找出所有处于该状态附近的观测，计算其中选择$d=1$的比例：

$$\hat{P}(d=1|x) = \frac{\sum_i \mathbb{1}(d_i=1, x_i \approx x)}{\sum_i \mathbb{1}(x_i \approx x)}$$

**优点：**

-   完全不需要模型假设
-   直观易懂
-   可以检验模型是否合理

**缺点：**

-   需要足够样本
-   在数据稀疏的状态区间估计方差大

**应用：** 作为"事实"来检验理论模型

### 4.2 第二层：理论 CCP（给定参数）

**做法：**

1. 给定参数（成本函数、折现因子、维修成本）
2. 求解动态规划问题 → 得到价值函数$V(x)$
3. 根据价值差异计算 CCP

$$P(d=1|x) = \frac{1}{1 + \exp(V_0(x) - V_1(x))}$$

其中：

-   $V_0(x)$ = 不维修的价值 = $-c(x) + \beta V(x+1)$
-   $V_1(x)$ = 维修的价值 = $-RC + \beta V(0)$

**关键观察：** 理论 CCP 取决于参数的选择

不同的参数 → 不同的理论 CCP 曲线

### 4.3 第三层：参数估计（反向求解）

**问题：** 哪个参数值能使理论 CCP 最接近数据中的经验 CCP？

**做法：**

1. 猜测一个参数值
2. 计算该参数下的理论 CCP
3. 与数据中的经验 CCP 比较
4. 调整参数，重复直到最接近

这就是**最大似然估计**或**矩估计**

---

## 第五章：CCP 在识别参数中的角色

### 5.1 识别问题的本质

**困境：** 我们想估计维修成本 RC，但它不能直接观测

怎么办？**用 CCP 来反推！**

### 5.2 识别的关键：边界条件

在维修阈值$x^*$处（即 CCP = 0.5 的地方），有一个特殊性质：

两个选择的价值相等，这给了我们一个**等式约束**：

$$V_0(x^*) = V_1(x^*)$$

展开： $$-c(x^*) + \beta V(x^*+1) = -RC + \beta V(0)$$

从这个等式中解出 RC： $$RC = c(x^*) - c(0) + \beta[V(0) - V(x^*+1)]$$

**这说明：** 如果我们知道上面等式右边的所有项，就能唯一确定 RC！

### 5.3 具体识别步骤

**第一步：从数据估计经验 CCP**

-   用非参数方法从原始数据计算
-   得到一条 S 形曲线

**第二步：找维修阈值**

-   从 CCP 曲线找到 P(d=1|x) = 0.5 的点
-   记这个状态为$x^*$

**第三步：固定其他参数**

-   假设我们知道成本函数$c(x)$的参数（从账目数据）
-   假设我们知道折现因子$\beta$（从其他来源）
-   这两个假设很重要！

**第四步：反向求解**

-   计算$c(x^*) - c(0)$
-   用边界条件等式反解 RC

**第五步：验证**

-   用估计的 RC 和其他参数，求解动态规划
-   计算理论 CCP
-   检查理论 CCP 是否与数据的经验 CCP 吻合

### 5.4 为什么说 RC 的识别需要"强假设"

**关键问题：** 边界条件等式中，包含了$V(0)$和$V(x^*+1)$这两个未知的价值函数值。

要计算这两个值，需要：

-   知道成本函数$c(x)$的参数
-   知道折现因子$\beta$
-   知道维修成本 RC（但这正是我们要求的！）

**这里有个循环依赖！**

**解决方法：**

1. **外生固定 β 和 c(x)** → 然后可以唯一解出 RC
2. **或者用价值函数的形状约束** → 从 CCP 的二阶特征反推
3. **或者加入额外数据** → 如二手车价格、维修账单等

这就是为什么说："**RC 的识别需要强假设**"

---

## 第六章：识别的困难与直观理解

### 6.1 为什么 RC 难以识别

**根本原因：** 多组参数可能产生相同的 CCP

**例子：**

情况 A：

-   维修成本 RC = 10（很贵）
-   折现因子 β = 0.95（关心未来）
-   结果：人们倾向延迟维修
-   CCP 曲线：平缓，阈值较高

情况 B：

-   维修成本 RC = 5（便宜）
-   折现因子 β = 0.80（不太关心未来）
-   结果：人们也倾向延迟维修（高成本的人更不舍得修，不关心未来的人也不修）
-   CCP 曲线：也是平缓，阈值也较高

**同一条 CCP 曲线，不同的参数组合都能解释！**

### 6.2 识别强度的直观判断

**CCP 识别 RC 的"识别强度"取决于：**

1. **CCP 的变化幅度**

    - 如果 CCP 从 0 变到 1，变化剧烈 → 识别强
    - 如果 CCP 始终在 0.3 到 0.7 之间 → 识别弱

2. **样本量和数据覆盖**

    - 如果在很多不同状态都有观测 → 识别强
    - 如果观测集中在某个小区间 → 识别弱

3. **模型假设的确定性**
    - 如果 β 和 c(x)完全确定 → 识别相对强
    - 如果这些也不确定 → 识别很弱

### 6.3 "部分识别"的概念

有时候，我们无法唯一确定 RC，但可以得到一个**范围**：

比如，可能推断出：**RC 在 8 到 12 之间**，而不是精确值。

这叫做**部分识别**（Partial Identification）。

---

## 第七章：CCP 与其他参数的替代关系

### 7.1 RC 与 β 的替代

两种方式可以导致相同的维修延迟倾向：

**方式 1：** RC 很高 + β 正常

-   维修很贵，所以宁可忍受高维护成本

**方式 2：** RC 一般 + β 很低

-   维修费用不算太贵，但不关心未来的节省

**结果：** CCP 曲线完全相同

**解决方案：** 必须从**外部信息**固定其中一个，才能识别另一个

通常的做法：

-   从其他研究/数据固定 β（比如利率数据）
-   或固定 RC（比如从维修账单）
-   然后反推另一个

### 7.2 RC 与 c(x)的替代

**高维修成本的世界** vs **高维护成本的世界**

如果：

-   世界 A：维修成本 RC = 10，维护成本 c(x) = 0.1x
-   世界 B：维修成本 RC = 5，维护成本 c(x) = 0.2x

可能产生完全相同的 CCP！

**解决方案：** 必须直接观测维护成本

-   从维修账单数据估计 c(x)
-   或从二手车市场价格推断

---

## 第八章：三种估计方法的对比

### 8.1 非参数 CCP 估计

**方法：** 直接从数据计算选择比例

**步骤：**

1. 将状态空间分成若干箱（或用平滑方法）
2. 在每个箱内，计算选择 d=1 的比例
3. 画出 CCP 曲线

**特点：**

-   不需要任何模型假设
-   快速简单
-   但有样本量要求

### 8.2 半参数 CCP 方法（Hotz-Miller）

**方法：** 用非参数 CCP 反演出价值函数，然后估计参数

**步骤：**

1. 估计经验 CCP（非参数）
2. 从 logit 的逆函数，反推价值差异：$V_0(x) - V_1(x)$
3. 利用贝尔曼方程的递推关系，反演出$V(x)$
4. 用反演的价值函数估计参数

**特点：**

-   相对快速（不用反复求解动态规划）
-   半参数更稳健
-   仍然需要 logit 假设

### 8.3 参数最大似然估计

**方法：** 猜测参数 → 求解模型 → 计算理论 CCP → 与数据比较 → 优化

**步骤：**

1. 给定参数值
2. 求解贝尔曼方程得到$V(x)$
3. 计算理论 CCP
4. 计算似然函数（理论 CCP 与观测选择的匹配度）
5. 调整参数最大化似然

**特点：**

-   最直接（直接拟合原始模型）
-   但计算慢（需要多次求解动态规划）
-   对模型假设敏感

---

## 第九章：实际应用中的 CCP

### 9.1 CCP 的主要应用场景

#### 应用 1：设备维修与更新决策

-   状态：设备年龄或故障率
-   决策：是否进行大修或更新
-   CCP 用途：识别维修成本，优化维修计划

#### 应用 2：汽车购买与维修

-   状态：车龄和里程
-   决策：是否进行大修或换新车
-   CCP 用途：识别消费者对可靠性的价值评估

#### 应用 3：医疗干预决策

-   状态：患者的健康指标
-   决策：是否进行昂贵的医疗程序
-   CCP 用途：评估医疗成本效益

#### 应用 4：环保投资

-   状态：企业的污染排放水平
-   决策：是否投资污染控制
-   CCP 用途：估计污染控制的成本

### 9.2 CCP 在实证分析中的检验作用

即使不关心参数识别，CCP 也很有用：

**检验 1：模型是否合理**

-   计算理论 CCP
-   与经验 CCP 比较
-   如果接近 → 模型假设合理
-   如果偏离 → 需要改进模型

**检验 2：边界条件**

-   CCP 应该从 0 变到 1（如果数据充分）
-   CCP 应该是单调的（至少在总体趋势上）
-   如果出现奇怪的非单调性 → 可能有数据问题或模型误设

**检验 3：稳健性检验**

-   用子样本估计 CCP（如不同时期、不同地区）
-   应该大致相同
-   如果有系统差异 → 可能有遗漏的状态变量

---

## 第十章：核心要点总结

### 核心概念

**CCP** = 给定状态$x$，选择$d=1$的概率 = $P(d=1|x)$

### 三个来源

1. **经验 CCP**：直接从数据计算（非参数）
2. **理论 CCP**：给定参数求解动态规划（参数模型）
3. **私人冲击**：解释同状态下的决策异质性

### 识别逻辑

$$\text{经验CCP} + \text{模型假设} \rightarrow \text{反推参数}$$

关键使用点：

-   在维修阈值$x^*$处（CCP=0.5），价值相等
-   这给了我们一个等式约束
-   利用此约束可以识别成本参数

### 识别强度的决定因素

✓ 强识别的条件：

-   CCP 有剧烈的 S 形变化（从接近 0 到接近 1）
-   样本量足够，数据充分覆盖状态空间
-   其他参数（β, c(x)）外生确定

✗ 弱识别的条件：

-   CCP 始终在中间范围（如 0.3-0.7）
-   样本集中在某个小区间
-   其他参数也不确定

### 最重要的警告

**RC 的识别依赖于以下关键假设：**

1. 折现因子 β 已知或可外生固定
2. 维护成本函数 c(x)的形式已知
3. 私人冲击服从极值分布（logit）
4. 没有遗漏的状态变量
5. 状态测量准确

任何一个假设违反，识别可能就失效。

---

## 快速参考：CCP 的四个关键公式

### 公式 1：经验 CCP（从数据）

$$\hat{P}(d=1|x) = \frac{n_1(x)}{n(x)}$$ 其中$n_1(x)$是状态$x$附近选择$d=1$的观测数，$n(x)$是状态$x$附近的总观测数。

### 公式 2：理论 CCP（从模型）

$$P(d=1|x) = \frac{1}{1 + \exp(V_0(x) - V_1(x))}$$ 其中$V_d(x)$是选择$d$的价值。

### 公式 3：价值函数关系（贝尔曼方程）

$$V_0(x) = -c(x) + \beta V(x+1)$$ $$V_1(x) = -RC + \beta V(0)$$

### 公式 4：识别约束（维修阈值处）

$$V_0(x^*) = V_1(x^*) \Rightarrow RC = [c(x^*) - c(0)] + \beta[V(0) - V(x^*+1)]$$

这些公式构成了从数据到参数估计的完整逻辑链条。
