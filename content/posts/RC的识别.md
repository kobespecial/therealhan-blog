---
date: "2025-12-30T11:25:27+08:00"
draft: false
title: "示例：维修成本 RC 的识别"
toc: true
math: true
slug: RC
categories:
    - 结构方程模型
tags:
    - 结构估计方法
    - 例子
---

通过一个实际的例子来了解结构估计的全过程。

<!--more-->

# RC（维修成本）识别问题：完全指南

---

## 第一章：为什么RC难以识别？

### 1.1 识别问题的本质

**问题的陈述**：

观察到的数据：
- $x_t$：机器/车辆的状态（里程、年龄等）
- $d_t$：是否维修（0或1）

但我们**看不到**：
- $RC$：维修成本（私人信息）
- 个体的时间偏好 $\rho$
- 成本函数参数 $c_1, c_2$

**从选择行为反推参数**是结构估计的核心。

---

### 1.2 识别问题的源头：多重均衡

**例子**：假设观察到"在状态 $x=5$ 时，50% 的车选择维修"

这可能由以下情况引起：
```
情况1：RC很高（10000元）
       → 只有特别积极的人维修
       → CCP = 0.5

情况2：RC很低（1000元）
       → 几乎所有人都愿意维修
       → 但由于其他成本（时间等），CCP = 0.5

情况3：RC = 5000元，但折现率ρ很高
       → 人们对未来不关心
       → 不太愿意维修
       → CCP = 0.5
```

**同一个CCP，多个参数组合都能解释！** 这就是识别问题。

---

### 1.3 识别的层次

```
强识别（Strongly Identified）
  ↑ 参数唯一确定
  │
部分识别（Partially Identified）
  │ 参数落在某个区间内
  │
弱识别（Weakly Identified）
  │ 参数区间很宽
  ↓
完全不识别（Unidentified）
  参数无法从数据确定
```

**RC通常落在"弱识别"到"部分识别"的范围**。

---

## 第二章：识别策略总览

### 2.1 识别工具箱

| 方法 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| **排除限制** | 用外生变量 | 理论清晰 | 需找到合适的IV |
| **功能形式** | 假设成本函数形式 | 增加约束 | 可能有模型误设 |
| **大样本结构** | 用转移概率矩阵 | 稳健 | 计算复杂 |
| **实验数据** | 随机维修干预 | 因果识别强 | 成本高 |
| **替代估计** | 用替代品价格 | 直观 | 需要额外数据 |
| **边界识别** | 用极限情况 | 理论基础强 | 需强假设 |

---

## 第三章：Rust模型的识别理论

### 3.1 标准Rust模型的设定

**模型**：
```
状态：x_t（里程或年龄）
决策：d_t ∈ {0,1}（维修决策）
流收益：u(x,d) = -c(x) if d=0
                 = -RC - c(0) if d=1
折现率：β（或ρ）
```

**CCP（条件选择概率）**：
```
P(d=1|x) = Λ(u(x,1) - u(x,0) + β[V(0) - V(x)])
         = Λ(β V(0) - β V(x) + [-RC - c(0) + c(x)])
         = Λ(-RC + [c(x) - c(0)] + β[V(0) - V(x)])
```

其中 Λ 是logit函数（隐含着离散选择冲击）。

### 3.2 直观识别论证（Hotz-Miller, 1993）

**关键洞察**：不同参数组合能否产生相同的CCP？

#### 参考状态与相对参数化

定义"**维修边界**" $x^*$：在这个状态，维修和不维修无差异。

在 $x = x^*$ 处：
```
P(d=1|x^*) = 0.5

即：u(x^*,1) + βV(0) = u(x^*,0) + βV(x^*)
    -RC - c(0) + βV(0) = -c(x^*) + βV(x^*)
    RC = [c(x^*) - c(0)] + β[V(x^*) - V(0)]
```

**这个方程唯一确定RC！** 如果我们知道：
1. $x^*$ 的值（从数据）
2. $c(\cdot)$ 的参数
3. $\beta$ 和 $V(\cdot)$ 的值

### 3.3 识别的必要条件

**Rust(1987)的充分识别条件**：

假设：
1. **成本函数已知**：$c(x) = c_1 x + c_2 x^2$（参数已知或已估计）
2. **折现率已知**：$\beta$ 给定（如 $\beta = 0.95$）
3. **私人冲击是Type-I EV分布**（标准假设）
4. **状态转移确定**：$x_{t+1} = x_t + 1 + \epsilon_t$（新车里程增长）

**在这些条件下**：从CCP的形状和位置可以唯一识别 $RC$。

---

## 第四章：识别方法详解

### 4.1 方法1：边界条件识别法（Rust's Approach）

**核心思想**：利用维修阈值处的边界条件。

#### Step 1：从数据估计CCP

```python
def estimate_ccp_empirical(data, x_grid):
    """
    从数据中非参数估计CCP
    用局部多项式回归或kernel方法
    """
    from scipy.ndimage import uniform_filter1d

    # 对每个x值，计算该状态下的平均维修率
    ccp = np.zeros_like(x_grid)

    for i, x in enumerate(x_grid):
        # 找状态在x附近的观测
        neighbors = np.abs(data['x'] - x) <= 0.5

        if np.sum(neighbors) > 0:
            ccp[i] = np.mean(data['d'][neighbors])

    # 平滑化（局部多项式回归）
    from statsmodels.nonparametric.smoothers_lowess import lowess

    smoothed = lowess(ccp, x_grid, frac=0.2)

    return smoothed[:, 1]  # 返回平滑的CCP
```

#### Step 2：找维修阈值 $x^*$

```python
def find_maintenance_threshold(x_grid, ccp):
    """
    找到CCP = 0.5的状态值（维修阈值）
    """
    # 找CCP最接近0.5的x值
    idx = np.argmin(np.abs(ccp - 0.5))
    x_star = x_grid[idx]

    return x_star, idx
```

#### Step 3：用一阶条件反推RC

**理论**：在 $x = x^*$ 处，维修决策无差异：

```
Λ(...) = 0.5  （logit的反函数在0处）
⟹ u(x^*,1) - u(x^*,0) + β[V(0) - V(x^*)] = 0
⟹ -RC - c(0) + c(x^*) + β[V(0) - V(x^*)] = 0
⟹ RC = c(x^*) - c(0) + β[V(0) - V(x^*)]
```

**计算**：
```python
def identify_RC_boundary_method(x_star, params, V, beta=0.95):
    """
    用边界条件识别RC

    参数：
    x_star: 维修阈值
    params: (c1, c2) 成本函数参数
    V: 价值函数
    beta: 折现因子
    """
    c1, c2 = params

    # 成本函数
    c_x_star = c1 * x_star + c2 * x_star**2
    c_0 = 0  # c(0) = 0

    # 价值函数在阈值处的值
    idx_star = np.argmin(np.abs(x_grid - x_star))
    V_x_star = V[idx_star]
    V_0 = V[0]

    # 边界条件：无差异
    RC = (c_x_star - c_0) + beta * (V_x_star - V_0)

    return RC
```

#### Step 4：完整算法

```python
class RCIdentificationBoundaryMethod:
    """Rust (1987)的边界识别法"""

    def __init__(self, data, x_grid):
        self.data = data
        self.x_grid = x_grid

    def estimate(self, c_params, beta=0.95, assumed_rho=0.05):
        """
        识别RC

        假设：
        - c_params已知
        - beta已知
        """

        # Step 1: 非参数估计CCP
        print("Step 1: 从数据估计CCP...")
        ccp_empirical = self.estimate_ccp_empirical()

        # Step 2: 找维修阈值
        print("Step 2: 找维修阈值...")
        x_star, idx_star = self.find_maintenance_threshold(ccp_empirical)
        print(f"  维修阈值 x* = {x_star:.2f}")

        # Step 3: 需要求解HJB得到V(x)
        # （给定c_params和beta，但RC未知）
        print("Step 3: 求解含未知RC的HJB方程...")

        # 这里困难：HJB依赖于RC，但RC正是我们要找的！
        # 需要迭代或反向求解

        # Step 4: 反向求解RC
        print("Step 4: 从边界条件反向求解RC...")

        # 用迭代法：
        # (a) 猜测RC_0
        # (b) 解HJB得到V(x; RC_0)
        # (c) 检查边界条件是否满足
        # (d) 调整RC_0，重复

        RC = self.iterative_RC_identification(
            c_params, beta, x_star, ccp_empirical
        )

        return RC, x_star, ccp_empirical

    def iterative_RC_identification(self, c_params, beta, x_star, ccp_empirical):
        """
        迭代求解RC
        """

        # 初值猜测
        RC = 10.0
        tol = 1e-6

        for iteration in range(100):
            RC_old = RC

            # (1) 给定RC，解HJB
            V = self.solve_hjb(c_params, beta, RC)

            # (2) 计算边界条件应该满足的RC值
            c1, c2 = c_params
            c_x_star = c1 * x_star + c2 * x_star**2
            c_0 = 0

            idx_star = np.argmin(np.abs(self.x_grid - x_star))
            V_x_star = V[idx_star]
            V_0 = V[0]

            # 边界条件给出的RC
            RC_implied = (c_x_star - c_0) + beta * (V_x_star - V_0)

            # (3) 更新：平均当前值和隐含值
            RC = 0.5 * RC + 0.5 * RC_implied

            # 检查收敛
            if np.abs(RC - RC_old) < tol:
                print(f"  收敛于第{iteration+1}次迭代")
                break

        return RC

    def solve_hjb(self, c_params, beta, RC):
        """求解HJB方程"""
        # 与之前相同的HJB求解器...
        pass

    def estimate_ccp_empirical(self):
        """非参数估计CCP"""
        from statsmodels.nonparametric.smoothers_lowess import lowess

        # 分bin计算CCP
        bins = np.linspace(self.x_grid.min(), self.x_grid.max(), 20)
        bin_indices = np.digitize(self.data['x'], bins)

        ccp = np.zeros_like(self.x_grid)
        for i, x in enumerate(self.x_grid):
            bin_idx = np.digitize(x, bins)
            in_bin = bin_indices == bin_idx

            if np.sum(in_bin) > 0:
                ccp[i] = np.mean(self.data['d'][in_bin])

        # 平滑化
        smoothed = lowess(ccp, self.x_grid, frac=0.3)

        return smoothed[:, 1]

    def find_maintenance_threshold(self, ccp):
        """找维修阈值"""
        idx = np.argmin(np.abs(ccp - 0.5))
        x_star = self.x_grid[idx]

        return x_star, idx

# 使用示例
est = RCIdentificationBoundaryMethod(data, x_grid)
RC_identified, x_star, ccp = est.estimate(
    c_params=(0.005, 0.0001),
    beta=0.95
)

print(f"\n识别结果：")
print(f"  RC = {RC_identified:.2f}")
print(f"  阈值 x* = {x_star:.2f}")
```

### 4.2 方法2：排除限制识别法（IV方法）

**思路**：找一个变量 $Z$，它影响状态转移但不直接影响当期效用。

#### 例子：汽车维修中的季节性

**观察**：
- 冬季的路面条件差（更多磨损），增加了需要维修的概率
- 但冬季本身不改变维修成本

```python
def identification_with_exclusion_restriction(data):
    """
    用排除限制识别RC

    假设：季节性S只通过状态转移影响决策，
    不直接影响维修成本
    """

    # 数据：(x_t, d_t, s_t, x_{t+1})
    # 其中s_t是季节虚拟变量

    # 第一步：估计条件成本函数 c(x)
    # 用OLS回归：维修后的成本 ~ 状态
    maintenance_costs = data[data['d']==1]['cost']
    mileage_at_maintenance = data[data['d']==1]['x']

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(mileage_at_maintenance.values.reshape(-1,1))

    model = LinearRegression()
    model.fit(X_poly, maintenance_costs)

    # c(x) = model的系数
    c_params = model.coef_

    # 第二步：估计状态转移过程
    # 用IV：利用季节性预测状态增长

    # 状态增长方程：
    # Δx_{t+1} = α + β_0 · s_t + β_1 · d_t + ε_t

    import statsmodels.api as sm

    # 构造变量
    delta_x = data['x_next'] - data['x']
    X = sm.add_constant(data[['season', 'd']])

    # OLS
    model_state = sm.OLS(delta_x, X).fit()
    print(model_state.summary())

    # 第三步：估计价值函数（不需知道RC）
    # 从CCP的形状...

    # 第四步：识别RC
    # 通过反演CCP = 0.5的条件
```

### 4.3 方法3：替代品定价识别法

**思路**：直接观测市场数据中的RC。

#### 例子：二手车市场

如果有二手车的交易数据，可以看：
- 同样里程的车，维修状态好的与坏的价格差
- 这个差价反映了维修的价值 = RC + 未来成本节省

```python
def identify_RC_from_secondhand_market(car_data):
    """
    用二手车市场价格识别RC

    假设：维修历史在二手车市场中可见
    """

    # 数据：二手车价格 P, 状态质量 quality, 里程 x, 维修历史 repair_history

    # 对照组：同样状态、不同维修历史
    high_maintenance = car_data[car_data['repairs'] > 2]['price']
    low_maintenance = car_data[car_data['repairs'] <= 2]['price']

    # 匹配控制变量（里程、年龄等）
    from causalml.match import NearestNeighborMatch

    X = car_data[['x', 'age', 'brand']].values
    y = car_data['price'].values
    treatment = (car_data['repairs'] > 2).astype(int).values

    # 倾向分数匹配
    psm = NearestNeighborMatch(X, treatment, y)
    ate = psm.estimate_ate()

    # ATE = 维修历史好的车价格溢价
    # 这反映RC和维护成本的未来价值

    # 用动态规划模型，反演出RC
    estimated_RC = invert_from_price_premium(ate, other_params)

    return estimated_RC
```

### 4.4 方法4：实验识别法

**最强的因果识别：随机实验**

#### 设计：随机维修补贴

```python
def experimental_RC_identification():
    """
    实验设计：随机对一些车/企业提供维修补贴

    因果效应 = E[d | subsidy=1] - E[d | subsidy=0]

    这直接反映对RC改变的反应
    """

    # 分组
    treatment_group = data[data['subsidy']==1]
    control_group = data[data['subsidy']==0]

    # 维修率变化
    maintenance_rate_treatment = treatment_group['d'].mean()
    maintenance_rate_control = control_group['d'].mean()

    ATE = maintenance_rate_treatment - maintenance_rate_control

    # 用结构模型：给定ATE，反推RC变化的幅度
    # Δ RC = f(ATE, other parameters)

    # 原始决策条件：
    # P(d=1) = Λ(utility_with_RC)

    # 补贴后：
    # P(d=1) = Λ(utility_with_reduced_RC)

    # 反演得到RC的减少量

    return estimate_RC_from_ate(ATE, baseline_params)
```

---

## 第五章：RC识别的统计推断

### 5.1 识别强度的度量

#### 度量1：固有差异化（Inherent Variation）

```python
def compute_identification_strength(x_grid, ccp, c_params, beta):
    """
    衡量CCP对参数的敏感性
    """

    # 对RC的敏感性：
    # ∂CCP/∂RC 有多大？

    # 数值微分
    eps = 0.01

    RC_baseline = 10.0

    # 求解两个HJB
    V_baseline = solve_hjb(c_params, beta, RC_baseline)
    V_perturbed = solve_hjb(c_params, beta, RC_baseline + eps)

    # CCP的变化
    ccp_baseline = compute_ccp(V_baseline, c_params)
    ccp_perturbed = compute_ccp(V_perturbed, c_params)

    # 敏感性（数值导数）
    sensitivity = np.mean(np.abs(ccp_perturbed - ccp_baseline) / eps)

    # 如果sensitivity很小 → RC识别弱
    # 如果sensitivity很大 → RC识别强

    print(f"识别强度（CCP对RC的敏感性）: {sensitivity:.6f}")

    return sensitivity
```

#### 度量2：信息矩阵特征值

```python
def compute_fisher_information(theta_true, data):
    """
    计算Fisher信息矩阵
    特征值小 → 识别弱
    """

    def log_likelihood_gradient(theta):
        """L对θ的梯度"""
        return numerical_gradient(log_likelihood, theta, eps=1e-6)

    def outer_product(theta):
        """梯度的外积 g·g'"""
        g = log_likelihood_gradient(theta)
        return np.outer(g, g)

    # 在很多点计算并平均
    fisher = np.zeros((len(theta_true), len(theta_true)))

    for _ in range(100):
        # 从参数分布采样
        theta_sample = theta_true + np.random.normal(0, 0.01, len(theta_true))
        fisher += outer_product(theta_sample)

    fisher /= 100

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(fisher)

    print("Fisher信息矩阵特征值:")
    for i, ev in enumerate(sorted(eigenvalues, reverse=True)):
        print(f"  λ_{i+1} = {ev:.6f}")

    # 条件数 = λ_max / λ_min
    condition_number = eigenvalues.max() / eigenvalues.min()
    print(f"条件数: {condition_number:.2f}")
    print(f"  条件数 > 100 → 识别弱")

    return eigenvalues, eigenvectors
```

### 5.2 识别域：参数的支撑

有时参数无法唯一识别，但可以得到**识别域**（identified set）。

```python
def compute_identified_set(data, x_grid, c_params_range, beta_range):
    """
    计算RC的识别域

    通过网格搜索：哪些(RC, beta)组合能很好地拟合数据？
    """

    rc_values = np.linspace(1, 50, 50)
    beta_values = np.linspace(0.90, 0.99, 20)

    likelihood_matrix = np.zeros((len(rc_values), len(beta_values)))

    for i, rc in enumerate(rc_values):
        for j, beta in enumerate(beta_values):

            # 给定(RC, beta)，求解HJB并计算似然
            V = solve_hjb(c_params, beta, rc)
            ccp = compute_ccp(V, c_params)

            ll = compute_log_likelihood(ccp, data)
            likelihood_matrix[i, j] = ll

    # 绘制
    plt.contour(beta_values, rc_values, likelihood_matrix)
    plt.xlabel('折现因子 β')
    plt.ylabel('维修成本 RC')
    plt.title('对数似然热力图')

    # 识别域：似然在最大值的95%范围内
    ll_max = likelihood_matrix.max()
    ll_threshold = ll_max - 1.92  # 95% 置信区间（χ²(2) / 2）

    identified_set = likelihood_matrix > ll_threshold

    plt.contourf(beta_values, rc_values, identified_set.astype(int),
                 levels=[0.5, 1.5], colors=['lightblue'])
    plt.title('RC的识别域（95%）')
    plt.show()

    return identified_set, likelihood_matrix
```

---

## 第六章：具体案例：Harold Zurcher数据

### 6.1 数据背景

**Rust (1987)的著名数据集**：
- 140辆公交车
- 每月里程和维修记录
- 时间跨度：1974-1985年

```python
def load_zurcher_data():
    """加载Zurcher数据"""

    # 下载或导入数据...
    data = pd.DataFrame({
        'bus_id': [...],
        'date': [...],
        'mileage': [...],  # 里程表读数（每10000里为1个单位）
        'repaired': [...],  # 是否大修（1=是）
    })

    # 创建面板结构
    data = data.sort_values(['bus_id', 'date'])
    data['mileage_lag'] = data.groupby('bus_id')['mileage'].shift(1)
    data['mileage_change'] = data['mileage'] - data['mileage_lag']

    return data

# 描述统计
data = load_zurcher_data()
print(data.describe())

# 维修率随里程的分布
repair_rate_by_mileage = data.groupby(pd.cut(data['mileage'], 20))['repaired'].mean()
print(repair_rate_by_mileage)
```

### 6.2 Rust原文的识别策略

**Rust (1987)的做法**：

1. **假设折现率**：$\beta = 0.999$（几乎无折现）
   - 理由：对于公司决策，短期非常重要

2. **假设维护成本函数**：$c(x) = c_1 \cdot \frac{x}{12}$（线性）
   - 理由：简单且合理

3. **估计参数**：从CCP的形状估计RC和$c_1$

### 6.3 识别过程详解

```python
class ZurcherDataAnalysis:
    """
    Rust (1987)的识别步骤
    """

    def __init__(self, data):
        self.data = data
        self.x_grid = np.linspace(0, 450, 150)  # 里程网格

    def step1_estimate_ccp_nonparametric(self):
        """
        Step 1: 从原始数据非参数估计CCP
        这不需要任何模型假设！
        """

        # 用局部多项式回归
        from statsmodels.nonparametric.smoothers_lowess import lowess

        # 对每个里程值，计算该里程附近的维修率
        x_data = self.data['mileage'].values
        d_data = self.data['repaired'].values

        # LOWESS平滑
        smoothed = lowess(d_data, x_data, frac=0.3)

        # 插值到网格
        self.ccp_empirical = np.interp(
            self.x_grid,
            smoothed[:, 0],
            smoothed[:, 1]
        )

        # 绘制
        plt.figure(figsize=(12, 5))
        plt.scatter(x_data, d_data, alpha=0.1, s=10)
        plt.plot(self.x_grid, self.ccp_empirical, 'r-', linewidth=2,
                label='非参数CCP估计')
        plt.xlabel('里程 (单位: 10,000 miles)')
        plt.ylabel('维修概率')
        plt.title('条件选择概率（CCP）')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return self.ccp_empirical

    def step2_specify_maintenance_cost(self, c1_assumed=0.2367):
        """
        Step 2: 假设维护成本函数形式

        Rust (1987)假设: c(x) = c1 · x
        其中c1是参数（未知）

        这里我们"假设" c1已知，用Rust原文的值
        实际应该估计c1...
        """

        self.c1_assumed = c1_assumed
        self.c_func = lambda x: c1_assumed * x

        print(f"维护成本函数: c(x) = {c1_assumed} · x")

    def step3_set_discount_factor(self, beta_assumed=0.9999):
        """
        Step 3: 假设折现因子

        Rust (1987)假设: β ≈ 0.9999（几乎无折现）
        """

        self.beta_assumed = beta_assumed
        print(f"折现因子: β = {beta_assumed}")

    def step4_find_maintenance_threshold(self):
        """
        Step 4: 从CCP数据找维修阈值
        """

        # CCP = 0.5的里程值
        idx_threshold = np.argmin(np.abs(self.ccp_empirical - 0.5))
        self.x_star = self.x_grid[idx_threshold]

        print(f"\n维修阈值 x* = {self.x_star:.2f}")
        print(f"  在此里程，维修概率 = {self.ccp_empirical[idx_threshold]:.3f}")

    def step5_identify_RC_iteratively(self):
        """
        Step 5: 迭代识别RC

        关键：给定c_params和β，用边界条件反推RC
        """

        print("\n迭代识别RC...")

        # 初值
        RC = 5.0
        tol = 1e-6

        for iteration in range(50):
            RC_old = RC

            # (a) 给定RC、β、c_params，解连续HJB方程
            V = self.solve_hjb_continuous(RC)

            # (b) 检查边界条件
            # 在x = x*处，应满足：
            # c(x*) - RC + β[V(x*) - V(0)] = 0

            idx_star = np.argmin(np.abs(self.x_grid - self.x_star))

            c_x_star = self.c_func(self.x_star)
            c_0 = self.c_func(0)

            V_x_star = V[idx_star]
            V_0 = V[0]

            # 隐含的RC
            RC_implied = c_x_star - c_0 + self.beta_assumed * (V_0 - V_x_star)

            # 更新（阻尼迭代避免振荡）
            RC = 0.7 * RC + 0.3 * RC_implied

            error = np.abs(RC - RC_old)

            if (iteration + 1) % 10 == 0:
                print(f"  迭代{iteration+1}: RC = {RC:.4f}, 误差 = {error:.2e}")

            if error < tol:
                print(f"  收敛！")
                break

        self.RC_identified = RC

        return RC

    def solve_hjb_continuous(self, RC):
        """
        求解连续时间HJB方程（月度离散化）
        """

        # 状态动力学：每月增加约0.1万里（给定/预测）
        delta_t = 1/12  # 一个月
        x_growth_per_month = 0.1  # 万里/月

        # 初始化
        V = np.zeros_like(self.x_grid)

        # 反向迭代（从大到小）
        for iteration in range(100):
            V_old = V.copy()

            for i in range(len(self.x_grid) - 1, -1, -1):
                x = self.x_grid[i]

                # 不维修的选择价值
                flow0 = -self.c_func(x)
                x_next_0 = min(x + x_growth_per_month, self.x_grid[-1])
                V_next_0 = np.interp(x_next_0, self.x_grid, V_old)

                bar_v0 = delta_t * flow0 + (1 - 0.12/12) * V_next_0
                # β ≈ 1 - ρ·dt，这里ρ ≈ 0.12

                # 维修的选择价值
                flow1 = -RC
                x_next_1 = 0
                V_next_1 = V_old[0]

                bar_v1 = delta_t * flow1 + (1 - 0.12/12) * V_next_1

                # 用logit结合（处理离散选择冲击）
                # V(x) = (1/ρ) · log(exp(ρ · bar_v0) + exp(ρ · bar_v1))
                # 这里ρ是logit的scale parameter（与折现ρ无关，容易混淆）

                scale = 1000  # logit scale
                V[i] = (1 / scale) * np.log(
                    np.exp(scale * bar_v0) + np.exp(scale * bar_v1)
                )

            # 检查收敛
            if np.max(np.abs(V - V_old)) < 1e-6:
                break

        return V

    def run_full_identification(self):
        """运行完整识别流程"""

        print("="*60)
        print("Rust (1987) 识别步骤")
        print("="*60)

        # Step 1
        self.step1_estimate_ccp_nonparametric()

        # Step 2
        self.step2_specify_maintenance_cost(c1_assumed=0.2367)

        # Step 3
        self.step3_set_discount_factor(beta_assumed=0.9999)

        # Step 4
        self.step4_find_maintenance_threshold()

        # Step 5
        RC = self.step5_identify_RC_iteratively()

        print("\n" + "="*60)
        print(f"最终识别结果：RC = {RC:.4f}")
        print("="*60)

        # 与Rust原文结果比较
        rust_original_RC = 11.7258
        print(f"\n与Rust (1987)原文的对比：")
        print(f"  原文RC:        {rust_original_RC:.4f}")
        print(f"  识别得到RC:    {RC:.4f}")
        print(f"  差异:          {np.abs(RC - rust_original_RC):.4f}")

        return RC

# 执行
analysis = ZurcherDataAnalysis(data)
RC_identified = analysis.run_full_identification()
```

---

## 第七章：RC识别困难的原因与解决方案

### 7.1 为什么RC容易与其他参数混淆？

#### 问题1：RC与折现率β的替代关系

```
情况A：RC = 10, β = 0.95
  → 人们不太愿意维修（高成本，不关心未来）

情况B：RC = 5, β = 0.90
  → 人们同样不太愿意维修（低成本，但更不关心未来）
```

**解决方案**：
- ✓ 从外部信息固定 β（如调查、实验）
- ✓ 或用具有不同β的个体的异质性

#### 问题2：RC与维护成本函数c(x)的混淆

```
情况A：RC = 10, c(x) = 0.1x
  → 高维修成本，低运营成本
  → 选择延迟维修

情况B：RC = 5, c(x) = 0.2x
  → 低维修成本，高运营成本
  → 同样的维修决策
```

**解决方案**：
- ✓ 直接观测c(x)（如维修账单）
- ✓ 从二手市场价格差异识别

#### 问题3：RC与私人冲击分布的混淆

```
情况A：RC = 10, 冲击分布scale很大
  → 冲击掩盖了成本的影响
  → CCP很平坦

情况B：RC = 5, 冲击分布scale很小
  → 成本差异清晰表现
  → CCP很陡峭
```

**解决方案**：
- ✓ 增大样本量（让冲击平均化）
- ✓ 用多个时期的面板数据

### 7.2 识别强化的实用建议

```python
def checklist_for_RC_identification():
    """RC识别的检查清单"""

    checklist = {
        "数据质量": {
            "样本量": "N > 100?",
            "时间跨度": "T > 24 periods?",
            "维修率": "15% < repair_rate < 85%?",
        },
        "参数固定": {
            "折现率β": "从外部来源固定?",
            "成本函数c(x)": "从账目数据估计?",
            "冲击分布": "Type-I EV合理?",
        },
        "数据特征": {
            "CCP变化": "CCP从0到1有充分变化?",
            "状态分布": "状态空间充分覆盖?",
            "决策多样性": "维修决策在多个状态出现?",
        },
        "稳健性检验": {
            "敏感性": "RC估计对先验假设敏感?",
            "子样本": "不同车型/时期的RC一致?",
            "替代模型": "非参数结果与参数模型一致?",
        },
    }

    return checklist
```

---

## 第八章：前沿方法

### 8.1 机器学习辅助识别

**idea**：用神经网络学习CCP和价值函数，而不显式求解HJB

```python
import torch
import torch.nn as nn
from torch.optim import Adam

class NeuralRCIdentification(nn.Module):
    """用神经网络辅助识别RC"""

    def __init__(self, hidden_dim=64):
        super().__init__()

        # CCP网络：输入state，输出维修概率
        self.ccp_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 概率在[0,1]
        )

        # 价值函数网络
        self.v_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 参数
        self.RC = nn.Parameter(torch.tensor([10.0]))
        self.c1 = nn.Parameter(torch.tensor([0.1]))
        self.c2 = nn.Parameter(torch.tensor([0.01])))
        self.rho = nn.Parameter(torch.tensor([0.05]))

    def forward(self, x):
        """计算CCP"""
        return self.ccp_net(x)

    def hjb_loss(self, x_batch):
        """
        HJB方程的残差损失

        约束：∀x,
        ρV(x) = max_a {u(x,a) + V'(x)·f(x,a)}
        """

        x = x_batch.clone().requires_grad_(True)

        # 价值函数和导数
        V = self.v_net(x)
        dV_dx = torch.autograd.grad(
            V.sum(), x, create_graph=True
        )[0]

        # CCP
        p = self.forward(x)

        # 不维修：流收益 + 继续价值
        u0 = -(self.c1 * x + self.c2 * x**2)
        # 状态转移：dx/dt = 0.1（固定的）
        bar_v0 = u0 + dV_dx * 0.1

        # 维修：流收益 + 新价值
        u1 = -self.RC - (self.c1 * 0 + self.c2 * 0**2)
        x_after_repair = torch.zeros_like(x)
        V_after_repair = self.v_net(x_after_repair)
        bar_v1 = u1 + 0 + V_after_repair  # dx = 0 after repair

        # HJB残差
        # ρV - max{ū0, ū1} = 0
        # 用logit形式：
        # ρV ≈ log(exp(bar_v0) + exp(bar_v1)) （有scale factor）

        scale = 1000
        expected_value = (1 / scale) * torch.log(
            torch.exp(scale * bar_v0) + torch.exp(scale * bar_v1)
        )

        hjb_residual = self.rho * V - expected_value

        return torch.mean(hjb_residual ** 2)

    def likelihood_loss(self, x_data, d_data):
        """
        对数似然损失
        """
        p = self.forward(x_data)

        # 二元交叉熵
        return nn.functional.binary_cross_entropy(
            p.squeeze(), d_data
        )

    def total_loss(self, x_data, d_data, x_hjb, lambda_hjb=1.0):
        """
        总损失 = 似然 + λ·HJB残差
        """
        ll_loss = self.likelihood_loss(x_data, d_data)
        hjb_loss_val = self.hjb_loss(x_hjb)

        return ll_loss + lambda_hjb * hjb_loss_val

# 训练
def train_neural_rc_identification(data, epochs=1000):

    model = NeuralRCIdentification(hidden_dim=64)
    optimizer = Adam(model.parameters(), lr=0.01)

    # 准备数据
    x_data = torch.tensor(data['x'].values, dtype=torch.float32).unsqueeze(1)
    d_data = torch.tensor(data['d'].values, dtype=torch.float32)

    # HJB点（网格）
    x_hjb = torch.linspace(0, 450, 100).unsqueeze(1)
    x_hjb.requires_grad_(True)

    for epoch in range(epochs):

        loss = model.total_loss(x_data, d_data, x_hjb, lambda_hjb=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.6f}")
            print(f"  RC = {model.RC.item():.4f}")
            print(f"  c1 = {model.c1.item():.6f}, c2 = {model.c2.item():.8f}")
            print(f"  ρ = {model.rho.item():.6f}")

    return model

# 使用
model = train_neural_rc_identification(data)
print(f"\n最终识别：RC = {model.RC.item():.4f}")
```

### 8.2 贝叶斯识别方法

```python
def bayesian_rc_identification(data, prior_RC_mean=10, prior_RC_std=5):
    """
    贝叶斯方法：考虑参数的先验分布

    可以：
    1. 合并专家意见（先验）
    2. 量化不确定性（后验分布）
    3. 处理部分识别（credible set）
    """

    import pymc as pm

    with pm.Model() as model:

        # 先验分布
        RC = pm.Normal('RC', mu=prior_RC_mean, sigma=prior_RC_std)
        c1 = pm.Normal('c1', mu=0.2, sigma=0.1)
        rho = pm.Exponential('rho', lam=1/0.05)

        # 给定参数，计算观测的概率
        ccp = compute_ccp_symbolic(self.x_grid, RC, c1, rho)

        # 似然（二项分布）
        y = pm.Bernoulli('y', p=ccp, observed=data['d'].values)

        # MCMC采样
        trace = pm.sample(2000, tune=1000, cores=4, return_inferencedata=True)

    # 后验分布
    az.plot_posterior(trace, var_names=['RC', 'c1', 'rho'])

    # 后验均值和可信区间
    rc_posterior_mean = trace.posterior['RC'].mean().values
    rc_credible_interval = az.hdi(trace, var_names=['RC'])

    print(f"RC的后验均值：{rc_posterior_mean:.4f}")
    print(f"RC的95%可信区间：{rc_credible_interval}")

    return trace
```

---

## 总结：RC识别的完整框架

### 核心思路

| 步骤 | 操作 | 数据要求 |
|------|------|--------|
| 1 | 非参数估计CCP | 足够的观测 |
| 2 | 固定外生参数（β, c(x)） | 外部来源/假设 |
| 3 | 找维修阈值 x* | CCP数据 |
| 4 | 用HJB边界条件反推RC | 模型框架 |
| 5 | 迭代直到收敛 | 数值稳定性 |

### 识别强弱的决定因素

```
强识别：
  ✓ 大样本（N > 500）
  ✓ 长时间序列（T > 60）
  ✓ CCP有清晰的S形曲线
  ✓ 维修决策在多个状态出现
  ✓ 外生参数精确已知

弱识别：
  ✗ 小样本
  ✗ 维修率过高或过低（<10%或>90%）
  ✗ CCP接近0或1（没有变化）
  ✗ 维修主要在某一个状态区间
  ✗ 外生参数不确定
```

### 实用建议

1. **永远从非参数CCP开始**
   - 不依赖任何模型假设
   - 可以直观看出数据特征

2. **小心处理边界条件**
   - x=0附近（新产品）和x=max（报废）
   - 可能有特殊的决策规则

3. **进行敏感性分析**
   - 改变β的值，RC如何变化？
   - 改变c(x)的形式，RC如何变化？

4. **与其他方法交叉验证**
   - 用实验数据（如果有）
   - 用二手市场价格
   - 用会计账目中的实际维修成本

5. **报告识别域而非点估计**
   - 诚实地量化不确定性
   - 给出RC的可能范围

---

**最终答案**：RC可以识别，但**需要强假设** （如固定β和c(x)），或**需要额外数据** （实验、价格、账目）。在实践中，通常采用**混合方法**：从外部来源固定部分参数，其余用结构估计识别。