# 10.30笔记

## 赞卓里奇

两卷凌空起，卓然万里奇。
几何连代拓，一式统微积。

## 赞李植

万象连介里，值此少年时。
张量森罗密，不过基并矢。

## 应变张量的主轴和主分量

选取$\mathring{\tilde{\epsilon}}$的主坐标系为初始构形的$L$氏坐标，有

$$
\mathring{\tilde{\epsilon}} 
= \sum_i \mathring{\epsilon_i} \mathring{\vec{e}^i} \mathring{\vec{e}^i} 
= \begin{pmatrix} 
\mathring{\epsilon_1}&0&0\\
0&\mathring{\epsilon_2}&0\\
0&0&\mathring{\epsilon_3}
\end{pmatrix}
$$

$$
\tilde{\epsilon} 
= \sum_i \mathring{\epsilon_i} \vec{e}^i \vec{e}^i 
= \sum_i \mathring{\epsilon_i} g^{ii} \frac{\vec{e}^i}{|\vec{e}^i|} \frac{\vec{e}^i}{|\vec{e}^i|}
$$

选取了$\mathring{\tilde\epsilon}$的主坐标系为$L$氏坐标后，由于$\mathring{\tilde\epsilon}$没有非对角分量，这即说明原来正交的该$L$氏坐标轴在变形后保持正交，由此得到

$$
g^{ii}=\frac {1} {g_{ii}}=(1+l_i)^{-2}
$$

进而有

$$
\epsilon_i
= \mathring{\epsilon_i}g^{ii}
= \mathring{\epsilon_i} \frac{1}{g_{ii}}
= \frac 1 2 \cdot ((1+l_i)^2-1) \cdot (1+l_i)^{-2}
\\
= \frac{\mathring{\epsilon_i}}{1+2\mathring{\epsilon}_i}
$$

由此我们有纯变形的概念，即对于任意的变形，在选取好了应变张量的主轴方向作为主坐标系后，在忽略旋转的意义下，我们可以将这个变形看作仅仅沿着$\mathring{\tilde \epsilon}$主轴方向的长度变化。

## 变形梯度张量

我们希望利用一个张量描述两个切空间之间的变换关系，为此考虑引入变形梯度张量，从数学上看，这应当表现为

$$
dr=F(d\mathring{r})
$$

从线性空间的角度分析，$d\mathring{r}$为原线性空间（切空间）的向量，$dr$则为变形后线性空间（切空间）的对应映射得到的像，F表明某种线性变换。
利用张量的语言，这一数学概念可以表述为

$$
d\vec{r} = \tilde F \cdot d\mathring{\vec r}
$$

$F$即称为变形梯度张量，下面推导$F$的具体表达形式：

注意到

$$
d\mathring{\vec r}=\mathring{\vec e_i} d\xi^i \quad 
d\vec r = \vec e_i d\xi^i
$$

不难看出取

$$
\tilde F = \vec e_i\mathring{\vec e^i}
$$

即有$d\vec{r} = \tilde F \cdot d\mathring{\vec r}$，这即为变形梯度张量的定义。

接下来尝试用变形梯度张量表出应变张量

$$
\tilde F^T \cdot \tilde F=\mathring {\vec e^i} \vec e_i \cdot \vec e_j \mathring{\vec e^j}
$$

注意到$\vec r = \mathring{\vec r} +\vec w$，作用$\nabla_i$求得$\vec e_i=\mathring{\vec e_i}+\nabla_i\vec w$。

$$
\tilde F^T \cdot \tilde F=\mathring {\vec e^i} \vec e_i \cdot \vec e_j \mathring{\vec e^j}
\\
=\mathring {\vec e^i} (\mathring{\vec e_i}+\nabla_i\vec w) \cdot (\mathring{\vec e_j}+\nabla_j\vec w) \mathring{\vec e^j}
\\
=(\tilde g + \mathring \nabla \vec w)
\cdot
(\tilde g + (\mathring \nabla \vec w)^T)
\\
=\tilde g + 2\mathring {\tilde \epsilon}
$$

故

$$
\mathring{\tilde \epsilon} = \frac 1 2 (\tilde F^T \cdot \tilde F-\tilde g)
$$

类似的可以得到

$$
\tilde \epsilon = \frac 1 2 (\tilde g - {\tilde F^{-1}}^T \cdot \tilde F^{-1})
$$

## 关于度量张量$\tilde g$和$\mathring{\tilde g}$

事实上我们已经利用过了$\tilde g=\mathring{\tilde g}$的结论，这里对这个结论做一点说明。首先我们已经证明过，度量张量$\tilde g$确实是一个张量，张量最为重要的性质就是作为一个整体是一个坐标变换下的不变量，而$\tilde g$到$\mathring{\tilde g}$事实上就是做了一个坐标变换，两个坐标下的的度量张量当然是相等的，尽管他们按各自基矢量的展开系数不同。进一步的，对于所有的坐标系，他们都有着同样的度量张量，尽管各种坐标系下的基矢量展开系数不同。也有一个事实是所有坐标系下的度量张量，全部都可以用直角坐标的基矢量展开成一个单位矩阵的形式。

因此我们再次强调，张量的重要性质就是作为一个整体是一个坐标变换下的不变量，这一句话事实上就足以概括这套体系所定义的张量的本质。

# 第三次作业

## 某题目

$$
\nabla (\tilde A \cdot \tilde B)
=(\nabla \tilde A) \cdot \tilde B
+\tilde A \cdot (\nabla \tilde B)
$$

成立的特例有

1. $\tilde A$ 是标量
2. $\tilde A$ = 0
3. $\nabla \tilde B = 0$
4. $\tilde A = \tilde B = \vec r$
5. $\tilde A = \alpha \tilde g$ ，其中 $\alpha$ 为任意的标量场
6. ...别的情况自己思考

# 11.6笔记

## 极分解定理

任意的二阶可逆张量存在如下唯一分解

$$
\tilde F =
\tilde R \cdot \tilde U
=\tilde V \cdot \tilde R
$$

其中 $\tilde R$ 为正交矩阵，$\tilde U,\tilde V$ 为对称正定张量。

若 $\tilde F$ 为变形梯度张量，这表明所有的变形都可以视作纯变形后旋转或者旋转后纯变形的形式。

## 应变率张量

在 $L$ 观点下（注意以下皆为在$L$观点下的）

$$
\epsilon_{ij} = \epsilon_{ij}(\xi^\alpha,t)
\\
e_{ij}=\frac {\partial \epsilon_{ij}(\xi ^\alpha ,t)}{\partial t}
\\
=\frac 1 2 \nabla_i v_j+\nabla_j v_i
$$

其中 $\vec v$ 为速度矢量，这里利用了$\frac {\partial \vec e_i}{\partial t}=\nabla_i \vec v$，定义张量

$$
\tilde e = e_{ij}\vec e^i \vec e^j
=\frac 1 2(\nabla \vec v+{\nabla \vec v}^T)
$$

为应变率张量。有说明如下：

1. 物理意义上，对角元素反应物质线伸长的随时间的变化率，非对角元素反应物质线夹角随时间的变化率。
2. 且注意到这是一个对称张量，也可以引入主值、主轴、主坐标系的概念。值得说明$\tilde e$和$\tilde \epsilon$二者的主轴没有什么关系。
3. $\tilde e$的第一不变量（迹）即为$\nabla \cdot \vec v$。

下面考虑体积的时间变化率

$$
\frac \partial {\partial t} (dV)
=\frac \partial {\partial t} (d\vec r_1 \cdot (d\vec r_2 \times d\vec r_3))
=\frac \partial {\partial t} \epsilon_{ijk} d\xi^i_1 d\xi^j_2 d\xi^k_3
$$

可以推导得到

$$
\frac {\partial \epsilon_{ijk}} {\partial t}
=\nabla \cdot \vec v \epsilon_{ijk}
$$

故

$$
\frac \partial {\partial t} (dV)
=\nabla \cdot \vec v dV
\frac{\frac \partial {\partial t} (dV)}{dV}
\\
=\nabla \cdot \vec v
$$

再考虑长度的时间变化率可得

$$
\frac {\partial}{\partial t} (d\vec r)
=\frac {\partial}{\partial t}(\vec e_i d\xi^i)
=d\vec r \cdot \nabla \vec v
=(\nabla \vec v)^T\cdot d\vec r
$$

还可得

$$
\frac {\partial \vec e^i}{\partial t}
=-\nabla (v^i)
$$

$$
\frac {\partial |d\vec r|}{\partial t}
=\frac {e_{ij} d\xi^i d\xi^j}{|d\vec r|}
=\frac {d\vec r \cdot \tilde e \cdot d\vec r}{|d\vec r|}
$$

对于面积有（算过了，是对的）

$$
\frac {\partial (d\vec \Sigma)}{\partial t}
=\frac {\partial}{\partial t} (d\vec r_1 \times d\vec r_2)
=\frac {\partial}{\partial t} (\epsilon_{ijk}   d\xi^i_1 d\xi^j_2 \vec e^k)
\\
=(\nabla \cdot \vec v)d\vec \Sigma
-(\nabla \vec v) \cdot d\vec \Sigma
$$

## 局部速度分布

有

$$
\vec v(\xi^i+d\xi^i)
=\vec v(\xi^i)
+\nabla_i \vec v d\xi^i
=\nabla_i v_j \vec e^jd\xi^i
\\
=\vec v(\xi^i)
+\frac 1 2(\nabla_i v_j+\nabla_j v_i) \vec e^j d\xi^i
+\frac 1 2(\nabla_i v_j-\nabla_j v_i) \vec e^j d\xi^i
\\
=\vec v(\xi^i)
+\tilde e \cdot d\vec r
+\frac 1 2(\nabla_i v_j-\nabla_j v_i) \vec e^j d\xi^i
$$

# 11.13笔记

## 协调方程

应变率张量$\tilde e$满足的协调方程据说为

$$
\nabla_i\nabla_j e_{kl}
+\nabla_k\nabla_l e_{ij}
-\nabla_i\nabla_l e_{kj}
-\nabla_k\nabla_j e_{il}
=0
$$

注意到$e_{ij}=\frac {1}{2}(\nabla_i v_j+\nabla_j v_i)$和小变形下$\epsilon_{ij}=\frac {1}{2}(\nabla_i w_j+\nabla_j w_i)$，故在小变形下$\epsilon_{ij}$应当满足同样的协调方程。

---

$\vec w$是两个位形的位移矢量，则$\exist \vec w\Leftrightarrow\mathring K,K$为欧式空间$\Leftrightarrow R^{l\cdot\cdot\cdot}_{\cdot ijk}=0$，其中四阶张量$\tilde R$定义为

$$
\nabla_i\nabla_j a_k
-\nabla_j\nabla_i a_k
=R^{l\cdot\cdot\cdot}_{\cdot ijk} a_l
$$

其中$\vec a$应为任意矢量

---

思考题

欧拉观点下$\epsilon_{ij} = \epsilon_{ij}(x,y,z)$，那么欧拉观点下的物质导数$\frac{d \epsilon_{ij}}{dt}\neq e_{ij}(x,y,z)$，这是因为$\epsilon_{ij}$本身不是坐标变换下的不变量，欧拉坐标下得到的$\epsilon_{ij}$和拉格朗日坐标下得到的$\epsilon_{ij}$根本就不是一个量，从数学上看就是作变量代换的复合映射不能够从一个映射得到另一个映射。故不能直接用欧拉观点下的物质导数定义。

## 连续性方程（质量守恒定律）

拉格朗日观点下，质量守恒定律表述为

$$
\frac{d}{dt}\int_{V_*} \rho(\xi^\alpha,t) dV=0
\\\Rightarrow
\frac{d}{dt}
\iiint_{V_*} \rho \epsilon_{123} 
d\xi^1 d\xi^2 d\xi^3=0
\\\Rightarrow
\iiint_{V_*} \frac{\partial }{\partial t}
(\rho \epsilon_{123}) 
d\xi^1 d\xi^2 d\xi^3=0
\\\Rightarrow
\iiint_{V_*} (\frac{\partial \rho}{\partial t}\epsilon_{123}+\rho \frac{\partial \epsilon_{123}}{\partial t}) 
d\xi^1 d\xi^2 d\xi^3=0
\\\Rightarrow
\int_{V_*} (\frac{\partial \rho}{\partial t}+\rho \nabla \cdot \vec v) dV=0
$$

注意到$V_*$是拉格朗日坐标下的任意空间区域，故上式对任意的$V_*$都成立，故有

$$
\frac{\partial \rho}{\partial t}+\rho \nabla \cdot \vec v=0
$$

可以得到

$$
\nabla \cdot \vec v = \frac{\frac{\partial }{\partial t}(\frac{1}{\rho})}{\frac{1}{\rho}}
$$

对于$\nabla \cdot \vec v=0$则称不可压缩介质。

---

欧拉观点下可以得到的是

$$
\frac {\partial\rho}{\partial t}+\nabla\cdot(\rho \vec v)=0
$$

# 11.27笔记

## 守恒律

守恒凡五律，质动矩能熵。
妙理原相导，殊功各所当。
积分连面体，散度饮篇章。
连介诸事难，皆寻应力方。

## 连续性方程（质量守恒定律）

### 控制体方法

对于物理量$\tilde A$一般的有，

$$
\frac {\partial}{\partial t} \int_V \tilde A dV
=\int_V \tilde B dV
+\int_S \tilde C \cdot \vec n d\Sigma
$$

其中$\tilde B$为体积内部的变化，$\tilde C$为体积边界上的作用，通常包括（质量、动量）流，（压强）作用等。

所谓控制体方法就是将上述这个方程应用到一些特殊的区域内，即选取特殊的（例如维度低的对称性好的）的积分区域后，推导出一些方程。例如准一维管道流动。

## 动量守恒定律

### 质量力和面力

质量力指正比于质量的力，面力指正比于面积的力。质量力是trivial的，但是面力很复杂，因为其大小方向都依赖于面元的方向取向。（静止流体的压强仅仅是退化情况）

### 应力的性质

引入应力作为一种微元面力。面元法向规定为$\vec n$，那么应力$\vec p_n$表示经过面元法向所指的部分对被面元分割的另一部分的作用力。

$$
面力=\int_{\Sigma} \vec p_n d\Sigma
$$

一般的$\vec p_n=p_{nn}\vec n+p_{n\tau}\vec \tau$，即根据切向和法向作分解。

### 应力分解公式与应力张量

假设已知法向矢量$\vec n_{(\alpha)}$，其中$\alpha = 1,2,3$，且对应的面元上的应力为$\vec p_{(\alpha)}$，其中$\alpha = 1,2,3$。容易得到在任意面元$\vec n$上的应力为

$$
\vec p =\sum\limits_\alpha \vec n\cdot \vec n^{(\alpha)} \vec p_{(\alpha)}
$$

其中$\vec n^{(\alpha)}$为$\vec n_{(\alpha)}$的共轭矢量。

于是有应力张量

$$
\tilde P=\sum\limits_\alpha \vec n^{(\alpha)} \vec p_{(\alpha)}
$$

有$\tilde P=\tilde P^T$(在经典情况下动量矩守恒的结论)，而$\tilde P$的分量的物理意义是明显的。即$P^{ij}$表示$\vec e^i$方向面元应力在$\vec e^{j}$上的投影。

于是有微分形式的动量方程

$$
\rho\frac{d\vec v}{dt}=\rho\vec F+\nabla\cdot \tilde P
$$

# 12.4笔记

## 动量矩定理

取物质体$V_*$有

$$
\frac d {dt} \int_{V_*}\vec r \times \vec v \rho dV
=\int_{V_*}\vec r\times \vec F\rho dV
+\int_{\partial V_*}\vec r\times \vec p_nd\sigma
$$

$$
\int_{V_*}\vec r \times \frac {d\vec v} {dt} \rho dV
=\int_{V_*}\vec r\times \vec F\rho dV
+\int_{V_*}\nabla_i (\vec r\times p_{ij} \vec e^j)dV
$$

再结合动量方程就可得到

$$
\int_{V_*}\vec e_i\times\vec e_j p^{ij}=0
$$

即

$$
\vec e_i\times\vec e_j p^{ij}=0
$$

由此可以得到$p^{ij}=p^{ji}$，也就是说，在经典情况下（不考虑电磁场的影响），应力张量是对称的，这是动量矩定理对应力张量提出的约束条件。

## Euler方程

在理想流体的情况下，即认为应力只有沿着法向的分量的情况下，这时$\vec p_n=-p\vec n=\vec n\cdot \tilde P$，那么$\tilde P=-p\tilde g$可以有Euler方程

$$
\rho \frac{d\vec v}{dt}=\rho \vec F-\nabla p
$$

# 12.11笔记

**注：这堂课我旷课了，参考程阳的笔记补上的**

## 粘性流体

### 各向同性线性黏性流体（牛顿流体）

我们将忽略的切向应力重新考虑，那么（粘性）应力张量的分量化作

$$
p^{ij}=-pg^{ij}+\tau^{ij}(e_{\alpha\beta},T,g_{\alpha\beta},\cdots)
$$

同时有约束极限

$$
\lim_{e_{\alpha\beta}\to 0}\tau^{ij}=0
$$

粘性流体的一些约束方程，存在一些实验定律，例如

$$
\tau^{ij}=A^{ij\alpha\beta}e_{\alpha\beta}\quad \text{且} \quad \frac{\partial A^{ij\alpha\beta}}{\partial e_{\alpha\beta}}=0
$$

加上各向同性（旋转变化下分量不变）的性质，可知张量$\bm{A}$原本的 $81$ 个独立分量现在仅有两个。
**注记.** 二阶各项同性张量必然为度量张量的倍数；三阶各项同性张量必然为置换张量的倍数。在此处的结果，我们考虑以下分量

$$
A^{ij\alpha\beta}=\lambda g^{ij}g^{\alpha\beta}+\mu(g^{i\alpha}g^{j\beta}+g^{j\alpha}g^{i\beta})+\nu(g^{i\alpha}g^{j\beta}-g^{j\alpha}g^{i\beta})
$$

由于此处应力张量的对称性，有

$$
\tau^{ij}=\lambda g^{ij}e^{\alpha\cdot}_{\cdot\alpha}+2\mu e^{ij}
$$

定义 $\mu$ 为（剪切）黏度，$\lambda$ 为体积黏度，$\bm{A}$为黏性系数张量。
**我们来讲一点历史** ：第一个研究流体黏性的是牛顿，他将大面积平板放在牛顿流体上，施加水平力产生滑动，则有

$$
\frac{F}{S}=\tau=\mu\frac{\mathrm{d}u(y)}{\mathrm{d}y}
$$

为了纪念此人，故有牛顿流体的概念。
**讨论** ：

* 均匀剪切流动：$u=u(y)$，则有

  $$
  \mu\frac{\mathrm{d}u}{\mathrm{d}y}=\tau^{12}=\tau^{21}
  $$
* 先看平均法向应力：

  $$
  p_{nn\text{ave}}=\lim_{a\to 0}\frac{1}{4\pi a^2}\int_{\text{sphere}}p_{nn}\mathrm{d}\Sigma=\frac{P^{i\cdot}_{\cdot i}}{3}=-p+(\lambda+\frac{2\mu}{3})\nabla\cdot\bm{v}
  $$

  对于不可压缩流体，有

  $$
  p_{nn\text{ave}}=-p
  $$

  所以综上所述，这里这些 $p$ 已经失去了一般意义下压强的含义。注意到应变率张量的第一不变量为

  $$
  e^{i\cdot}_{\cdot i}=\nabla\cdot\bm{v}
  $$
* 我们把牛顿流体的应力分量代入单位体积中内应力的功率：

  $$
  \frac{\delta W_{\Sigma}^{(i)}}{\mathrm{d}V\mathrm{d}t}=-p^{ij}e_{ij}=p\nabla\cdot\bm{v}-\lambda(\nabla\cdot\bm{v})^2-2\mu\bm{e}:\bm{e}=p\nabla\cdot \bm{v}-\Phi
  $$

  定义耗散函数

  $$
  \Phi=\lambda(\nabla\cdot\bm{v})^2+2\mu\bm{e}:\bm{e}
  $$

  但是更本质的写法应当是

  $$
  \Phi=\big(\lambda+\frac{2\mu}{3}\big)(\nabla\cdot\bm{v})^2+2\mu\big(\bm{e}:\bm{e}-\frac{(\nabla\cdot\bm{v})^2}{3}\big)
  $$

  **注意，可以证明上述第二项为非负数，且均匀膨胀时为0；事实上，前面那一项也一般为0（实验的结果，剪切黏度好测，连牛顿都可以测，但是第二黏度测量，有人靠这个吃饭，尤其是气体），定义 $\mu'=\lambda+\frac{2\mu}{3}$为第二黏度，理论上可以说明单原子分子气体的第二黏度为0**
  **在温度极低的情况下，量子效应出现，$\mu=0$，为超流体（例如液氦的一种同位素）**
  那么此时动能定理可以写作

  $$
  \rho\frac{\mathrm{d}}{\mathrm{d}t}\frac{v^2}{2}=\rho\bm{F}\cdot\bm{v}+\nabla_i(p^{ij}v_j)+p\nabla\cdot\bm{v}-\Phi
  $$

  为了方便，我们经常称$p$为热力学压强，那么我们视作这一热力学压强的做功为体积功，那么右侧四项分别为：质量力做功，外面力做功，体积功与耗散（这部分耗散进入了一温度为标志的内能）。
  **问题：有限体积保温杯中有没有装满的水，问是否可能通过摇晃使之沸腾。**
  **注记：两个黏度均为正，联系热力学第二定律可以逐个说明其大于0**

### 粘性流体的Navier-Stokes方程

动量定理有

$$
\rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}=\rho\bm{F}+\nabla\cdot\bm{P}
$$

根据上述说明，由条件：牛顿流体；$\mu,\lambda=\text{const}$，故可以得到动量定理的具体形式

$$
\rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}=\rho\bm{F}-\nabla p+(\lambda+\mu)\nabla\nabla\cdot\bm{v}+\mu\Delta\bm{v}\\\text{or}\Rightarrow\quad\rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}=\rho\bm{F}-\nabla p+(\frac{\mu}{3}+\mu')\nabla\nabla\cdot\bm{v}+\mu\Delta\bm{v}
$$

**注记：如果黏度与温度有关，上述方程会有温度的梯度项，这一点值得注意**
方程的边界条件为牛顿黏附条件（固液同速，实验给出，事实上理论可以推导）

