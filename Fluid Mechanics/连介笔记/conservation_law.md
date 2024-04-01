## 应变率张量和应变张量物质导数的分量关系
先取静止的Euler坐标系 $(x^1,x^2,x^3)$，在这个坐标系下有速度分量 $v^i(x^1,x^2,x^3,t)$ 与应变张量分量 $\varepsilon_{ij}(x^1,x^2,x^3,t)$，此时应变率张量分量为 

# 质量守恒定律 
## 守恒定律的一般形式
对于一个物理量，我们考虑其本身、产生量和通量，可写作形如
$$
\frac{\partial}{\partial t}\int_V\bm{A}\mathrm{d}V=\int_V\bm{B}\mathrm{d}V+\int_{\Sigma}\bm{n}\cdot\bm{C}\mathrm{d}\Sigma
$$
利用这一个积分形式的方程，可以推导得到微分形式的结果。
## 控制体方法
我们举一个准一维流动的例子来说明控制体方法。取一根很窄的管道，沿$x$方向截面积分布函数为$A(x)$，希望得到基本的控制方程（微分方程）。取$x-x+\Delta x$段为一个控制体$V$，同时假定各个参量在截面上均匀分布（事实上对速度，这个假设很不valid），利用质量守恒定律，有$$\int_V\frac{\partial\rho}{\partial t}\mathrm{d}V=-\int_{\Sigma}\rho v_n\mathrm{d}\Sigma$$利用积分中值定理，可以知道，$\exists x_0\in [x,x+\Delta x],s.t.$ $$\int_V\frac{\partial\rho}{\partial t}\mathrm{d}V=\frac{\partial\rho}{\partial t}(x_0)A(x_0)\Delta x=(\rho vA)(x,t)-(\rho vA)(x+\Delta x,t)$$化简则自然得到所求的微分方程$$A\frac{\partial\rho}{\partial t}+\frac{\partial}{\partial x}(\rho vA)=0$$但是我们同样可以考虑利用一般形式下的质量守恒定律得到结果（思考一下perhaps）$$\frac{\partial\rho}{\partial t}+\nabla\cdot(\bm{v}\rho)=0$$
### 思考(halfdone)
考虑上述问题，同样再考虑另外的问题：假设$A(x,t)$是随时间变化函数，考虑上述推导需要改进的地方
$$\frac{\mathrm{d}}{\mathrm{d}t}\int^{x+\Delta x}_x\rho A\mathrm{d}x=\int^{x+\Delta x}_x\frac{\partial(\rho A)}{\partial t}\mathrm{d}x$$
# 动量守恒定律
## 质量力与面力
引入质量力：单位质量受到的力，记作$\bm{F}$，有特例重力加速度$\bm{g}$。质量力很常见，但是并不是十分关键，其运动的核心驱动力是面力，而不是质量力。
引入面力：给定面积受到的力。面力的大小通常和面法向量方向有关（特例也存在，比如静止流体存在的Pascal定律：各向压强相同）。
引入应力:$\bm{p}_n$，面力是可以通过应力分布计算$$\bm{f}=\int_{\Sigma}\bm{p}_n\mathrm{d}\Sigma$$此处需要对方向进行说明，单位法向量$\bm{n}$指向的方向为1时，$\bm{p}_n$为1侧对2侧（单位法向量的反方向）的面作用力的强度。则上述$\bm{f}$其实是此面受到的力。同时有应力的分解$$\bm{p}_n=p_{nn}\bm{n}+p_{n\tau}\bm{\tau}$$特例：静止流体中，$p_{n\tau}=0$，这是Pascal定律的结果，同时$-p_{nn}$经常被定义为压强。
### 思考 
1.此时应力的大小被定义为压强，但是事实上，在非静止情形时，压强的定义可能会很复杂。
2.负压强（对液体而言，一般定义为）？固体的压强？
## 积分形式的动量守恒定律
利用Newton第二定律，取静止的控制体，动量守恒定律，写作$$\int_V\frac{\partial(\rho\bm{v})}{\partial t}\mathrm{d}V=\int_V\bm{F}\rho\mathrm{d}V+\int_{\Sigma}\bm{p}_n\mathrm{d}\Sigma-\int_{\Sigma}\bm{n}\cdot\bm{v}\bm{v}\rho\mathrm{d}\Sigma$$结合质量守恒定律与高斯定理，我们可以把上述方程写作$$\int_V\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}\rho\mathrm{d}V=\int_V\bm{F}\rho\mathrm{d}V+\int_{\Sigma}\bm{p}_n\mathrm{d}\Sigma$$由于以上守恒定律的存在，对应力的性质事实上有许多限制，导致它需要由一个二阶张量来说明，我们下面会讨论这一点。
## 应力的性质
有以下性质，部分容易证明，均陈述如下：
* $\bm{p}_n=-\bm{p}_{-n}$，在 $\bm{p}_n+\bm{p}_{-n}$连续的情形下成立，特例：激波的情形下，存在突变。
* 应力分解公式：一点的应力状态可以由三个应力表示出来。假设已知的三个不共面单位法向量 $\bm{n}_{(1)},\bm{n}_{(2)},\bm{n}_{(3)}$，以及其上的应力 $\bm{p}_1,\bm{p}_2,\bm{p}_3$，则$\bm{p}_n$可以表示如下（在不失一般性的情况下，可以取一个四面体$MABC$，同时使得$\bm{n}$是面$ABC$的外法向量，另外三个面的内法向量定义作上述的三个法向量，且要求其满足右手系，取控制体$MABC$）$$\int_{MABC}\big(\rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}-\rho\bm{F}\big)\mathrm{d}V=\int_{\partial MABC}\bm{p}_n\mathrm{d}\Sigma$$再次利用积分中值定理得到$$\bm{p}_n=\sum_{\alpha}\frac{S_{\alpha}}{S}\bm{p}_{\alpha}$$利用高斯定理的一个简单结果$\int_{\partial V}\bm{n}\mathrm{d}\Sigma=0$，我们可以得到$$S\bm{n}=\sum_{\alpha}S_{\alpha}\bm{n}_{(\alpha)}$$为了求出这些系数，我们可以取一些逆变的法向量 $\bm{n}^{(\beta)},s.t. \bm{n}^{(\beta)}\cdot\bm{n}_{(\alpha)}=\delta_{\alpha}^{\beta}.$则$$\frac{S_\alpha}{S}=\bm{n}\cdot\bm{n}^{(\alpha)}$$从而我们可以得到$$\bm{p}_n=\bm{n}\cdot\sum_{\alpha}\bm{n}^{(\alpha)}\bm{p}_\alpha=\bm{n}\cdot\bm{P}$$ __思考__：如何求出这些法向量的显式结果。
## 应力张量
上一部分，我们事实上已经陈述了应力张量的存在性，且具有形式$$\bm{p}_n=P^{ij}n_i\bm{e}_j$$其中有 $P^{ij}=\bm{e}^i\cdot\bm{P}\cdot\bm{e}^j$，其实左式已经具有一定的物理意义某方向应力在另一个方向投影的某个倍数。
 __练习__：可以求特定方向，使得法向应力（极大值：特征值最大值）和切向应力（最小值为0，极大值为特征值极差之一半）分别取极值。
 连续介质问题（固体或者流体）的一个核心问题是应力的表示问题。下面简单地说明一下静止流体时小控制体地受力为$$Q=\oint_{\Sigma}\bm{p}_n\mathrm{d}\Sigma=-\oint_{\Sigma}p\bm{n}\mathrm{d}\Sigma=-\int_V\nabla p\mathrm{d}V$$且此时静止流体中的应力张量为$$\bm{P}=-p\bm{g}$$进而得到流体静力学方程$$\rho\bm{F}-\nabla p=0$$若此时质量力为重力场，则会有$$\nabla p = -\rho g\bm{k}$$
## 微分形式的动量方程
根据应力张量，写作$$\rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}=\rho\bm{F}+\nabla\cdot\bm{P}$$
## 动量矩守恒定律
在经典的情况下（不考虑电磁作用等），描述物质体的动量矩守恒定律$$\frac{\mathrm{d}}{\mathrm{d}t}\int_{V^{\ast}}\bm{r}\times\bm{v}\rho\mathrm{d}V=\int_{V^{\ast}}\bm{r}\times\bm{F}\rho\mathrm{d}V+\int_{\Sigma^{\ast}}\bm{r}\times\bm{p}_n\mathrm{d}\Sigma$$利用连续性方程，不难证明，在Euler观点下对物质体积分的物质导数有以下公式$$\frac{\mathrm{d}}{\mathrm{d}t}\int_{V^{\ast}}\bm{A}\rho\mathrm{d}V=\int_{V^{\ast}}\frac{\mathrm{d}\bm{A}}{\mathrm{d}t}\rho\mathrm{d}V$$结合这一公式，我们可以得到$$\int_{V^{\ast}}\rho\bm{r}\times\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}\mathrm{d}V=\int_{V^{\ast}}\bm{r}\times\bm{F}\rho\mathrm{d}V+\int_{V^{\ast}}\nabla_i(\bm{r}\times p^{ij}\bm{e}_j)\mathrm{d}V\\ \Rightarrow\quad \int_{V^{\ast}}\rho\bm{r}\times\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}\mathrm{d}V=\int_{V^{\ast}}\bm{r}\times\bm{F}\rho\mathrm{d}V+\int_{V^{\ast}}(\bm{e}_i\times\bm{e}_jp^{ij}+\bm{r}\times\nabla_ip^{ij}\bm{e}_j)\mathrm{d}V$$代入动量守恒方程，我们可以得到$$\bm{e}_i\times\bm{e}_jp^{ij}=0$$下面我们依靠上式说明经典情况下应力张量的对称性$$\bm{e}_i\times\bm{e}_j(p^{ij}-p^{ji})=\varepsilon_{ijk}(p^{ij}-p^{ji})\bm{e}^k=0$$
## Euler方程（静止流体的直接推广）
将动量守恒定律化简$$\rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}=\rho\bm{F}-\nabla p$$此方程求解事实上比粘性的情况要容易一些，利用Bernoulli积分或者Lagrange积分可以求解上述方程。此方程的数值解法事实上很成熟了。
# 动能定理
直接在动量守恒定律微分方程两侧乘上$\bm{v}$，称$\frac{v^2}{2}$为质量动能，得到$$\frac{1}{2}\rho\frac{\mathrm{d}(\bm{v}\cdot\bm{v})}{\mathrm{d}t}=\rho\bm{F}\cdot\bm{v}+(\nabla\cdot\bm{P})\cdot\bm{v}=\rho\bm{F}\cdot\bm{v}+\nabla_ip^{ij}v_j+=\rho\bm{F}\cdot\bm{v}+\nabla_i(p^{ij}v_j)-p^{ij}\nabla_iv_j$$我们取物质体$V$，然后将上式写作积分形式，有$$\frac{\mathrm{d}}{\mathrm{d}t}\int_V\rho\frac{v^2}{2}\mathrm{d}V=\int_V\rho\bm{F}\cdot\bm{v}\mathrm{d}V+\int_{\Sigma}\bm{p}_n\cdot\bm{v}\mathrm{d}\Sigma-\int_V\bm{P}:\nabla\bm{v}\mathrm{d}V$$关于上式，我们可以逐项理解每一部分的物理意义：物质体动能变化率、质量力功率、外面力的功率与内面力的功率（物质体内的积分，因有此名）。注意到，由于经典情形下的应力张量的对称性，则可以将内面力做功写作$$-\int_V\bm{P}:\bm{e}\mathrm{d}V$$其中$$\bm{e}=\frac{1}{2}(\nabla\bm{v}+(\nabla\bm{v})^T)$$
## 举例
考虑活塞-气缸模型的压缩过程，匀速且缓慢，此时速度线性分布（这不是假设，是已有假设的推论，因为它需要满足Euler方程，**试证明之**）则我们考虑上述积分在此情形下的结果。外面力功率为$$\frac{\delta W_{\Sigma}^{(e)}}{\mathrm{d}t}=pAv_0$$内面力功率为$$\frac{\delta W_{\Sigma}^{(i)}}{\mathrm{d}t}=-pAv_0$$
**试着在匀加速过程中讨论上述问题。** “不慢”的时候，需要考虑声速，即考虑空气中的振动。
# 流体模型
## 理想流体(作用力仅只有法向作用力，这当然与实际差异很大)
本部分在**Euler方程**部分中已经讨论过一部分了。我们先给出已有的不封闭的方程（一般情形下需要加入热力学方程，才能封闭该方程组）$$\rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}=\rho\bm{F}-\nabla p\\ \frac{\mathrm{d}\rho}{\mathrm{d}t}+\rho\nabla\cdot\bm{v}=0$$
## 粘性流体
### 各向同性线性黏性流体（牛顿流体）
我们将忽略的切向应力重新考虑，那么（粘性）应力张量的分量化作$$p^{ij}=-pg^{ij}+\tau^{ij}(e_{\alpha\beta},T,g_{\alpha\beta},\cdots)$$同时有约束极限$$\lim_{e_{\alpha\beta}\to 0}\tau^{ij}=0$$粘性流体的一些约束方程，存在一些实验定律，例如$$\tau^{ij}=A^{ij\alpha\beta}e_{\alpha\beta}\quad \text{且} \quad \frac{\partial A^{ij\alpha\beta}}{\partial e_{\alpha\beta}}=0$$加上各向同性（旋转变化下分量不变）的性质，可知张量$\bm{A}$原本的 $81$ 个独立分量现在仅有两个。
**注记.** 二阶各项同性张量必然为度量张量的倍数；三阶各项同性张量必然为置换张量的倍数。在此处的结果，我们考虑以下分量$$A^{ij\alpha\beta}=\lambda g^{ij}g^{\alpha\beta}+\mu(g^{i\alpha}g^{j\beta}+g^{j\alpha}g^{i\beta})+\nu(g^{i\alpha}g^{j\beta}-g^{j\alpha}g^{i\beta})$$由于此处应力张量的对称性，有$$\tau^{ij}=\lambda g^{ij}e^{\alpha\cdot}_{\cdot\alpha}+2\mu e^{ij}$$定义 $\mu$ 为（剪切）黏度，$\lambda$ 为体积黏度，$\bm{A}$为黏性系数张量。
**我们来讲一点历史** ：第一个研究流体黏性的是牛顿，他将大面积平板放在牛顿流体上，施加水平力产生滑动，则有$$\frac{F}{S}=\tau=\mu\frac{\mathrm{d}u(y)}{\mathrm{d}y}$$为了纪念此人，故有牛顿流体的概念。
**讨论** ：
* 均匀剪切流动：$u=u(y)$，则有$$\mu\frac{\mathrm{d}u}{\mathrm{d}y}=\tau^{12}=\tau^{21}$$
* 先看平均法向应力： $$p_{nn\text{ave}}=\lim_{a\to 0}\frac{1}{4\pi a^2}\int_{\text{sphere}}p_{nn}\mathrm{d}\Sigma=\frac{P^{i\cdot}_{\cdot i}}{3}=-p+(\lambda+\frac{2\mu}{3})\nabla\cdot\bm{v}$$对于不可压缩流体，有$$p_{nn\text{ave}}=-p$$
所以综上所述，这里这些 $p$ 已经失去了一般意义下压强的含义。注意到应变率张量的第一不变量为$$e^{i\cdot}_{\cdot i}=\nabla\cdot\bm{v}$$
* 我们把牛顿流体的应力分量代入单位体积中内应力的功率：$$\frac{\delta W_{\Sigma}^{(i)}}{\mathrm{d}V\mathrm{d}t}=-p^{ij}e_{ij}=p\nabla\cdot\bm{v}-\lambda(\nabla\cdot\bm{v})^2-2\mu\bm{e}:\bm{e}=p\nabla\cdot \bm{v}-\Phi$$定义耗散函数$$\Phi=\lambda(\nabla\cdot\bm{v})^2+2\mu\bm{e}:\bm{e}$$但是更本质的写法应当是$$\Phi=\big(\lambda+\frac{2\mu}{3}\big)(\nabla\cdot\bm{v})^2+2\mu\big(\bm{e}:\bm{e}-\frac{(\nabla\cdot\bm{v})^2}{3}\big)$$**注意，可以证明上述第二项为非负数，且均匀膨胀时为0；事实上，前面那一项也一般为0（实验的结果，剪切黏度好测，连牛顿都可以测，但是第二黏度测量，有人靠这个吃饭，尤其是气体），定义 $\mu'=\lambda+\frac{2\mu}{3}$为第二黏度，理论上可以说明单原子分子气体的第二黏度为0** 
**在温度极低的情况下，量子效应出现，$\mu=0$，为超流体（例如液氦的一种同位素）**
那么此时动能定理可以写作$$\rho\frac{\mathrm{d}}{\mathrm{d}t}\frac{v^2}{2}=\rho\bm{F}\cdot\bm{v}+\nabla_i(p^{ij}v_j)+p\nabla\cdot\bm{v}-\Phi$$为了方便，我们经常称$p$为热力学压强，那么我们视作这一热力学压强的做功为体积功，那么右侧四项分别为：质量力做功，外面力做功，体积功与耗散（这部分耗散进入了一温度为标志的内能）。
**问题：有限体积保温杯中有没有装满的水，问是否可能通过摇晃使之沸腾。**
**注记：两个黏度均为正，联系热力学第二定律可以逐个说明其大于0**
### 粘性流体的Navier-Stokes方程
动量定理有$$\rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}=\rho\bm{F}+\nabla\cdot\bm{P}$$根据上述说明，由条件：牛顿流体；$\mu,\lambda=\text{const}$，故可以得到动量定理的具体形式
$$\rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}=\rho\bm{F}-\nabla p+(\lambda+\mu)\nabla\nabla\cdot\bm{v}+\mu\Delta\bm{v}\\\text{or}\Rightarrow\quad\rho\frac{\mathrm{d}\bm{v}}{\mathrm{d}t}=\rho\bm{F}-\nabla p+(\frac{\mu}{3}+\mu')\nabla\nabla\cdot\bm{v}+\mu\Delta\bm{v}$$
 **注记：如果黏度与温度有关，上述方程会有温度的梯度项，这一点值得注意**
方程的边界条件为牛顿黏附条件（固液同速，实验给出，事实上理论可以推导）
# 固体模型
## 线性弹性体的本构方程
# 热力学模型
## 热力学第二定律
