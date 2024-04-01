# 12.18笔记
## 黏性流体和线性弹性体（小变形）
**黏性流体**

**本构关系：**
$$
\begin{align*}
p^{ij}&=-pg^{ij}+A^{ij\alpha\beta}e_{\alpha\beta}\\
&=-pg^{ij}+\lambda g^{ij}\nabla\cdot \vec v+2\mu e^{ij}
\end{align*}
$$
1. $\mu$称为剪切黏度，$\lambda$称为体积黏度。

**平均法向应力：**
$$
\begin{align*}
p_{nnav}=\frac 1 3 p^{i\cdot}_{\cdot i}=-p+(\lambda +\frac 2 3 \mu)\nabla\cdot \vec v
\end{align*}
$$

**N-S方程**

**单位体积内应力功率**
$$
-p^{ij}e_{ij}=p\nabla\cdot \vec v-\phi
$$

---分割线---

**线性弹性体**

**本构关系**
$$
\begin{align*}
p^{ij}&=C^{ij\alpha\beta}\epsilon_{\alpha\beta}\\
&=\lambda g^{ij}\nabla\cdot \vec w+2\mu\epsilon^{ij}
\end{align*}
$$
1. 这里考虑的是小变形的情况即$\epsilon_{\alpha\beta}=\frac{1}{2}(\nabla_\alpha w_\beta+\nabla_\beta w_\alpha)$
2. $C^{ij\alpha\beta}$称为弹性模量张量，也是一个四阶对称各向同性张量。
3. $\lambda,\mu$称为拉梅（法国人喔）系数。

**平均法向应力**
$$
\begin{align*}
    p_{nnav}=(\lambda+\frac 2 3 \mu)\nabla\cdot \vec w=K\theta
\end{align*}
$$
1. 这里的$\mu$是剪切模量，$K=\lambda+\frac 2 3 \mu$是体积模量，$\theta=\nabla \cdot \vec w$是体胀系数。

**拉梅方程**

$$
\rho \frac{d}{dt}\frac{d\vec w}{dt}=\rho\vec F+(\lambda+\mu)\nabla \nabla\cdot \vec w+\mu\nabla^2\vec w
$$

**单位体积内应力功率**
$$
-p^{ij}e_{ij}=-p^{ij}\frac{d\epsilon_{ij}}{dt}
$$
1. 这里利用了近似$e_{ij} dt=d\epsilon_{ij}$？
2. 若定义$X=\frac 1 2 C^{ij\alpha\beta}\epsilon{ij}\epsilon_{\alpha\beta}$，有$\frac{\partial X}{\partial \epsilon_{ij}}=C^{ij\alpha\beta}\epsilon_{\alpha\beta}=p^{ij}$，且$\frac{\partial X(\epsilon_{ij})}{\partial t}=\frac{\partial X}{\partial \epsilon_{ij}}\frac{\partial \epsilon_{ij}}{\partial t}=p^{ij}e_{ij}$,于是我们看出$X$就是弹性势能。

对于单轴拉伸，有$p^{11}=E\epsilon^{11}$和$\epsilon^{22}=\epsilon^{33}=-\nu\epsilon^{11}$，其中$E$为杨氏模量，$\nu$为泊松比。

## 热力学

