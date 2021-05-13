# 笔记

## 激活函数
### relu
前向传播 $f(x)=max(0,x)$
反向传播$f'(x)=
\left \{
        \begin{aligned}
            1 \quad x>0 \\
            0 \quad x \le0
        \end{aligned}
\right.
$  
### sigmoid
前向传播$f(x)=\frac 1 {1+ e^x}$
反向传播$f'(x)=\frac 1 {1+ e^x} * (1-\frac 1 {1+ e^x}) = f(x)*(1-f(x))$

