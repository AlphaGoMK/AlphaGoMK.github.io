# Object Detection

## AutoAssign: Differentiable Label Assignment for Dense Object Detection

> Label-assignment, 选择正/负样本点(anchor-free)需要动态确定，且正负样本点损失加权。权重：cate+物体计算
>
> center prior: 训练时固定先验
>
> (instance) confidence map: 根据数据动态调整

之前空间上采样正负样本的方法为 1. IoU 2. 物体框中心固定区域都为正样本。但存在物体框中部分区域没有物体，且固定区域无法优化 *obstacle caused by feature shifting when backgrounds are sampled as positives may decrease the performance.*

提出根据category&instance信息动态产生pos/neg weight map [**differentiable/data-driven**]

![image-20200715142809070](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200715142809070.png)

![image-20200715142902639](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200715142902639.png)

👆更加dynamic

![image-20200715143039412](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200715143039412.png)

通过Center Weighting + Confidence Weighting得到Weight Map(![latex_equ](https://latex.codecogs.com/svg.latex?w^+,w^-))

#### Center Weighting

学习一个category-wise分布，即带参数的Gaussian-shape weighting function，用在FPN的每一层。

![latex_equ](https://latex.codecogs.com/svg.latex?G%28\vec{d}%20\mid%20\vec{\mu},%20\vec{\sigma}%29=e^{\frac{-%28\vec{d}-\vec{\mu}%29^{2}}{2%20\vec{\sigma}^{2}}})，其中![latex_equ](https://latex.codecogs.com/svg.latex?\vec{d})为某个位置的xy偏移量，一种类别有一组![latex_equ](https://latex.codecogs.com/svg.latex?\vec{\mu}\;\vec{\sigma})

center prior可以<u>增加类别的先验分布信息</u>，防止网络冷启动后出现过拟合（不断优化第一次w高的点）

由于conf weighting作用的是<u>每个GT-Box中所有点</u>，所以应用center prior时已知类别

#### Confidence Weighting

![截屏2020-07-15 下午3.02.53](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/%E6%88%AA%E5%B1%8F2020-07-15%20%E4%B8%8B%E5%8D%883.02.53-4796696.png)

增加Implicit-Objectness分支，抑制false-pos（框内部分点不在物体上）

训练时分类分支一起训练 ![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{P}_{i}%28\operatorname{cls}\mid\theta%29=\mathcal{P}_{i}%28\operatorname{cls}\mid%20obj,%20\theta%29%20\mathcal{P}_{i}%28\text%20{obj}%20\mid%20\theta%29)

认为之前固定匹配模型的分类分支为<u>已知点是正样本</u>![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{P}_i%28obj\mid\theta%29=1)，预测点为某个类别的概率，即分类分支的结果为是![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{P}_i%28cls|obj,\theta%29)条件概率。![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{P}_i%28cls\mid\theta%29=\mathcal{P}_i%28cls\mid%20obj,\theta%29), ![latex_equ](https://latex.codecogs.com/svg.latex?\theta)是参数

现在改为Impl-Obj分支预测![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{P}_i%28obj\mid\theta%29)，再和分类分支![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{P}_{i}%28cls\mid%20obj,%20\theta%29)一起得到![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{P}_i%28cls\mid\theta%29)

且增加localization conf作为计算样本权重的依据，通过指数函数将定位准确率转为likelihood![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{P}_i%28\theta%29=\mathcal{P}_i%28cls\mid\theta%29\mathcal{e}^{-\lambda\mathcal{L}_i^{loc}%28\theta%29})，在计算loss体现。类似*Learning From Noisy Anchors for One-Stage Object Detection* 综合考虑分类和定位的性能

![latex_equ](https://latex.codecogs.com/svg.latex?\begin{aligned}%20\mathcal{L}_{i}%28\theta%29%20&=\mathcal{L}_{i}^{c%20l%20s}%28\theta%29+\lambda%20\mathcal{L}_{i}^{l%20o%20c}%28\theta%29%20\\%20&=-\log%20\left%28\mathcal{P}_{i}%28c%20l%20s%20\mid%20\theta%29\right%29+\lambda%20\mathcal{L}_{i}^{l%20o%20c}%28\theta%29%20\\%20&=-\log%20\left%28\mathcal{P}_{i}%28c%20l%20s%20\mid%20\theta%29%20e^{-\lambda%20\mathcal{L}_{i}^{l%20o%20c}%28\theta%29}\right%29%20\\%20&=-\log%20\left%28\mathcal{P}_{i}%28c%20l%20s%20\mid%20\theta%29%20\mathcal{P}_{i}%28loc%20\mid%20\theta%29\right%29%20\\%20&=-\log%20\left%28\mathcal{P}_{i}%28\theta%29\right%29%20\end{aligned})

joint conf representation ![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{P}_i%28\theta%29)经过指数计算，增强![latex_equ](https://latex.codecogs.com/svg.latex?C\left%28\mathcal{P}_{i}\right%29=e^{\frac{\mathcal{P}_{i}%28\theta%29}{\tau}}) （只会有一小部分落在物体上，得分高）

#### Weight Map

对一个**GT框内的点**计算权重，focus on proper loc inside bbox

融合，求正样本和负样本权重

![latex_equ](https://latex.codecogs.com/svg.latex?w_{i}^{+}=\frac{C\left%28\mathcal{P}_{i}\right%29%20G\left%28\vec{d}_{i}\right%29}{\sum_{j%20\in%20S_{n}}%20C\left%28\mathcal{P}_{i}\right%29%20G\left%28\vec{d}_{i}\right%29})

![latex_equ](https://latex.codecogs.com/svg.latex?w_{i}^{-}=1-f\left%28\frac{1}{1-\mathrm{i}%20\mathrm{ou}_{i}}\right%29)，f normalize to [0,1]，<u>分数为sharpen权重分布</u>

Loss函数![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{L}%28\theta%29=-\sum_{n=1}^{N}%20\log%20\left%28\sum_{i%20\in%20S_{n}}%20w_{i}^{+}%20\mathcal{P}_{i}^{+}\right%29-\sum_{j%20\in%20S}%20\log%20\left%28w_{j}^{-}%20\mathcal{P}_{j}^{-}\right%29)

👆学习策略为一个box的训练，对一个box构建![latex_equ](https://latex.codecogs.com/svg.latex?w^+/w^-)训练

不同尺度热力图可视化👇

![image-20200715150853109](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200715150853109.png)

👇ImplicitObjectness抑制噪声效果显著

![image-20200807112856084](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200807112856084.png)

基于FCOS，提升2-3点，mAP=52.1%

---

## Towards Accurate One-Stage Object Detection with AP-Loss

> Label-assignment

正负样本不均衡问题 *It is observed that the classification metric could be very high for a trivial solution which predicts negative label for almost all candidate boxes, while the detection performance is poor*

<img src="Figures/image-20200716164022633.png" alt="image-20200716164022633" style="zoom:50%;" />

负样本多，正样本分类结果对loss贡献少，对检测acc影响小

提出直接对AP(average precision)优化：AP Loss

<u>把分类分支问题看为一个rank问题，计算AP Loss，采用error-driven方式优化</u>

#### Ranking Task & AP Loss

因为计算AP过程实际是对预测结果排序，再计算，所以把分类问题看作一个ranking问题：对于每一类，所有的正样本点排在负样本点之前（更高分）

AP-Loss的primary term为：![latex_equ](https://latex.codecogs.com/svg.latex?%283%29\;\;L_{i%20j}%28\boldsymbol{x}%29=\frac{H\left%28x_{i%20j}\right%29}{1+\sum_{k%20\in%20\mathcal{P}%20\cup%20\mathcal{N},%20k%20\neq%20i}%20H\left%28x_{i%20k}\right%29}=L_{i%20j}) . 其中![latex_equ](https://latex.codecogs.com/svg.latex?x_{ij})为两点的排序差距：![latex_equ](https://latex.codecogs.com/svg.latex?\forall%20i,%20j,%20\quad%20x_{i%20j}=-\left%28s_{i}-s_{j}\right%29)将分类预测的分数转化为排序x，作为损失函数的输入，H为Heaviside step function阶跃函数：![latex_equ](https://latex.codecogs.com/svg.latex?H%28x%29=\left\{\begin{array}{ll}0%20&%20x<0%20\\%201%20&%20x%20\geq%200\end{array}\right.)

即只有j排在i之前时，才对![latex_equ](https://latex.codecogs.com/svg.latex?L_{ij})有贡献。GT定义为![latex_equ](https://latex.codecogs.com/svg.latex?\forall%20i,%20j,%20\quad%20y_{i%20j}=\mathbf{1}_{t_{i}=1,%20t_{j}=0})

AP-Loss为：

![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{L}_{A%20P}=1-\mathrm{AP}=1-\frac{1}{|\mathcal{P}|}%20\sum_{i%20\in%20\mathcal{P}}%20\frac{\operatorname{rank}^{+}%28i%29}{\operatorname{rank}%28i%29}) 
![latex_equ](https://latex.codecogs.com/svg.latex?=1-\frac{1}{|\mathcal{P}|}%20\sum_{i%20\in%20\mathcal{P}}%20\frac{1+\sum_{j%20\in%20\mathcal{P},%20j%20\neq%20i}%20H\left%28x_{i%20j}\right%29}{1+\sum_{j%20\in%20\mathcal{P},%20j%20\neq%20i}%20H\left%28x_{i%20j}\right%29+\sum_{j%20\in%20\mathcal{N}}%20H\left%28x_{i%20j}\right%29}=\frac{1}{\mathcal{P}}\left%28\mathcal{P}-\sum_{i%20\in%20\mathcal{P}}%20\frac{1+\sum_{j%20\in%20\mathcal{P},%20j%20\neq%20i}%20H\left%28x_{i%20j}\right%29}{1+\sum_{j%20\in%20\mathcal{P},%20j%20\neq%20i}%20H\left%28x_{i%20j}\right%29+\sum_{j%20\in%20\mathcal{N}}%20H\left%28x_{i%20j}\right%29}\right%29=\frac{1}{\mathcal{P}}\sum_{i\in\mathcal{P}}\frac{\sum_{j\in\mathcal{N}}H%28x_{ij}%29}{1+\sum_{j\in\mathcal{P},j\neq%20i}H%28x_{ij}%29+\sum_{j\in\mathcal{N}}H%28x_{ij}%29})

![latex_equ](https://latex.codecogs.com/svg.latex?=\frac{1}{|\mathcal{P}|}%20\sum_{i%20\in%20\mathcal{P}}%20\sum_{j%20\in%20\mathcal{N}}%20L_{i%20j}=\frac{1}{|\mathcal{P}|}%20\sum_{i,%20j}%20L_{i%20j}%20\cdot%20y_{i%20j}=\frac{1}{|\mathcal{P}|}\langle\boldsymbol{L}%28\boldsymbol{x}%29,%20\boldsymbol{y}\rangle) （由于只有i j分属正负时![latex_equ](https://latex.codecogs.com/svg.latex?L_{ij})非0，同![latex_equ](https://latex.codecogs.com/svg.latex?y)的定义）

即为计算任意一个正样本P和一个负样本N的primary term。正样本之间没有，正负样本之间有损失

![latex_equ](https://latex.codecogs.com/svg.latex?\min%20_{\boldsymbol{\theta}}%20\mathcal{L}_{A%20P}%28\boldsymbol{\theta}%29=1-\operatorname{AP}%28\boldsymbol{\theta}%29=\frac{1}{|\mathcal{P}|}\langle\boldsymbol{L}%28\boldsymbol{x}%28\boldsymbol{\theta}%29%29,%20\boldsymbol{y}\rangle)

#### Error-driven Update

*the update is directly derived from the difference between desired output and current output*：![latex_equ](https://latex.codecogs.com/svg.latex?\Delta%20x_{i%20j}=L^*_{i,j}-L_{i%20j})

最小化AP-Loss到0，即最佳优化方向![latex_equ](https://latex.codecogs.com/svg.latex?\Delta%20x_{i%20j}=-L_{i%20j}%20\cdot%20y_{i%20j})，y控制是否对于loss有贡献

优化时，衡量改变参数![latex_equ](https://latex.codecogs.com/svg.latex?\theta)后x的变化与最佳优化方向之间的相似度，以及L2正则

![latex_equ](https://latex.codecogs.com/svg.latex?\arg%20\min%20_{\Delta%20\boldsymbol{\theta}}\left\{-\left\langle\Delta%20\boldsymbol{x},%20\boldsymbol{x}\left%28\boldsymbol{\theta}^{%28n%29}+\Delta%20\boldsymbol{\theta}\right%29-\boldsymbol{x}\left%28\boldsymbol{\theta}^{%28n%29}\right%29\right\rangle+\lambda\|\Delta%20\boldsymbol{\theta}\|_{2}^{2}\right\})，一阶泰勒展开算参数的优化方向

由于参数变化![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{\theta}^{%28n+1%29}-\mathcal{\theta}^{%28n%29})在BP过程中不变，所以最佳优化方向![latex_equ](https://latex.codecogs.com/svg.latex?\Delta%20x_{ij})即为x的梯度，链式求s的梯度

![latex_equ](https://latex.codecogs.com/svg.latex?%2813%29\;\;\begin{aligned}%20g_{i}&=-\sum_{j,%20k}%20\Delta%20x_{j%20k}%20\cdot%20\frac{\partial%20x_{j%20k}}{\partial%20s_{i}}=\sum_{j}%20\Delta%20x_{i%20j}-\sum_{j}%20\Delta%20x_{j%20i}%20\\%20&=\sum_{j}%20L_{j%20i}%20\cdot%20y_{j%20i}-\sum_{j}%20L_{i%20j}%20\cdot%20y_{i%20j}%20\end{aligned})

<img src="Figures/image-20200716173631622.png" alt="image-20200716173631622" style="zoom:50%;" />

性能基于RetinaNet提升3个点

---

## DR Loss: Improving Object Detection by Distributional Ranking

> 基于AP-Loss，采用类似SVM方式对正负样本排序（margin）
>
> 增加对正负样本分布的改变 reweight

<img src="Figures/image-20200716173939029.png" alt="image-20200716173939029" style="zoom:50%;" />

## Scale-Equalizing Pyramid Convolution for Object Detection (SEPC)

>采用3D卷积进行多尺度特征融合，不同层融合权重不同

特征金字塔不同层之间存在semantic gap，之前采用feature fusion解决但没有提取intrinsic property

**Pyramid Conv** `PConv`: **在尺度维上 (不同层特征图)** 做卷积（like spatial conv），3D conv

选择三层，底层采用stride卷积，中间普通卷积，上层bilinear upsampling。控制不同层的不同stride来获得相同大小的输出，并相加。时间成本为1.5倍

![image-20200704214032431](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200704214032431.png)

BN修改为整个特征金字塔层共享统计量，效果类似sync_bn，同步不同位置上的统计量，统计范围更广

BN = standardization(![latex_equ](https://latex.codecogs.com/svg.latex?\mu),![latex_equ](https://latex.codecogs.com/svg.latex?\sigma)) + scale_shift(![latex_equ](https://latex.codecogs.com/svg.latex?\gamma),![latex_equ](https://latex.codecogs.com/svg.latex?\beta))，统计卷积的bn为每张特征图，即在channel维的统计（channel个统计量），**统计不同样本在同一个channel上所有wh点的**batch statistics ![latex_equ](https://latex.codecogs.com/svg.latex?\mu)和![latex_equ](https://latex.codecogs.com/svg.latex?\sigma)（reduce at batch dimension）。每张特征图一对parameters ![latex_equ](https://latex.codecogs.com/svg.latex?\gamma)和![latex_equ](https://latex.codecogs.com/svg.latex?\beta) （https://zhuanlan.zhihu.com/p/43200897）

可以证明PConv能够提取Gaussian Pyramid(不同高斯模糊核对图片处理) 尺度不变的特征，即如果原图物体尺度变化，则可通过对PConv提取的特征shift得到改变后物体的特征 *<u>论文中有证明</u>*

**Scale-Equalizing Pyramid Conv** `SEPC`: PConv的升级版，通过deformable conv实现对不同尺度特征的对齐。

```python
def pconv_module_forward(x, conv2D_list):
	# x: input feature list [p3,p4,p5,p6,p7] # conv2D_list: conv2D module list,
	# [nn.Conv2D(stride=2),nn.Conv2D(),nn.Conv2D()]
	out_x = []
	for level in range(len(x)):
		tmp = conv2D_list[1](x[level])
		if level > 0:
			tmp += conv2D_list[0](x[level-1])
		if level < len(x) - 1:
			tmp += Upsample(conv2D_list[2](x[level+1])
		out_x.append(tmp) 
	return out_x
```

最底层普通卷积kernel size不变，其他层使用deformable conv

![image-20200704213956083](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200704213956083.png)

因为feature pyramid中由于backbone非线性操作，相对于gaussian pyramid特征不对齐，所以只通过dilation提取特征不全面，改用d-conv

*SEPC is an improved version of pconv, to relax the discrepancy of feature pyramid from a Gaussian pyramid by aligning the feature map of higher layers with the lowest layer*

底层特征采用普通卷积，卷积结果作为权重向上层的d-conv共享（卷积结果作为d-conv的一个输入求offset，类似CentripetalNet中）

SEPC模块用于取代RetinaNet的detection head的卷积模块

本质为一种pix-wise or fine-grained 的feature fusion，多尺度特征图通过卷积操作进行特征融合

![image-20200704214335943](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200704214335943.png)

性能提升明显

![image-20200704214411536](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200704214411536.png)

---

## PSConv: Squeezing Feature Pyramid into One Compact Poly-Scale Convolutional Layer

> 提取多尺度特征

一个卷积核不同通道不同dilation rate（不同颜色表示不同dilate）

![image-20200717174956433](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200717174956433.png)

![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{H}_{c,%20x,%20y}=\sum_{k=1}^{C_{i%20n}}%20\sum_{i=-\frac{K-1}{2}}^{\frac{K-1}{2}}%20\sum_{j=-\frac{K-1}{2}}^{\frac{K-1}{2}}%20\mathcal{G}_{c,%20k,%20i,%20j}%20\mathcal{F}_{k,%20x+i%20D_{%28c,%20k%29},%20y+j%20D_{%28c,%20k%29}})

扩张率在in通道上循环交替排布（👆T=4循环排列），在out通道维度上也循环排列

![image-20200717175208977](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200717175208977.png)

Fast RCNN替换backbone中所有卷积，FPN neck不变，提升2点左右

## Rethinking Classification and Localization for Object Detection (Double-Head RCNN)

> 检测头进行双路分支改进

二阶段检测器中，对proposal处理的传统检测头采用conv层，或者fast rcnn使用fc层处理

![image-20200706183720979](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200706183720979.png)

fc层作为head，由于全连接操作不同位置权重不共享，对于位置更加敏感（小位置变化大的输出结果变化），适合分类

conv层作为head，由于卷积操作不同位置权重共享，适合回归整个物体（coco2018结果表明可以使用conv层作为bbox的预测头）

![image-20200706185341027](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200706185341027.png)

分析conv作为head的分类回归结果👇

![image-20200706184401851](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200706184401851.png)

fc作为head的分类回归结果👆

1. 分类：fc作为head时，其高iou部分分类结果明显比conv作为head好。计算Pearson correlation coefficient👇fc-head的score-iou关系大于conv-head的score-iou关系

![image-20200706184732698](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200706184732698.png)

2. 定位：conv-head略好于fc-head

可视化关联关系：conv-head采用得到的特征图cos距离，7x7grid，每个点产生7x7大小的和其他任意点的关联矩阵；fc-head对fc层权重变换输出7x7特征图

![image-20200706185049024](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200706185049024.png)

conv-head计算的点和周围关联关系大

#### Double head RCNN

两个分支，conv-head用于回归，fc-head用于分类

![image-20200706185402627](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200706185402627.png)

改进：extend，训练时利用conv-head的分类损失和fc-head的回归损失进行监督(unfocused task supervision)

![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{L}^{f%20c}=\lambda^{f%20c}%20L_{c%20l%20s}^{f%20c}+\left%281-\lambda^{f%20c}\right%29%20L_{r%20e%20g}^{f%20c}), ![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{L}^{c%20o%20n%20v}=\left%281-\lambda^{c%20o%20n%20v}\right%29%20L_{c%20l%20s}^{c%20o%20n%20v}+\lambda^{c%20o%20n%20v}%20L_{r%20e%20g}^{c%20o%20n%20v})

同时对分类分数进行二分支融合(classifiers are complimentary)

![image-20200706185811416](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200706185811416.png)

FPN+2.5~3.8 val AP

---

## D2Det: Towards High Quality Object Detection and Instance Segmentation

> 一阶段回归分支操作移植到二阶段
>
> deformable pooling + weighted pooling

![image-20200706234538094](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200706234538094.png)

#### Dense local regression

把RoI看作类似一阶段中的特征图。之前二阶段检测器对于一个RoI只预测一个框 (FC+预测)，改成RoI中每个点都预测一个框(dense prediction)。预测中间点到box上下左右的距离 (FCOS)

增加binary overlap prediction (![latex_equ](https://latex.codecogs.com/svg.latex?\hat{m}))，对RoI和GT重叠部分为1，只有为1的点的预测结果有效。

<u>类似</u>一阶段中每点预测bbox，预测IoU/objectness

#### Discriminative RoI pooling

<u>deformable pooling + weighted pooling</u>

pooling的weighted权重通过输入特征操作得到（类似self-attention）👇W为计算的权重

![latex_equ](https://latex.codecogs.com/svg.latex?\tilde{F}=W%28F%29%20\odot%20F)

![image-20200707000247234](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200707000247234.png)性能提升较明显: multi-scale 50.1

---

## NMS by Representative Region: Towards Crowded Pedestrian Detection by Proposal Pairing (PBM)

>基于bi-box regre. 根据vis框进行NMS

行人检测中重叠，类内重叠Replusion Loss和AggLoss惩罚两个人中间的box，但是box的重叠仍然会导致在NMS过程中被误删

Adaptive NMS中GT-density和Pred-density的inconsistency

![image-20200707105643820](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200707105643820.png)

GT框不密集，但Pred-box是密集的。Adaptive NMS预测GT的密集程度，而NMS和Pred的密集程度有关。👆预测绿框不密集，所以红框不会被保留

#### NMS by representative region

通过计算visible部分的IoU而不是full部分的IoU进行NMS

重叠不同物体的检测框full-IoU大，vis-IoU小。重叠同一物体full-IoU和vis-IoU都大 ![latex_equ](https://latex.codecogs.com/svg.latex?\to) vis-IoU判别

![image-20200707132234249](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200707132234249.png)

#### Paired BBox Faster RCNN

Paired RPN + Paired Proposal Feature Extractor +  Pair RCNN

##### Paired RPN

GT标注为pair Q=(F,V)，全身框和可见框

**从一个anchor**回归出full和visible的proposal（为了inherent correspondence）

anchor和一对GT标注的匹配标准：

![latex_equ](https://latex.codecogs.com/svg.latex?\operatorname{IoU}%28\mathrm{A},%20\mathrm{F}%29%20\geq%20\alpha_{1}) and ![latex_equ](https://latex.codecogs.com/svg.latex?\operatorname{IoF}%28\mathrm{A},%20\mathrm{V}%29%20\geq%20\beta_{1})

![latex_equ](https://latex.codecogs.com/svg.latex?\operatorname{IoU}%28\mathrm{A},%20\mathrm{F}%29=\frac{\operatorname{Area}%28\mathrm{A}%20\cap%20\mathrm{F}%29}{\operatorname{Area}%28\mathrm{A}%20\cup%20\mathrm{F}%29},\;\;\operatorname{IoF}%28\mathrm{A},%20\mathrm{V}%29=\frac{\operatorname{Area}%28\mathrm{A}%20\cap%20\mathrm{V}%29}{\operatorname{Area}%28\mathrm{V}%29})

full和vis的回归训练策略同bi-box regression

##### Paired Proposal Feature Extractor

融合full/vis两个proposal的特征

![image-20200707133322847](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200707133322847.png)

<font color="#00dd00">绿色分支</font>+<font color="dd0000">红色分支</font>：简单的concat两个proposal的特征向量

<font color="00dd00">绿色分支</font>+黑色分支：visible region attention/mask，构建visible部分的0/1 mask，与full特征![latex_equ](https://latex.codecogs.com/svg.latex?F_f)点乘得![latex_equ](https://latex.codecogs.com/svg.latex?F_m)，再concat ![latex_equ](https://latex.codecogs.com/svg.latex?F_m)和![latex_equ](https://latex.codecogs.com/svg.latex?F_v)

##### Paired RCNN

融合特征和proposal作为输入，两个分支分别预测full和vis的检测框

proposal和一对GT的匹配标准：

![latex_equ](https://latex.codecogs.com/svg.latex?\operatorname{IoU}\left%28\mathrm{P}_{\mathrm{f}},%20\mathrm{F}\right%29%20\geq%20\alpha_{2}) and ![latex_equ](https://latex.codecogs.com/svg.latex?\operatorname{IoU}\left%28\mathrm{P}_{\mathrm{v}},%20\mathrm{V}\right%29%20\geq%20\beta_{2})

性能提升明显

![image-20200707134837180](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200707134837180.png)

---

## Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection

>NAS整体搜索，基于FBNet

![image-20200708170606633](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200708170606633.png)

二阶段检测器：backbone + neck(feature pyramid & fusing) + RPN(fixed) + Head

NATS和DetNAS搜索backbone，NAS-FPN搜FPN，Auto-FPN搜fusing和Head

单独搜索效果较差，采用整体每个部件(backbone: ![latex_equ](https://latex.codecogs.com/svg.latex?\alpha), neck: ![latex_equ](https://latex.codecogs.com/svg.latex?\beta), head: ![latex_equ](https://latex.codecogs.com/svg.latex?\gamma))一起搜索，**end-to-end search**

![latex_equ](https://latex.codecogs.com/svg.latex?\alpha^{*},%20\beta^{*},%20\gamma^{*}=\underset{\alpha,%20\beta,%20\gamma}{\arg%20\min%20}%20\mathcal{L}_{v%20a%20l}^{d%20e%20t}\left%28\alpha,%20\beta,%20\gamma,%20w^{*}%28\alpha,%20\beta,%20\gamma%29\right%29=\underset{\alpha,%20\beta,%20\gamma}{\arg%20\min%20}%20\mathcal{L}_{v%20a%20l}^{d%20e%20t}\left%28\alpha,%20\beta,%20\gamma,%20\underset{w}{\arg%20\min%20}%20\mathcal{L}_{t%20r%20a%20i%20n}^{d%20e%20t}%28\alpha,%20\beta,%20\gamma,%20w%29\right%29)

每个component采用逐层搜索，构建每个操作用在每一层的得分矩阵![latex_equ](https://latex.codecogs.com/svg.latex?\alpha)，选择概率

对一个操作用在不同层的得分进行L2**正则化(column-space regularization)**，减少得分的数值大小：

![latex_equ](https://latex.codecogs.com/svg.latex?\min%20_{\alpha}%20f%28\alpha%29+\mu%20\min%20_{i}%28\sqrt{\sum_{l=1}^{L}%20\alpha_{l,%20i}^{2}}%29)

选择![latex_equ](https://latex.codecogs.com/svg.latex?\alpha_{l,i})最大max的一个操作，但为了可微采用softmax来选择(continuous relaxation)

搜索backbone时搜索空间的节点计算(differentiable)：![latex_equ](https://latex.codecogs.com/svg.latex?x_{l}=\sum_{o%20\in%20\mathcal{O}_{b}}%20\frac{\exp%20\left%28\alpha_{l}^{o}\right%29}{\sum_{o^{\prime}%20\in%20\mathcal{O}_{b}}%20\exp%20\left%28\alpha_{l}^{o^{\prime}}\right%29}%20o\left%28x_{l-1}\right%29)

搜索neck时搜lateral connection，搜索Head时搜fc预测前的block

优化时增加FLOPS约束：![latex_equ](https://latex.codecogs.com/svg.latex?\mathrm{C}%28\alpha%29=\sum_{l}%20\sum_{o%20\in%20\mathcal{O}_{b}}%20\alpha_{l}^{o}%20\mathrm{FLOPs}%28o,%20l%29)

首先固定![latex_equ](https://latex.codecogs.com/svg.latex?\{\alpha,\beta,\gamma\})在一半的数据上训练求![latex_equ](https://latex.codecogs.com/svg.latex?\partial%20\mathcal{L}/\partial%20\mathcal{w})更新![latex_equ](https://latex.codecogs.com/svg.latex?w)，再固定![latex_equ](https://latex.codecogs.com/svg.latex?w)在剩下的数据上训练求![latex_equ](https://latex.codecogs.com/svg.latex?\partial\mathcal{L}/\partial\alpha,\partial\mathcal{L}/\partial\beta,\partial\mathcal{L}/\partial\gamma)更新架构![latex_equ](https://latex.codecogs.com/svg.latex?\alpha,\beta,\gamma)

---

## SaccadeNet: A Fast and Accurate Object Detector

![image-20200708225553326](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200708225553326.png)

<u>Center Attentive Module</u>产生中心点和类别；<u>Attention Transitive Module</u>产生wh；<u>Aggregation Attentive Module</u>融合两分支的特征（双线性插值）和预测，并对预测框进行refine；<u>Corner Attentive Module</u>只有一个block产生角点预测，用于backbone监督信号，测试时去掉。

相比CenterNet改进：1. 通过Aggregation进行二步refine 2. 增加corner分支的监督信号有助于对边缘的检测（帮助预测wh的分支） 3. L1作为box训练损失

![image-20200708230331922](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200708230331922.png)

👆<u>test-dev</u>上性能，速度提升明显，快于yolov3

---

## Learning from Noisy Anchors for One-stage Object Detection

> loss函数增加权重，选择回归分类分数高的框增大其loss贡献
>
> 对loss学习时的修改

之前的检测器评价anchor好坏使用IoU，harsh的划分导致噪声和不易训练

<img src="Figures/image-20200711120826876.png" alt="image-20200711120826876" style="zoom:30%;" />

拖车：IoU大但包含其他物体；长颈鹿：包含关键特征但IoU小

以IoU作为正负样本划分会导致噪声，需要根据<u>回归分类结果(cleanliness)</u>划分

**分类上**采用对cleanliness的预测代替之前根据IoU的pos/neg的0/1预测，RPN的**soft-label**，<u>类似IoU-Net的预测IoU/soft-objectness</u>

![latex_equ](https://latex.codecogs.com/svg.latex?c=\left\{\begin{array}{ll}\alpha%20\cdot%20\text{loc\_acc}+%281-\alpha%29%20\cdot%20\text%20{%20cls\_conf%20}%20&%20\text%20{%20for%20}%20b%20\in%20\mathcal{A}_{p%20o%20s}%20\\%200%20&%20\text%20{%20for%20}%20b%20\in%20\mathcal{A}_{n%20e%20g}\end{array}\right.)

对按IoU划分的![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{A}_{pos})中部分较高IoU的anchor计算cleanliness，即回归分类的性能

稳定训练，前几个iter采用anchor-gt的IoU而不是预测结果的IoU计算c

![latex_equ](https://latex.codecogs.com/svg.latex?L_{c%20l%20s}%20=\sum_{i}^{\mathcal{A}_{p%20o%20s}}%20r_{i}%20\operatorname{BCE}\left%28p_{i},%20c_{i}\right%29+\sum_{j}^{\mathcal{A}_{n%20e%20g}}%20\operatorname{BCE}\left%28p_{j},%20c_{j}\right%29)，用于RPN或objectness预测

**定位上**采用根据c**对![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{A}_{pos})中**anchor计算**reweight** r，对anchor回归<u>损失加权</u>。![latex_equ](https://latex.codecogs.com/svg.latex?\operatorname{loc\_a})为box和**匹配到的**GT的IoU（匹配用最大IoU方法）

![latex_equ](https://latex.codecogs.com/svg.latex?r=\left%28\alpha\cdot%20f\left%28\text{loc\_a}\right%29+%281-\alpha%29%20\cdot%20f\left%28\text{loc\_c}\right%29\right%29^{\gamma})，其中![latex_equ](https://latex.codecogs.com/svg.latex?f%28x%29=\frac{1}{1-x},\;\gamma=1)，为增大variance，并将![latex_equ](https://latex.codecogs.com/svg.latex?r)均值正则化到1

![latex_equ](https://latex.codecogs.com/svg.latex?%20L_{r%20e%20g}%20=\sum_{i}^{\mathcal{A}_{p%20o%20s}}%20r_{i}%20\operatorname{smooth}_{-}%20\ell_{1})

<img src="Figures/image-20200711123828365.png" alt="image-20200711123828365" style="zoom:50%;" />

---

## CSPNet: A New Backbone that can Enhance Learning Capability of CNN

> 使用卷积层截断梯度，减少直接相连，防止梯度重复计算

densenet中不断的concat会让梯度反传时后面的梯度不断传到前面，计算量大，重复计算

![latex_equ](https://latex.codecogs.com/svg.latex?\begin{aligned}%20x_{1}%20&=w_{1}%20*%20x_{0}%20\\%20x_{2}%20&=w_{2}%20*\left[x_{0},%20x_{1}\right]%20\\%20&%20\vdots%20\\%20x_{k}%20&=w_{k}%20*\left[x_{0},%20x_{1},%20\ldots,%20x_{k-1}\right]%20\end{aligned})

反传

![latex_equ](https://latex.codecogs.com/svg.latex?\begin{aligned}%20w_{1}^{\prime}%20&=f_{1}\left%28w_{1},%20\left\{%20g_{0}\right\}\right%29%20\\%20w_{2}^{\prime}%20&=f_{2}\left%28w_{2},\left\{g_{0},%20g_{1}\right\}\right%29%20\\%20w_{k}^{\prime}%20&=f_{k}\left%28w_{k},\left\{g_{0},g_{1},%20\ldots,%20g_{k-1}\right\}\right%29%20\end{aligned})

例如在最后一层计算过程中，由于和前面层直接连接，梯度会一直回传到w1

提出：**Partial Dense Block** 输入特征channel上分为两部分![latex_equ](https://latex.codecogs.com/svg.latex?\left[x_0^\prime,\;x_0^{\prime\prime}\right])，一部分直接相连，另外部分dense block。**Partial Transition Layer** 融合时采用transition层截断梯度的传播

![image-20200724223413132](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200724223413132.png)

![latex_equ](https://latex.codecogs.com/svg.latex?\begin{aligned}%20x_{k}%20&=w_{k}%20*\left[x_{0}^{\prime\prime},%20x_{1},%20\ldots,%20x_{k-1}\right]%20\\%20x_{T}%20&=w_{T}%20*\left[x_{0}^{\prime\prime},%20x_{1},%20\ldots,%20x_{k}\right]%20\\%20x_{U}%20&=w_{U}%20*\left[x_{0}^{\prime},%20x_{T}\right]%20\\%20w_{k}^{\prime}%20&=f_{k}\left%28w_{k},\left\{g_{0}^{\prime%20\prime},%20g_{1},%20\ldots,%20g_{k-1}\right\}\right%29%20\\%20w_{T}^{\prime}%20&=f_{T}\left%28w_{T},\left\{g_{0}^{\prime%20\prime},%20g_{1},%20\ldots,%20g_{k}\right\}\right%29%20\\%20w_{U}^{\prime}%20&=f_{U}\left%28w_{U},\left\{g_{0}^\prime,%20g_{T}\right\}\right%29%20\end{aligned})

![latex_equ](https://latex.codecogs.com/svg.latex?w_{U})的梯度只传递到![latex_equ](https://latex.codecogs.com/svg.latex?w_T)，之后梯度全部从![latex_equ](https://latex.codecogs.com/svg.latex?w_T)开始传

![image-20200724222732419](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200724222732419.png)

Fusion first能截断更多梯度传递（如后续部分到part1的梯度），但性能较差；采用fusion last

提出EFM，临近层特征融合，防止distract

![image-20200724223519751](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200724223519751.png)

检测任务性能提升明显

---

## Feature Pyramid Transformer

> 采用transformer/self-attention方式进行多尺度特征融合/交互

Non-local只对同一尺度的不同空间特征交互，改进space+scale

![image-20200818143118059](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200818143118059.png)

不同层的不同空间位置的物体interaction（co-occurring in multiple scales）

#### Non-local

用于一张特征图不同位置特征。看作self-attention，输入![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{q}_i=f_q%28\mathbf{X}_i%29)表示![latex_equ](https://latex.codecogs.com/svg.latex?i)位置<u>经过![latex_equ](https://latex.codecogs.com/svg.latex?f_q)提取后</u>的query，![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{k}_j=f_k%28\mathbf{X}_j%29)为key，![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{v}_j=f_v%28\mathbf{X}_j%29)为![latex_equ](https://latex.codecogs.com/svg.latex?j)位置的value

先dot product计算i-query和j-key相似度，softmax标准化得到权重，和j-value相乘的到结果

![latex_equ](https://latex.codecogs.com/svg.latex?\begin{aligned}%20\text%20{%20Input:%20}%20&%20\mathbf{q}_{i},%20\mathbf{k}_{j},%20\mathbf{v}_{j}%20\\%20\text%20{%20Similarity:%20}%20&%20s_{i,%20j}=F_{\text%20{sim}}\left%28\mathbf{q}_{i},%20\mathbf{k}_{j}\right%29%20\\%20\text%20{%20Weight:%20}%20&%20w_{i,%20j}=F_{\text%20{nom}}\left%28s_{i,%20j}\right%29%20\\%20\text%20{%20Output:%20}%20&%20\tilde{\mathbf{X}}_{i}=F_{\text%20{mul}}\left%28w_{i,%20j},%20\mathbf{v}_{j}\right%29,%20\end{aligned})

#### Self-Transformer (ST)

用于一张图上co-occurring物体特征。把query和key分成![latex_equ](https://latex.codecogs.com/svg.latex?N)份，每部分计算相似度![latex_equ](https://latex.codecogs.com/svg.latex?s_{i,%20j}^{n}=F_{s%20i%20m}\left%28\mathbf{q}_{i,%20n},%20\mathbf{k}_{j,%20n}\right%29)，使用Mixture of Softmax作为标准化函数（加权求和![latex_equ](https://latex.codecogs.com/svg.latex?\pi_n)）

![latex_equ](https://latex.codecogs.com/svg.latex?w_{i,j}%20=%20F_{\operatorname{mos}}%28s_{i,%20j}^{n}%29=\sum_{n=1}^{\mathcal{N}}%20\pi_{n}%20\frac{\exp%20%28s_{i,%20j}^{n}%29}{\sum_{j}%20\exp%20%28s_{i,%20j}^{n}%29})

<img src="Figures/image-20200818145143482.png" alt="image-20200818145143482" style="zoom:50%;" />

#### Grounding Transformer (GT)

top-down融合，ground高层concept特征到低层pixel特征，高层![latex_equ](https://latex.codecogs.com/svg.latex?k\;v)低层![latex_equ](https://latex.codecogs.com/svg.latex?q)

采用负欧式距离计算不同层之间特征相似度 ![latex_equ](https://latex.codecogs.com/svg.latex?F_{\text%20{eud}}\left%28\mathbf{q}_{i},%20\mathbf{k}_{j}\right%29=-\left\|\mathbf{q}_{i}-\mathbf{k}_{j}\right\|^{2})

![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{q}_i=f_q%28\mathbf{X}_i^f%29)表示![latex_equ](https://latex.codecogs.com/svg.latex?f)层![latex_equ](https://latex.codecogs.com/svg.latex?i)位置的特征，![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{k}_j=f_K%28\mathbf{X}_j^c%29)表示![latex_equ](https://latex.codecogs.com/svg.latex?c)层![latex_equ](https://latex.codecogs.com/svg.latex?j)位置，计算的到![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{\hat{X}}_i^f)

分割任务需要局部信息，传统采用直接相加，而用GT会带来全局信息，提出**Locality-constrained Grounding Transformer**，![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{q}_i)只和一部分![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{k}_j\;\mathbf{v}_j)交互

<img src="Figures/image-20200818145429855.png" alt="image-20200818145429855" style="zoom:50%;" />

#### Rendering Transformer (RT)

bottom-up, rendering high-level concept with low-level pixels

低层的![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{K})和![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{V})，高层为![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{Q})

 ![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{K})首先GAP计算为权重![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{w})，权重![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{w})再和![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{Q})相乘refine，最后和downsample (conv+stride)的![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{V})相加

![latex_equ](https://latex.codecogs.com/svg.latex?\begin{aligned}%20\text%20{%20Input:%20}%20&%20\mathbf{Q},%20\mathbf{K},%20\mathbf{V}%20\\%20\text%20{%20Weight:%20}%20&%20\mathbf{w}=G%20A%20P%28\mathbf{K}%29%20\\%20\text%20{%20Weight%20Query:%20}%20&%20\mathbf{Q}_{a%20t%20t}=F_{a%20t%20t}%28\mathbf{Q},%20\mathbf{w}%29%20\\%20\text{Down-sampled%20Value:%20}%20&%20\mathbf{V}_{d%20o%20w}=F_{sconv}%28\mathbf{V}%29\\%20\text{Output:%20}%20&%20\hat{\mathbf{X}}^{c}=F_{a%20d%20d}\left%28F_{\text%20{conv%20}}\left%28\mathbf{Q}_{a%20t%20t}\right%29,%20\mathbf{V}_{\text%20{dow}}\right%29\end{aligned})

#### 总体架构

resnet，faster rcnn head

![image-20200818150612705](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200818150612705.png)

高层计算用于GT的特征，低层计算用于RT的特征，transformer融合的到新特征，concat

参数量，计算量增加大

---

## BorderDet: Border Feature for Dense Object Detection

> explicit border information

![image-20200819230010228](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200819230010228.png)

borderdet增强了边缘的特征

#### Border Align

原始特征通道数为C，构建5C通道的特征，表示上下左右特征和原始点的特征。输入coarse box reg的bbox。

对于每个ij点，4C分别选择其对应预测框的边上响应最大的特征作为这个点的输出特征（+原始特征=5C x W x H特征图）

![latex_equ](https://latex.codecogs.com/svg.latex?F_{c}%28i,%20j%29=\left\{\begin{array}{ll}I_{c}%28i,%20j%29%20&%200%20\leq%20c<C%20\\%20\max%20_{0%20\leq%20k%20\leq%20N-1}\left%28I_{c}\left%28x_{0},%20y_{0}+\frac{k%20h}{N}\right%29\right%29%20&%20C%20\leq%20c<2%20C%20\\%20\max%20_{0%20\leq%20k%20\leq%20N-1}\left%28I_{c}\left%28x_{0}+\frac{k%20w}{N},%20y_{0}\right%29\right%29%20&%202%20C%20\leq%20c<3%20C%20\\%20\max%20_{0%20\leq%20k%20\leq%20N-1}\left%28I_{c}\left%28x_{1},%20y_{0}+\frac{k%20h}{N}\right%29\right%29%20&%203%20C%20\leq%20c<4%20C%20\\%20\max%20_{0%20\leq%20k%20\leq%20N-1}\left%28I_{c}\left%28x_{0}+\frac{k%20w}{N},%20y_{1}\right%29\right%29%20&%204%20C%20\leq%20c<5%20C\end{array}\right.)

![latex_equ](https://latex.codecogs.com/svg.latex?%28i,j%29)的coarse预测结果为![latex_equ](https://latex.codecogs.com/svg.latex?%28x_0,y_0,x_1,y_1%29)

![image-20200819231021412](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200819231021412.png)

![image-20200819231044269](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200819231044269.png)

**Border Alignment Module**对特征升维，BorderAlign，降维👇

![image-20200819231203928](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200819231203928.png)

BorderDet分别在定位和分类分支使用BAM，BAM使用定位分支的第一步的定位结果，再去refine，组合成为最后输出👇

<img src="Figures/image-20200819231428705.png" alt="image-20200819231428705" style="zoom:50%;" />

---

## SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization

> NAS+检测，encoder+decoder结构不好 (scale decreased model)
>
> 尺度增大减小，不同尺度特征连接

提出scale-permuted model，保证：1. 特征图尺度可以随时增大减小 2. 不同尺度的特征可以连接进行融合

直接搜索整个网络，而不是分别搜索backbone和fpn

![image-20200820204545461](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200820204545461.png)

逐步替换resnet👇，红色为输出层，可以看作**backbone+FPN**

![image-20200820205947836](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200820205947836.png)

Multi-scale, DropBlock, stochastic depth, swish activation

性能提升明显 mAP=52.1

---

## Dually Supervised Feature Pyramid for Object Detection and Segmentation

> FPN top-down和botton-up两次都预测监督算loss
>
> 拆分cls和reg head

1. ![image-20200823190552173](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200823190552173.png)

2. ![image-20200823190617471](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200823190617471.png)

---

## Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition

> 一个卷积拆分为多个不同receptive field的分组卷积

![image-20200831204426110](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200831204426110.png)

Bottom-up: 感受野逐渐增大，kernel深度![latex_equ](https://latex.codecogs.com/svg.latex?FM_{out-i})逐渐减小

kernel depth (输出通道数): ![latex_equ](https://latex.codecogs.com/svg.latex?\left\{F%20M_{i},%20\frac{F%20M_{i}}{\left%28\frac{K_{2}^{2}}{K_{1}^{2}}\right%29},%20\frac{F%20M_{i}}{\left%28\frac{K_{3}^{2}}{K_{1}^{2}}\right%29},%20\ldots,%20\frac{F%20M_{i}}{\left%28\frac{K_{n}^{2}}{K_{1}^2}\right%29}\right\})

对SSD模型有提升

---

## Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

> 

1. 训练时cls-score和iou/objectness-score单独预测但测试时直接相乘
2. bbox没有建模uncertainty

#### QFL

![image-20200905160302886](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200905160302886.png)

👆IoU(quality)和cls-score不match，👇独立预测

![image-20200905160358215](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200905160358215.png)

![image-20200905160432718](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200905160432718.png)

👆传统为冲击函数(Dirac delta dist.)，无法建模不确定性

提出预测分类-质量联合表示（smooth label），离散label，将Focal Loss的优化改 ![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{Q%20F%20L}%28\sigma%29=-|y-\sigma|^{\beta}%28%281-y%29%20\log%20%281-\sigma%29+y%20\log%20%28\sigma%29%29) 「展开pos+neg」

#### DFL

![image-20200906105358216](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200906105358216.png)

👆之前是冲击函数，只有标注y出概率最大，其他为0，对于不确定的边界没有监督

![image-20200906110136608](https://github.com/AlphaGoMK/Collections/tree/master/Notes/Figures/image-20200906110136608.png)

👆不确定边界 bottom，双峰

<img src="Figures/image-20200906110316101.png" alt="image-20200906110316101" style="zoom:50%;" />

传统 ![latex_equ](https://latex.codecogs.com/svg.latex?\hat{y}=\int_{-\infty}^{+\infty}%20\delta%28x-y%29%20x%20\mathrm{d}%20x)，冲击函数，只预测一点

提出通过分布![latex_equ](https://latex.codecogs.com/svg.latex?P)预测label ![latex_equ](https://latex.codecogs.com/svg.latex?\hat{y}=\int_{-\infty}^{+\infty}%20P%28x%29%20x%20\mathrm{d}%20x=\int_{y_{0}}^{y_{n}}%20P%28x%29%20x%20\mathrm{d}%20x) 分布的积分

从冲击函数，到先验高斯分布，到任意分布，建模**随机性**

<img src="Figures/image-20200906110336213.png" alt="image-20200906110336213" style="zoom:50%;" />

离散化 ![latex_equ](https://latex.codecogs.com/svg.latex?\hat{y}=\sum_{i=0}^{n}%20P\left%28y_{i}\right%29%20y_{i})，![latex_equ](https://latex.codecogs.com/svg.latex?\sum_{i=0}^nP%28y_i%29=1)

减少计算量，真实分布与标注位置距离不会太远，只计算左右最近两个![latex_equ](https://latex.codecogs.com/svg.latex?y_l,\;y_r)

![latex_equ](https://latex.codecogs.com/svg.latex?\hat{y}=\sum_{j=0}^{n}%20P\left%28y_{j}\right%29%20y_{j}=\mathcal{S}_{i}%20y_{i}+\mathcal{S}_{i+1}%20y_{i+1}=\frac{y_{i+1}-y}{y_{i+1}-y_{i}}%20y_{i}+\frac{y-y_{i}}{y_{i+1}-y_{i}}%20y_{i+1})

![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{D%20F%20L}\left%28\mathcal{S}_{i},%20\mathcal{S}_{i+1}\right%29=-\left%28\left%28y_{i+1}-y\right%29%20\log%20\left%28\mathcal{S}_{i}\right%29+\left%28y-y_{i}\right%29%20\log%20\left%28\mathcal{S}_{i+1}\right%29\right%29)

最后预测为左右最近两个位置的线性组合 ![latex_equ](https://latex.codecogs.com/svg.latex?\hat{y}=y_l*p_{y_l}+y_r*p_{y_r}\;\left%28p_{y_l}+p_{y_r}=1\right%29)

#### GFL

QFL和DFL的统一表示 Generalized Focal Loss

![latex_equ](https://latex.codecogs.com/svg.latex?\mathbf{G%20F%20L}\left%28p_{y_{l}},%20p_{y_{r}}\right%29=-\left|y-\left%28y_{l}%20p_{y_{l}}+y_{r}%20p_{y_{r}}\right%29\right|^{\beta}\left%28\left%28y_{r}-y\right%29%20\log%20\left%28p_{y_{l}}\right%29+\left%28y-y_{l}\right%29%20\log%20\left%28p_{y_{r}}\right%29\right%29)

训练损失

![latex_equ](https://latex.codecogs.com/svg.latex?\mathcal{L}=\frac{1}{N_{p%20o%20s}}%20\sum_{z}%20\mathcal{L}_{\mathcal{QFL}}+\frac{1}{N_{p%20o%20s}}%20\sum_{z}%20\mathbf{1}_{\left\{c_{z}^{*}>0\right\}}\left%28\lambda_{0}%20\mathcal{L}_{\mathcal{Bbox}}+\lambda_{1}%20\mathcal{L}_{\mathcal{DFL}}\right%29)

对所有位置![latex_equ](https://latex.codecogs.com/svg.latex?z)计算QFL，对正样本![latex_equ](https://latex.codecogs.com/svg.latex?c^*_z>0)计算GIoU loss和DFL

DFL和IoU-loss优化bbox回归，分别建模uncertainty和IoU最大；QFL优化分类分支

对大目标提升效果好，+1 mAP

Ref: https://zhuanlan.zhihu.com/p/147691786

---
