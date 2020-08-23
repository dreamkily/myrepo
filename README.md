### Paper
#### 1.ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation
用像素级别的熵损失来处理语义分割非监督域适应（unsupervised domain adaptation）问题
**概述**
文章提出了一种无监督的域适应方法来解决将算法应用到真实世界中，训练集和测试集之间的差异，包含两种互补的方法： （1）直接应用熵损失， （2）对抗熵损失

**亮点**
作者注意到在源域unsupervised training的输出分割图置信度高，熵值低，迁移到目标域的图片分割置信度低，熵值高。因此作者认为一个可能的方法来缓解domain gap就是使得target prediction有高置信度（低熵）。作者提出了2个方法： （1）直接使用entropy loss最小化目标域分割图的熵值，独立的对所有的像素点施加约束； （2）用adversarial loss间接的最小化熵值，考虑到源域和目标域的结构适应。
![Alt text](./1596770299896.png)
首先是共有的source domain supervised training部分，$P_{x}$代表分割网络输出**soft-segmentation map**，在分割图的每个空间点处可以视为一个C维的向量，该向量就代表输出类别的离散分布，如果某个类别的置信度高，则该点的熵值就低，但如果每个类别分数接近，则熵值就会比较高，$F$代表分割网络参数。
\begin{array}{l}
\min _{G} \frac{1}{n_{s}} \sum_{i=1}^{n_{s}} L\left(G\left(\mathbf{x}_{i}^{s}\right), \mathbf{y}_{i}^{s}\right) \\
\quad+\frac{\lambda}{n_{s}} \sum_{i=1}^{n_{s}} e^{-H\left(\mathbf{g}_{i}^{s}\right)} \log \left[D\left(T\left(\mathbf{h}_{i}^{s}\right)\right)\right]+\frac{\lambda}{n_{t}} \sum_{j=1}^{n_{t}} e^{-H\left(\mathbf{g}_{j}^{t}\right)} \log \left[1-D\left(T\left(\mathbf{h}_{j}^{t}\right)\right)\right] \\
\max _{D} \frac{1}{n_{s}} \sum_{i=1}^{n_{s}} e^{-H\left(\mathbf{g}_{i}^{s}\right)} \log \left[D\left(T\left(\mathbf{h}_{i}^{s}\right)\right)\right]+\frac{1}{n_{t}} \sum_{j=1}^{n_{t}} e^{-H\left(\mathbf{g}_{j}^{t}\right)} \log \left[1-D\left(T\left(\mathbf{h}_{j}^{t}\right)\right)\right]
\end{array}
**方法1： 直接熵最小化**
输入目标域图片至分割网络，得到归一化的熵图：
$$\boldsymbol{E}_{\boldsymbol{x}_{t}}^{(h, w)}=\frac{-1}{\log (C)} \sum_{c=1}^{C} \boldsymbol{P}_{\boldsymbol{x}_{t}}^{(h, w, c)} \log \boldsymbol{P}_{\boldsymbol{x}_{t}}^{(h, w, c)}$$
**entropy loss**
$$\mathcal{L}_{e n t}\left(\boldsymbol{x}_{t}\right)=\sum_{h, w} \boldsymbol{E}_{\boldsymbol{x}_{t}}^{(h, w)}$$
联合训练，同步地在源域优化有监督地分割损失和在目标域优化无监督的熵损失：
$$\min _{\theta_{F}} \frac{1}{\left|\mathcal{X}_{s}\right|} \sum_{\boldsymbol{x}_{s}} \mathcal{L}_{s e g}\left(\boldsymbol{x}_{s}, \boldsymbol{y}_{s}\right)+\frac{\lambda_{e n t}}{\left|\mathcal{X}_{t}\right|} \sum_{\boldsymbol{x}_{t}} \mathcal{L}_{e n t}\left(\boldsymbol{x}_{t}\right)$$
方法2：对抗学习熵最小化
仅仅优化方法1会忽略局部语义之间的结构化依耐性，因为源域和目标域通常有很强的语义布局相似性，因此在结构化的输出空间adaptation是有益的，方法2通过使得目标的熵图与源域熵图相近，间接的最小化熵图，利用了域之间的结构一致性。
作者将方法1中的熵自带权重的自信息代替。
$$\boldsymbol{I}_{\boldsymbol{x}}^{(h, w)}=-\boldsymbol{P}_{\boldsymbol{x}}^{(h, w)} \cdot \log \boldsymbol{P}_{\boldsymbol{x}}^{(h, w)}$$
为了使得目标域语义图的熵与源域接近，作者引入一个判别器D，D接受自信息I，判断I来自哪个域，分割网络为了迷惑判别器，必须尽可能使得目标域的输出语义熵分布与源域接近，判别器与分割网络对抗目标的优化方式如下：
$$\min _{\theta_{D}} \frac{1}{\left|\mathcal{X}_{s}\right|} \sum_{\boldsymbol{x}_{s}} \mathcal{L}_{D}\left(\boldsymbol{I}_{\boldsymbol{x}_{s}}, 1\right)+\frac{1}{\left|\mathcal{X}_{t}\right|} \sum_{\boldsymbol{x}_{t}} \mathcal{L}_{D}\left(\boldsymbol{I}_{\boldsymbol{x}_{t}}, 0\right)
$$
$$\min _{\theta_{F}} \frac{1}{\left|\mathcal{X}_{t}\right|} \sum_{\boldsymbol{x}_{t}} \mathcal{L}_{D}\left(\boldsymbol{I}_{\boldsymbol{x}_{t}}, 1\right)$$
得到方法2的分割网络的总优化函数：
$$\min _{\theta_{F}} \frac{1}{\left|\mathcal{X}_{s}\right|} \sum_{\boldsymbol{x}_{s}} \mathcal{L}_{s e g}\left(\boldsymbol{x}_{s}, \boldsymbol{y}_{s}\right)+\frac{\lambda_{a d v}}{\left|\mathcal{X}_{t}\right|} \sum_{\boldsymbol{x}_{t}} \mathcal{L}_{D}\left(\boldsymbol{I}_{\boldsymbol{x}_{t}}, 1\right)$$
训练时，交替训练判别器D和分割网络F。
**总结**
提出了2中基于熵的unsupervised domain adaptation方法，利用对抗学习缩小一种额外的度量值得学习。


#### 2. Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
**Motivation**
文章的出发点在于，现在的领域自适应方法在适配过程中，并没有考虑任务特异性的决策边界，并且它们只是完全把两个domain的特征进行对齐，而忽略了其实每个domain都有各自的特点，任务决策边界(学习到的预测函数对于不同的决策目标，都应该扮演不同的角色，也就是说要把他们的特点考虑进来)
**Method**
用源域训练的网络如果用到目标域上，肯定因为目标域与源域的不同，效果也会有所不同，有的样本效果好，有的效果差，效果好的我们就不管了，重点关注效果不好的，因为这才能体现出领域的差异性，为了找到这些效果差的样本，作者引入了两个独立的分类器$F_{1}$和$F_{2}$，用二者的分期表示置信度不高，需要重新训练。
通俗点来说就是，如果$F_{1}$和$F_{2}$在一个目标样本上的结果不一致，就认为发生了分歧，此时就需要后续的介入，如果认为结果一致，认为这个样本比较好分，就先不用管。
![Alt text](./1596853544211.png)

1.训练的第一个阶段呢，产生了一些阴影，我们的目标是要最大化阴影区域(因为这样可以针对不同的分类器学习到不同的特征表示)
2.最大化两个分类器差异的结果是第二个图，此时的阴影比较大，所以我们的目标就是通过提取更好的特征来减少两个分类器的分歧
3.最小化分歧的结果是第三个图，此时阴影几乎不存在了，但问题是决策边界还是有一些紧凑，可能特征不太鲁棒，那就和上一步交替优化
4.交替优化的目标是第四个图，此时，阴影不存在，并且，决策边界也比较鲁棒。

**学习过程**
学习过程也和上面的图一一对应，作者将其分为A，B，C三个阶段，
+ 在A阶段，目标是首先训练出两个不同的分类器$F_{1}$和$F_{s}$。训练方式很简单，求源域上的分类误差即可。公式如下：
$$\min \mathcal{L}\left(X_{s}, Y_{s}\right)=-\mathbb{E}_{\left(\mathbf{x}_{s}, y_{s}\right) \sim\left(X_{s}, Y_{s}\right)} \sum_{k=1}^{K} \mathbb{I}_{\left[k=y_{s}\right.} \log p\left(\mathbf{y} \mid \mathbf{x}_{s}\right)$$
+ 在B阶段，固定特征提取器G，训练两个不同的分类器$F_{1}$和$F_{2}$，使得它们的差异最大。两个分类器的差异用最简单的$L_{1}$损失来衡量，优化目标如下：
$$\min _{F_{1}, F_{2}} \mathcal{L}\left(X_{s}, Y_{s}\right)-\mathcal{L}_{\text {adv }}\left(X_{t}\right)$$
$$\mathcal{L}_{\text {adv }}\left(X_{t}\right)=\mathbb{E}_{\mathbf{x}_{t} \sim X_{t}}\left[d\left(p_{1}\left(\mathbf{y} \mid \mathbf{x}_{t}\right), p_{2}\left(\mathbf{y} \mid \mathbf{x}_{t}\right)\right)\right]$$
$$d\left(p_{1}, p_{2}\right)=\frac{1}{K} \sum_{k=1}^{K}\left|p_{1 k}-p_{2 k}\right|$$
+ 在C阶段，与B阶段相反，固定两个分类器，优化特征生成器G，使得特征对两个分类器效果尽可能一样：
$$\min _{G} \mathcal{L}_{\mathrm{adv}}\left(X_{t}\right)$$
B阶段和C阶段的训练过程可以用下图来表示

![Alt text](./1596854178261.png)

**总结**
文章的方法思想非常简单，实现也很容易，但是效果却比那些重量级的、需要反复调整参数的方法有效的多，入选2018年CVPR的oral文章。
#### 3. Catastrophic Forgetting Meetings Negative Transfer: Batch Spectral Shrinkage for Safe Transfer Learning
降低奇异值增强模型的泛化能力
**abstract**
这篇文章主要针对模型的fine-tune问题进行优化，众所周知，在许多模型的训练中，使用预训练好的模型进行fine-tune可以使得模型的训练更加容易且效果更好，但是因为两个原因，灾难性遗忘和负迁移，使得fine-tune的效果降低了，本文提出了一种方法batch spectral shrinkage来克服这一情况。

**Introduction**
主要介绍两种导致fine-tune效果变差的原因，首先是灾难性遗忘，即模型在学习与目标任务相关的信息时，容易突然失去之前所学的知识，导致过度拟合，第二个是负迁移，并非所有预先训练得到的知识都可以跨域进行迁移，且不加选择的迁移所有的知识对模型来说是有害的。
首先作者提到了在优化fine-tune的先驱者们提出的算法，L2指的是在训练时直接正则化参数，L2-SP认为应该使用预训练模型中的信息来正则化某些参数，而非直接正则化所有参数，DELTA使用了注意力机制，使用了feature map中的知识。
基于奇异值分解，作者提出了batch spectral shrinkage(BSS)，来指引模型中参数和特征的可迁移性，进而增强模型迁移后的性能。
**算法**
$\sigma$是网络提出的特征的奇异值，这里用了$-i$是因为从后往前数，思想就是把奇异值小的部分给压制住。
$$L_{\mathrm{bss}}(F)=\eta \sum_{i=1}^{k} \sigma_{-i}^{2}$$
因此网络总体的loss就是：
$$\min _{\mathbf{W}} \sum_{i=1}^{n} L\left(G\left(F\left(\mathbf{x}_{i}\right)\right), y_{i}\right)+\Omega(\mathbf{W})+L_{\mathrm{bss}}(F)$$
#### 4. Deep Metric Learning via Lifted Structured Feature Embedding
**摘要**
&#8195;&#8195;本文提出一种距离度量的方法，充分的发挥training batches的优势，by lifting the vector of pariwise distances within the batch to the matrix of pairwise distances
引言部分开头讲了距离相似性度量的重要性，并且应用广泛，这里提到了三元组损失函数，就是讲在训练的过程中，尽可能地拉近两个相同物体之间地距离，而拉远不同物体之间地距离；这种做法会比普通地训练方法得到更好地效果。但是，文章中提到，现有地三元组方法无法充分利用哦个minibatch SGD training的training batches的优势，现有的方法首先随机的采样图像对或者三元组，构建训练batches，计算每一个pairs or triplets的损失。本文提出了一种方法，称为:lifts，将the vector of pairwise distance转换成the matrix of pairwise distance，然后再lifts problem上设计了一个新的结构损失目标，结果表明，在GoogleLeNet network上取得了比其他方法都要好的结果。
我们基于训练集合的正负样本，定义了一个结构化的损失函数：
$$\begin{aligned} J &=\frac{1}{2|\widehat{\mathcal{P}}|} \sum_{(i, j) \in \widehat{\mathcal{P}}} \max \left(0, J_{i, j}\right)^{2} \\ J_{i, j} &=\max \left(\max _{(i, k) \in \mathcal{N}} \alpha-D_{i, k}, \max _{(j, l) \in \mathcal{N}} \alpha-D_{j, l}\right)+D_{i, j} \end{aligned}$$
其中，$P$是正样本的集合，$N$是负样本的集合，这个函数提出了两个计算上的挑战
1：非平滑(non-smooth)
2：评价和计算其子梯度需要最小化所有样本对若干次
我们以两种方式解决了上述挑战：
首先，我们优化上述函数的一个平滑上界
第二，对于大数据常用的方式类似，我们采用随机的方法
然而，前人的工作都是用SGD的方法，随机的均匀的选择pairs or triplets。我们的方法从这之中得到了借鉴。
(1) it biases the sample towards including difficult pairs
(2) 一次采样就充分的利用了一个mini-batch的全部信息，而不仅仅是两个pair之间的信息。
所以，我们的方法改为并非完全随机，而是引入了重要性的元素，我们随机的采样了一些positive pairs，然后添加了一些他们的difficult neighbors来训练mini-batch， 这个增加了子梯度会用到的相关信息。
此外，搜索single hardest negative with nested max function实际上会导致网络收敛到一个bad local optimum，所以我们采用了如下的smooth upper bound，所以我们的损失函数定义为：
$$\begin{aligned} \tilde{J}_{i, j} &=\log \left(\sum_{(i, k) \in \mathcal{N}} \exp \left\{\alpha-D_{i, k}\right\}+\sum_{(j, l) \in \mathcal{N}} \exp \left\{\alpha-D_{j, l}\right\}\right)+D_{i, j} \\ \tilde{J} &=\frac{1}{2|\mathcal{P}|} \sum_{(i, j) \in \mathcal{P}} \max \left(0, \tilde{J}_{i, j}\right)^{2} \end{aligned}$$
其中，P是batch中positive pairs集合，N是negative pairs的集合，后向传播梯度计算如下所示：
$$\frac{\partial \tilde{J}}{\partial D_{i, j}}=\frac{1}{|\mathcal{P}|} \tilde{J}_{i, j} \mathbb{1}\left[\tilde{J}_{i, j}>0\right]$$
$$\frac{\partial \tilde{J}}{\partial D_{i, k}}=\frac{1}{|\mathcal{P}|} \tilde{J}_{i, j} \mathbb{1}\left[\tilde{J}_{i, j}>0\right] \frac{-\exp \left\{\alpha-D_{i, k}\right\}}{\exp \left\{\tilde{J}_{i, j}-D_{i, j}\right\}}$$
$$\frac{\partial \tilde{J}}{\partial D_{j, l}}=\frac{1}{|\mathcal{P}|} \tilde{J}_{i, j} \mathbb{1}\left[\tilde{J}_{i, j}>0\right] \frac{-\exp \left\{\alpha-D_{j, l}\right\}}{\exp \left\{\tilde{J}_{i, j}-D_{i, j}\right\}}$$
#### 5. Semi-supervised Domain Adaptation via Minimax Entropy
&#8196;&#8195;现阶段，关于半监督领域自适应学习的论文文献依然数量比较少。
作者一开始指出UDA能够通过匹配分布的方法来提升模型对目标域无标签样本的泛化性，但是却在目标域上无法学习到具有判别性的'类别边界'。作者就提出在目标域训练样本中增加少量的有标签样本来实现获取模型对目标域具有区别性的特征，并提出了一种叫做Minimax Entropy(MME)的方法来实现此目标。MME方法是一种基于对无标记数据的条件熵以及任务损失的优化极小极大损失，他能够减小分布差异，又能学习任务具有区别性的特征，作者使用这种方法来评估每一种类别具有代表性的数据点，以及提取判别性特征。
####6. Conditional Adversarial Domain Adaptation
**背景**
对抗学习已经被嵌入到深度网络中通过学习到可迁移的特征进行适应，并取得了不错的成果，作者指出当前的一些对抗域适应方法仍然存在问题：1.只是独立的对齐特征而没有对齐标签，而这往往是不充分的 2.当数据分布体现出复杂的多模态结构时，对抗性自适应方法可能无法捕获这种多模态结构，换句话说即使网络训练收敛，判别器完全被混淆，分辨不出样本来自哪个域，也无法保证此时源域和目标域足够相似(没有捕获到数据的多模态结构)。3.条件域判别器中使用最大最小优化方法也许存在一定的问题，因为与判别器强制不同的样本具有相同的重要性，然而那些不确定预测的难迁移样本也许会对抗适应产生不良的影响，作者提出的条件对抗域适应网络(CDANs)在一定的程度上解决了三个问题，针对1，CDAN通过对齐特征-类别的联合分布解决，针对2，CDAN使用了Multilinear Condtioning多线性调整的方法来解决，针对3，作者提出了在目标函数中添加Entropy Conditioning熵调整来解决。
**CDAN结构**
<center>
<img src="https://img-blog.csdnimg.cn/20181225121528124.png" width="50%" height="50%" />
Figure 1. CDAN图
</center>
&#8195;&#8195;上述框架很好理解，框架的前端通过深度神经网络，比如AlexNet/ResNet对源域和目标域提取特征$f$，然后得出预测结构$g$，$g$得出预测标签。
&#8196;&#8195;在最近的条件生成对抗网络Conditional Generative Adversarial Networks(CGAN)中揭示了不同的分布可以在相关信息上调整生成器和判别器匹配的更好，例如：将标签和附属状态关联，CGAN可以在具有高可变性和多模态的分布数据集上生成全局一致图像，收到CGAN的启示，作者观察到在对抗域适应中，<font face="微软雅黑" color='red'>分类器预测结果g中携带了潜在的揭露了多模态结果的判别信息</font>，这可以在对齐特征$f$时用于调整，从而在网络训练过程中捕获多模态信息，通过链接变量$h=(f,g)$在分类器预测结果$g$上调整域判别器$D$,这样可以解决上面所说的前两个问题，而最简单的一种连接方式就是$f \oplus g$，将$f \oplus g$丢入到判别器$D$中，这种连接策略被现有的CGANs方法中广泛的采用，然而这种连接策略中，$f$和$g$是相互独立的，导致了不能很好的捕捉到特征与分类器预测结果之间的相乘交互，而这对于域适应是至关重要的，作为结果，分类器预测中传达的多模态信息不能被充分利用来匹配复杂域的多峰分布。
**熵调整**
&#8196;&#8195;条件域判别器中使用最大最小优化方法也许存在一定的问题，因为与判别器强制不同的样本具有相同的重要性，然而那些不确定预测的难迁移样本也许会对对抗适应产生不良的影响。为了减少这种影响，作者通过熵$H(g)=-\sum_{c=1}^{C} g_{c} \log \left(g_{c}\right)$来定量分类器预测结果的不确定性，而预测结果的确定性则可以被计算为$e^{-H(g)}$。然后通过这种基于熵的确定性策略调整域判别器，然后最终的CDAN使用的minimax的目标函数则为：
$$\begin{aligned} \min _{G} & \frac{1}{n_{s}} \sum_{i=1}^{n_{s}} L\left(G\left(\mathbf{x}_{i}^{s}\right), \mathbf{y}_{i}^{s}\right) \\+& \frac{\lambda}{n_{s}} \sum_{i=1}^{n_{s}} e^{-H\left(\mathbf{g}_{i}^{s}\right)} \log \left[D\left(T\left(\mathbf{h}_{i}^{s}\right)\right)\right]+\frac{\lambda}{n_{t}} \sum_{j=1}^{n_{t}} e^{-H\left(\mathbf{g}_{j}^{t}\right)} \log \left[1-D\left(T\left(\mathbf{h}_{j}^{t}\right)\right)\right] \\ \max _{D} & \frac{1}{n_{s}} \sum_{i=1}^{n_{s}} e^{-H\left(\mathbf{g}_{i}^{s}\right)} \log \left[D\left(T\left(\mathbf{h}_{i}^{s}\right)\right)\right]+\frac{1}{n_{t}} \sum_{j=1}^{n_{t}} e^{-H\left(\mathbf{g}_{j}^{t}\right)} \log \left[1-D\left(T\left(\mathbf{h}_{j}^{t}\right)\right)\right] \end{aligned}$$
#### 6. GCAN: Graph Convolutional Adversarial Network for Unsupervised Domain Adaptation
**主要思想**
为实现源域和目标域之间的迁移学习或特征对齐，作者认为有三种重要的信息类型需要学习：数据结构、域标签和类标签。提出了一种端到端的图卷积对抗网络(GCAN)，通过在同意的深度模型中对数据结构、域标签或类标签进行联合建模，实现无监督域自适应。设计三种有效的对齐机制包括结构感知对齐、域对齐和类中心对齐，能够有效的学习域不变和语义表示，减少域自适应的离散性。
Data Structure: 数据结构通常反映数据集的固有属性，包括边缘或条件数据分布、目标统计信息、几何数据结构
Domain Label：域标签用于对抗域适应方法，可以帮助训练一个域分类器来对源域和目标域的全局分布建模
Class label: 类标签，特别是目标域伪标签，通常被用来强制语义对齐，这可以保证来自不同域的具有相同类标签的样本能够被映射到特征空间附近。
**main contributions**
+ GCAN在无监督域自适应算法中的作用
+ 三种信息的联合建模
+ 三种有效的对齐机制
**图卷积对抗网络**
+ 整体目标函数
$$\begin{aligned} \mathcal{L}\left(\mathcal{X}_{S}, \mathcal{Y}_{S}, \mathcal{X}_{T}\right) &=\mathcal{L}_{C}\left(\mathcal{X}_{S}, \mathcal{Y}_{S}\right)+\lambda \mathcal{L}_{D A}\left(\mathcal{X}_{S}, \mathcal{X}_{T}\right) \\ &+\gamma \mathcal{L}_{C A}\left(\mathcal{X}_{S}, \mathcal{Y}_{S}, \mathcal{X}_{T}\right)+\eta \mathcal{L}_{T} \end{aligned}$$
+ 域对齐
使用域对抗相似损失作为域对齐损失
$$\begin{aligned} \mathcal{L}_{D A}\left(\mathcal{X}_{S}, \mathcal{X}_{T}\right) &=\mathbb{E}_{x \in D_{S}}[\log (1-D(G(x)))] \\ &+\mathbb{E}_{x \in D_{T}}[\log (D(G(x)))] \end{aligned}$$
域分类器$\mathcal{D}$的主要作用是判别来自feature extractor G的特征是来自源域还是目标域，训练$\mathcal{G}$来欺骗$\mathcal{D}$
+ 结构感知对齐
第一步，用结构分析网络获取mini-batch样本的结构分数，第二部通过CNN提取特征作为GCN网络的输入
GCN的构建：CNN提取的特征作为节点特征，链接矩阵A由结构分数$X_{sc}$获得
$$\hat{\mathbf{A}}=\mathbf{X}_{s c} \mathbf{X}_{s c}^{T}$$
结构分数$X_{sc}$通过triplet loss由源域获得.
$$\mathcal{L}_{T}=\max \left(\left\|\mathbf{X}_{s c_{a}}-\mathbf{X}_{s c_{p}}\right\|^{2}-\left\|\mathbf{X}_{s c_{a}}-\mathbf{X}_{s c_{n}}\right\|^{2}+\alpha_{T}, 0\right)$$
这里的一个细节是源域和目标域被用来训练同一个GCN网络但是需要分开进行。
+ 类中心对齐
作者指出，特征具有领域不变性(Domain Invariance)与结构一致性(Structure Consistency)并不意味着具有判别性(Discriminability)。于是，作者利用构建源域/目标域特征的聚类中心保证所学到的特征的判别能力。
第一步，通过目标分类器$F$获得目标域的伪标签，第二步，通过有标签样本和伪标签样本计算类中心或形心(centriod)。中心对齐目标函数：
$$\mathcal{L}_{C A}\left(\mathcal{X}_{S}, \mathcal{Y}_{S}, \mathcal{X}_{T}, \mathcal{Y}_{T}\right)=\sum_{k=1}^{K} \phi\left(\mathcal{C}_{S}^{k}, \mathcal{C}_{T}^{k}\right)$$
#### 7. Semi-supervised classification with graph convolutional networks
**摘要**
&#8196;&#8196;通过谱图卷积的局部一阶近似，来确定卷积神经网络的结构，该模型在图的边数上线性缩放，该模型学习隐藏层表示，这些表示即编码局部图结构，也编码节点特征，通过图结构数据中部分有标签的节点数据对卷积神经网络模型训练，使网络模型对其余无标签的数据进行进一步的分类。
**半监督节点分类**



