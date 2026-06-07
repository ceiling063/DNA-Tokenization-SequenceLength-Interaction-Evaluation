# Goodlab export

**参考**

1\.LaTex网页版 Overleaf，GitHub,CSDN开发者社区

2\.Overleaf以及GitHub,Python数据分析\+可视化内容；

3\.生物信息相关内容\(CSDN\)

https://blog\.csdn\.net/2302\_80012625/article/details/142894695  生信基础知识\-RefSeq

https://blog\.csdn\.net/2301\_79168784/article/details/148934894   Overleaf;

https://cn\.overleaf\.com/project/6a1c5a3e8ac7f5fc4321445a github

[https://www\.ncbi\.nlm\.nih\.gov](https://www.ncbi.nlm.nih.gov/) NCBI官网链接

4\.（夸克\\CSDN）

词表特性差异——基本单位、语言覆盖、分词算法、规模大小、特殊标记

交互效应——各因素不同水平组合对因变量产生的联合影响

https://zhuanlan\.zhihu\.com/p/656630180

长距离依赖——句法结构中存在较多词汇间隔的依存关系

k\-mer——长度为k的核苷酸序列（基本分析单元）

神经网络架构：CNN、RNN、Transformer、GAN、Autoencoder    https://juejin\.cn/post/7513183180091342883 

统计Baseline:比较基准的参照标准（机器学习模型快速验证数据可行性）https://ask\.csdn\.net/questions/8442129

https://blog\.csdn\.net/weixin\_46163097/article/details/123933649?ops\_request\_misc=\&request\_id=\&biz\_id=102\&utm\_term=%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0baseline%E6%A8%A1%E5%9E%8B\&utm\_medium=distribute\.pc\_search\_result\.none\-task\-blog\-2\~all\~sobaiduweb\~default\-0\-123933649\.142^v102^pc\_search\_result\_base8\&spm=1018\.2226\.3001\.4187

TF\-IDF

https://blog\.csdn\.net/asialee\_bird/article/details/81486700

taxonomy 路径预测、Pipeline = 一连串顺序执行的处理步骤，上一步的输出是下一步的输入（改变输入方式，单个变量）

https://blog\.csdn\.net/2401\_88644935/article/details/159603956?spm=1001\.2101\.3001\.6650\.2\&utm\_medium=distribute\.pc\_relevant\.none\-task\-blog\-2%7Edefault%7EYuanLiJiHua%7ECtr\-2\-159603956\-blog\-123933649\.235%5Ev43%5Epc\_blog\_bottom\_relevance\_base2\&depth\_1\-utm\_source=distribute\.pc\_relevant\.none\-task\-blog\-2%7Edefault%7EYuanLiJiHua%7ECtr\-2\-159603956\-blog\-123933649\.235%5Ev43%5Epc\_blog\_bottom\_relevance\_base2\&utm\_relevant\_index=5

复杂度：https://blog\.csdn\.net/user11223344abc/article/details/81485842?ops\_request\_misc=elastic\_search\_misc\&request\_id=75a306933a318adefd83018fe423b9c7\&biz\_id=0\&utm\_medium=distribute\.pc\_search\_result\.none\-task\-blog\-2\~all\~top\_positive\~default\-1\-81485842\-null\-null\.142^v102^pc\_search\_result\_base8\&utm\_term=%E5%A4%8D%E6%9D%82%E5%BA%A6\&spm=1018\.2226\.3001\.4187

出现git bash \&git hub账号没有绑定成功： ssh \-T \-p 443@ssh\.github\.com

SSH克隆问题尚未解决，最后通过https克隆 创建测试文件克隆测试成功

（配置过程报错解决）



Overleaf文章结构参考学术英语阅读与写作所学内容及相关论文

**（Coupling LSTM and CNN Neural Networks for Accurate Carbon Emission Prediction in 30 Chinese Provinces）**

https://blog\.csdn\.net/young951023/article/details/79601664?ops\_request\_misc=elastic\_search\_misc\&request\_id=2e1a141756762d2984a638eb853c5b55\&biz\_id=0\&utm\_medium=distribute\.pc\_search\_result\.none\-task\-blog\-2\~all\~top\_positive\~default\-1\-79601664\-null\-null\.142^v102^pc\_search\_result\_base8\&utm\_term=LaTex%E5%91%BD%E4%BB%A4\&spm=1018\.2226\.3001\.4187

https://blog\.csdn\.net/QFire/article/details/81382048?ops\_request\_misc=\&request\_id=\&biz\_id=102\&utm\_term=LaTex%E5%91%BD%E4%BB%A4\&utm\_medium=distribute\.pc\_search\_result\.none\-task\-blog\-2\~all\~sobaiduweb\~default\-6\-81382048\.142^v102^pc\_search\_result\_base8\&spm=1018\.2226\.3001\.4187

【一个非常快速的 Latex 入门教程】 https://www\.bilibili\.com/video/BV11h41127FD/?share\_source=copy\_web\&vd\_source=eadb0770eed8a121456f2f56a568c539

**后续资料：**

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OTIwZDczNjdiODk0OTYwZGNlNzMyMjI2NjllMWMyNGRfZTNjZDIzZmQyNmEyY2RkZmRkZjVjZjhiYjg0YTQ1ZDlfSUQ6NzY0NjMyMTc0OTMxMTI5NDY3NF8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

1：前期网站及软件准备\+数据获取\+环境搭建\+关联github

2：实现三种tokenization策略（重叠k\-mer\\canonical k\-mer\\BPE\)\-\-\-\-模型输入关键步骤

3：实现统计Baseline\+简单神经网络模型

4：核心实验（策略\+长度、K值敏感性、长度鲁棒性）

5：整理结果、数据分析、写LaTex报告初稿

6：完善报告\+最终检查

任务：⾃主获取数据，构建⼀个 DNA 序列物种分类 pipeline：系统对⽐不同 tokenization 策略在 

不同序列⻓度下的分类性能、计算效率与词表特性差异。

**一、获取数据、划分、初步分析**

要求：

⾃⾏从 NCBI RefSeq、Ensembl 或 GTDB 等公开数据库获取包含多个物种的基因组序 

列，要求分属不同的⻔或纲。数据须包含短 reads（150\-500bp）与⻓ contigs（≥3000bp）两个⻓ 

度组，⾃⾏完成 train/validation/test 划分，并在报告中描述数据的基本统计特性（序列数量、⻓度 

分布等）。

完成Pycharm虚拟环境的配置；

AI使用：Pycharm相关配置以及目录创建，python相关代码书写；数据准备，从公开数据库下载多个物种的基因组序列，数据包含reads\\contigs两个长度组；

小数据看程序是否出错，调试改错：

自动下载程序出错：无法从官网中下载到数据\-\>导入已下载好的数据

数据选取：

为了保证实验具有准确性；数据应选取公开数据库中分属不同门或纲的DNA数据；

选取的是官网筛选过的推荐基因组，保证数据质量，数据存在较大差异，比较经典

1. Escherichia coli \(大肠杆菌\)：`GCF_000005845.2`

https://ftp\.ncbi\.nlm\.nih\.gov/genomes/all/GCF/000/005/845/GCF\_000005845\.2\_ASM584v2/GCF\_000005845\.2\_ASM584v2\_genomic\.fna\.gz

2. Bacillus subtilis \(枯草芽孢杆菌\)：`GCF_000009045.1` 

https://ftp\.ncbi\.nlm\.nih\.gov/genomes/all/GCF/000/009/045/GCF\_000009045\.1\_ASM904v1/GCF\_000009045\.1\_ASM904v1\_genomic\.fna\.gz

数据成功获取后利用代码处理和分析数据，画出数据统计图表（AI给的代码）

（AI辅助README文档初步书写）

**二、实现三种tokenization策略**

1\.重叠k\-mer：

https://zhuanlan\.zhihu\.com/p/659230386  步长为1，每次取三个碱基对；编号——python代码有ai辅助

OOV = Out Of Vocabulary = 不在词表中的生词；频数变量：统计编号出现次数

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YzhiNGIwY2UyZDRkNWZjZDg5ZTA4YThhZDg1Njc1ZTRfMmU3ZTRkNmRlZDlmM2NkYTgwNTA1NDU1YzgzNTRiNzBfSUQ6NzY0NjcxMzkwNTYwNzg4NzgyMF8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

第一次运行结果：初步实现k\-mer策略；

2\.Canonical k\-mer（重叠 k\-mer \+ 反向互补归⼀化）

代码实现过程中，无法导入第一个策略：Python 的 `import` 语句要求模块名是有效的变量名（只能包含字母、数字、下划线）。`overlap k-mer.py` 有空格和连字符，所以会报错。

\>\>将第一个文件名改成kmer，第二个文件名改成canonical

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NDAxMDcxOGQ0ZDc4YzU3YTc2OGNkMWNiYzNkMTRjY2RfNWY5YWQyMzRmMzEzN2YxZmNlNThlZDgyNzNkZTFkNGFfSUQ6NzY0NjcxNzI2Njg0MzAyODQyOV8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

第一次运行结果：初步实现 canonical k\-mer策略。

3\.BPE（基于 DNA 序列训练词表）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=Y2EzNzA0MDMxNmE0NDRiMjBiMjZjM2Q0ODMzZDJkNjVfODQzNTU1NzJmYmRhYmUxN2ZkZGU2YzQ3OTIzZDM4MmRfSUQ6NzY0NjcyNDU1MjI0OTc4OTYxOF8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

第一次运行结果：初步实现BPE（基于 DNA 序列训练词表）策略；

**三、对比任务（三种策略）**

词表特性差异：词表大小、OOV 率、平均 token 数

生成三种策略的对比脚本，对比三个策略的词表大小、OOV率、平均token数、tokenization耗时、词表稀疏度

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NTViMDFmZDhlMzBjNDdiZmU3YzIwMTE4NGUzYWNlMzdfNWFlNDMyM2Y1OWY4MjFjMzBhMzYyYzVjZGJiZjA2ZmVfSUQ6NzY0NzA1NDA2ODQ2NDc4MjU0NV8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

对比结果如上，以及生成了三种策略在（oov率、词表稀疏度、词表大小、耗时）对比图

对比实验控制变量

**四、统计Baseline（此时的Baseline只用了短片段）**

简单分类器采用逻辑回归。（如 k\-mer 频率 \+ TF\-IDF \+ 简单分类器）作为参照

Baseline运行时报错：重新编辑Baseline文件运行工作目录

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YTY5NzE1MWRhZjVhODM2NWIzYzk3NzcyOTdlMzBhNDhfM2Y2OTZmY2M5ZDZhMWYxNTkyNjcxNjZiYTdlMWVlZWVfSUQ6NzY0Njc2OTE4NDc1OTc5NDY0MV8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

系统对⽐不同 tokenization 策略在 不同序列⻓度下的分类性能、计算效率与词表特性差异。

**Baseline长片段：**

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MDU1MjgzY2U0NDM5ODBkMzdmNmI5MDllMjgyNzgyNTlfYjQxNTQwMDQxMzVmNzhlMzM1MWZmZWU4YTFiYWVlZWVfSUQ6NzY0Njk3MTY4NjQ4NzYyNDk0Ml8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)



**五、构建神经网络模型**

实验应采取严格的控制变量原则，同⼀模型在不同 tokenization 实验下须保持结构⼀致（层数、维 

度、参数量固定），仅改变输⼊ tokenization ⽅式。

1\.选择神经网络架构

- 神经网络 是一个具体的模型类型（比如 CNN、LSTM）

- 模型架构 是神经网络的具体结构（层数、参数等）

复杂度分析就是：当输入序列变长时，模型的计算量怎么增长？

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MDBlN2NjZTFlNjYyOWVlN2M5YzRjNzJjNmI5NDFmZWJfNmQ3MDQwNmUwZTNiN2ZmYjEwYzZmM2VmZjVlYzFhM2FfSUQ6NzY0NzAzOTA4NzMzNzkzNDAwNV8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

根据文档，CNN、RNN/LSTM、Transformer可以初步选择

根据输出数据进行复杂度计算

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=N2YwZjg5MzdhMDIxOTZjNTZlYmViNjE5YTZlM2Q5ZjRfMmIyNDYzODY3MmYzY2E3MDE3MzA2OGZiNDk4ZjNhN2JfSUQ6NzY0NzA0MzAwODA4OTUyNTQ4M18xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

卷积核大小选择最大值7,向量维度固定64

根据表格计算结果：CNN——计算量增长：1342208/143673\.6=9\.342、LSTM\(RNN）：12271616/1313587\.2=9\.342；Tranformer:574465024/6582303\.36=87\.274：相比起前两种神经网络架构，Transformer增长量太大，计算量太大；排除原因1

排除原因2：数据量过小：Yang, E\., Li, M\. D\., Raghavan, S\., Deng, F\., Lang, M\., Succi, M\. D\., Huang, A\. J\., \& Kalpathy\-Cramer, J\. \(2023\)\. Transformer versus traditional natural language processing: how much data is enough for automated radiology report classification? *The British Journal of Radiology*, 96\(1149\)\.

排除原因3：硬件不允许：Martin\-Salinas, I\., Badia, J\. M\., Valls, O\., Leon, G\., del Amor, R\., Belloch, J\. A\., Amor\-Martin, A\., \& Naranjo, V\. \(2024\)\. Evaluating and accelerating vision transformers on GPU\-based embedded edge AI systems\. *The Journal of Supercomputing*, 80\(1\)\. \(Springer出版\)

比较CNN、LSTM：

分别写个脚本带数据进去看两者效率；安装torch软件包；将两种模型在三种策略下进行对比

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NDdlNGQzNzM1OTRiN2Y1ZmZiNTllZGE1ZGE0YjZmYmFfODgzNjcwMzE5OGE4ZjcxNjllZjExZTAwOWFiZjhkMDNfSUQ6NzY0NzQ2NjY2ODc2Nzc0Mjk0MF8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MmM5MTRjMjUyZGQ5OWY2MDBhOTkzODlhYzAxMzM1NTNfYzIxMTQyYjZhMDhhZjk0MjhiNmE2ZTJlZGFkMjIxNTRfSUQ6NzY0NzQ2NjkwODE0OTAwOTY1MV8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

根据两个测试结果；CNN与LSTM相比，在相同策略相同序列长度2，整体上CNN的分类准确性更高，训练时间更短，推断时间更短；

而且在CNN与各个策略结合的对比中BPE\+CNN的准确率明显高于其他组合，但是训练时间及推断时间也相应较长，需要优化。

\-\>优化当前CNN模型（后续做）：减少推断时间以及训练时间，选取CNN\+short组合短时间进行参数调优；目前先保留原有参数进行后续实验（准确率已经相对较高93%，可以支撑分类）。

具体神经网络内容（Latex具体结构解释表格）

**六、核心实验前三**

1\.实验一：切分策略** × **⻓度交互 固定 k=5（BPE 固定词表⼤⼩），对⽐ 3 种策略在 short reads 与 long 

contigs 上的准确率、F1、效率

实验一用Baseline来做：固定K=5，改变输入策略和序列长度；记录准确率、F1、效率等结果。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MjY1MDBhMjRkOWFjN2VhNzcwMmZhM2NhMjE1ZGJkYmJfZjMzNmRlOWFiNWU3M2ExMDNjYjViZjE2NzI0MDY3YTBfSUQ6NzY0NzUzNzk5Nzk2MTE3MDEwOF8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

结果分析：当序列长度相同时，前两种的准确率高于BPE，F1的分数也高于BPE，但相比之下，BPE的训练及推断时间较短，重叠K\-mer训练时间最长

当输入策略相同时，长片段的准确性大于短片段，F1的分数也高于短片段训练时间及推断时间也相对较短。

？：

- 为什么 k\-mer 在这个任务上比 BPE 好这么多？

- 已经有了一个表现较好的 Baseline \(94\.7%\)，接下来 CNN 模型能否超越它，或者在哪些方面能做得更好。

2\.实验二：**k **值**/BPE **词表敏感性 固定重叠 k\-mer，对⽐ k=3,4,5,6；或固定 BPE，对⽐不同词表⼤⼩

固定重叠k\-mer，对比k=3,4,5,6判断相应结果

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OGI1NjU2ZWVhYWQwMzRiM2MxZTliNjViMTBlZTNkNWRfYmM3YTY1MjhjNDBiMWNiMzhjMWY2OTNhMTRiZjcwYjFfSUQ6NzY0NzU0MTk4ODUwOTYxNzEzOF8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

结果分析：固定重叠k\-mer，仅改变k值；对于相同序列长度：一定情况下：k值越大,准确性越高，F1得分越高，vocab\_size越大；

对于相同k值：一定情况下：长片段的准确性高于短片段，F1分数高于短片段，vocab\_size相差不大，但是当k=3时，短片段的准确率高于长片段，

3\.实验三：⻓度鲁棒性 同⼀策略在 short reads vs\. long contigs 上的性能差异与词表稀疏度变化

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=Yzk4NGI5ZjY3Y2JiM2Q2MWM2ZTdmNmRjMzE3MDg3MDdfYWIwNzA1NmUxNzA0Y2FiOWE2MzdiNDRlMTIxZDIwY2RfSUQ6NzY0Nzg5MjM2OTgwMDEzNzk2NF8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

根据三对比任务所得结果

（问题：gini系数不能为负：可能因为物种样本数量太少，OOV率无法很好判断词表稀疏度；更换判断方法\-AI）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZjY4ODQ5Yzg5OWRmNDhhYTRhMzQzOWY5NzkwZjFlOTJfNzVhNjI2ZGFmNzQ0MGM2NGM5MGNjMmMxYjk3OGM0MTRfSUQ6NzY0Nzg5NTU3ODI0Mzc5NTk0Nl8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

长度鲁棒性需要加上实验一准确率结果

1\.重叠k\-mer在词表大小及OOV率上没有差别，长片段相比短片段 平均token数更大、耗时更长

2\.Canonical k\-mer在词表大小及OOV率上也没有差别，长片段相比短片段 平均token数更大，耗时更长

3\.BPE在词表大小及OOV率上也没有差别，长片段相比短片段 平均token数更大，耗时更长，增长幅度更大

\-\>这里可以看出三种策略在不同长度上存在交互效应，长片段和短片段的表现趋势类似，只是增长幅度不同。

但稀疏度还是不好区分亟待解决

**七、最后一个对比实验（****Baseline 参照 同⼀ tokenization（BPE） 下，统计 Baseline 与神经⽹络的性能差距****）**

分类性能（模型）：准确率、F1（AI）

Baseline以及CNN的准确率在之前的实验结果及训练结果已经可以得出 Baseline（BPE\+short\):78\.082% 

Baseline\(BPE\+long\):78\.947%       CNN\(BPE\+short\):93\.151%    CNN\(BPE\+long\):84\.211%

计算效率：tokenization 耗时、推理时间（模型）、峰值内存（模型）

峰值内存（结果如下）Baseline:0\.07mb  CNN:0\.02mb CNN的峰值内存小于Baseline,\(相同情况下，选择峰值内存更低的）

推理时间：Baseline\(BPE\+short\):0    Baseline\(BPE\+long\)：0   CNN\(BPE\+short\):680\.305ms  CNN\(BPE\+long\):274\.621ms\(CNN推理时间远长于Baseline\)

tokenization耗时：Baseline与CNN耗时在同一序列长度下相同

F1分数：在短片段上CNN高于Baseline，在长片段上CNN低于Baseline\.

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjdkNTdjNjUzNzhhZTBjODE0MjUxZTAwYjExODE5OTFfMjAyZWM4ZWRmYmIwYWY2MmE0YTI2YjU2NDY2Yjg2MWFfSUQ6NzY0NzkwOTAwOTM2MTUyMTg2Ml8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzE4YzA3NTFlMzZmYjdhYTkxYmM5MjUzOGI4ZjQ3MTFfNTJmMWVlOTUyMjQxZjMzYmRkNDU4M2E0M2ZkNzk0NTRfSUQ6NzY0Nzg5ODUwNTAwNDA5MjQwMV8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

**八、结果分析与结论（实验结果在results文件夹）**

交互效应是指：两个因素共同作用时，产生的效果不是简单的相加，而是相互影响。

根据核心实验与策略对比：不同tokenization策略与不同序列长度存在交互效应

不同策略，不同长度对分类的性能有一定影响；且不同策略对分类性能的影响，不会因为长度改变而影响趋势；

但是不同策略之间在不同长度序列上的表现对比会更加明显

**九、撰写实验报告以及README文档**

**十、尝试进行实现层级分类（从⻔到种的 taxonomy 路径预测），并设计相应的层级评估指标。**

但是当前数据只有两个菌种分属于不同的门，层级分类没有特别的设计；需要增加更多的数据进行从门到种的taxonomy路径预测；每门需要两到三个物种以便对比；先大概有个框架后续进行实验处理

下载

铜绿假单胞菌 \(Pseudomonas aeruginosa\)RefSeq ID: GCF\_000006765\.1 \(PAO1 菌株\)

金黄色葡萄球菌 \(Staphylococcus aureus\)RefSeq ID: GCF\_000013425\.1 \(NCTC 8325 菌株\)

重新运行数据准备脚本，生成新的四个数据集（两个门，每个门有两个物种）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YmU3MDhiZjBiYWIwZjVlMmYwMzBhNWM2MGI4NDQwMWVfNTM2YjE1MDdkMmQ3OWJhMGNjNjY5M2E1NWZmNjFjN2RfSUQ6NzY0ODU1MDQ5MTQxMjI3MDA1Ml8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

对应的层级评估指标：门预测正确率，种预测正确率、门和种都预测正确率、训练时间、推理时间。（分别为短片段、长片段）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NTI5ZjE4Y2VkZjhmYTgwMjMxOTA1MGU1Mzc3ZTJiMjFfOTA0OWY4OTEwZjlmYzMwNjI2MDAxZWQ4OTI2MGYxMzJfSUQ6NzY0ODU2MzEzMTk3MDU1NDg0OV8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTJjNGI0NDcwZjAyYzBhY2EzMGE5YWYwOTE0NWFhOGFfMDZmOGZhMmZlYjdmOTFmM2I0OTFkNjkxMmJhZDc0ODdfSUQ6NzY0ODU2MzI0NjAwNTY1MjQyN18xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

根据训练结果：长片段准确率极高，短片段稍低，可能是因为短片段信息量少，长片段推理时间更长

增加数据集后重新运行实验三判断词表稀疏度（看看是否有变化\)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDcxZDZlMzMzMGRiM2FhNTczYTgxNWRkZjg2MDNiNWFfM2YxNDFiNzc4MTk3ZjU1MTkyZjMxODA2MGY2YTY5ODBfSUQ6NzY0ODU2NjI2MTY4MTQ0MTk3Ml8xNzgwODM0ODUzOjE3ODA5MjEyNTNfVjM)

还是没有很大的变化，后续再解决相应问题

