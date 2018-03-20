Reading Notes
===================

Paper List
-------------------

- :ref:`Efficient Estimation of Word Representations in Vector Space` (NLP)

- :ref:`On Availability for Blockchain-Based Systems` (区块链)

- :ref:`Personal Recommendation Using Deep Recurrent Neural Networks in NetEase` (推荐系统)


.. _Efficient Estimation of Word Representations in Vector Space:

Efficient Estimation of Word Representations in Vector Space
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

作者提出了两种新的结构模型,用于在大的数据集中,用向量来表示words.

- 作者的其中一个motivation是想设计一个可以有效的训练更多的数据,但在表示能力上可能没神经网络这么好(`precisely`)

- **CBOW** : 跟标准的词袋(bag-of-words)模型不同的是,该模型用连续的分布来代表上下文

    - 计算复杂度为 :math:`Q = N × D + D × log_2(V)`

- **Skip-gram** : `It tries to maximize classification of a word based on another word in the sam sentence`

    - 计算复杂度为 :math:`Q = C × (D + D × log_2(V))`


.. _On Availability for Blockchain-Based Systems:

On Availability for Blockchain-Based Systems
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

- 作者详细分析了在区块链中对commit times(交易确认时间)产生消极影响的原因, 并提出了一个明确的中断机制来减少/限制这种延迟的发生.

- 比特币中需要6个区块才能 **最终确认** 交易, 以太坊则需要12个区块(这个数字依赖于事物/交易的价值、挖矿的开销和攻击的威胁性), 这意味着攻击者难以控制足够的算力来破坏/改变当前的共识(`51%攻击`). 作者也提到一个使用少于51%的算力来攻击的工作

作者首先分析了transaction fees 和 locktimes, 得出了这两者对交易延迟的影响不大的结论


.. _Personal Recommendation Using Deep Recurrent Neural Networks in NetEase:

Personal Recommendation Using Deep Recurrent Neural Networks in NetEase
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

- 作者提出了 **DRNN** (Deep Recurrent Neural Networks)和 **FNN** (Feedforward Neural Network)模型来对用户网购的行为进行预测和实时推荐.

    - 在DRNN中, 因为用户访问的可能有多个网页, 因此把之前的/超出范围(n)的浏览记录合并成一个history state, 同时加上当前的一些浏览state作为输入
    - FNN的作用是根据用户的购买记录对用户的购买进行预测

    两个模型合并输出最终的预测. 与传统的方法(CF, 协同过滤)相比, 这篇文章提出的方法能达到实时推荐的效果, 而且表现也更加好.

- history state的表示为:

.. math::
    \bar{V} = \sum_{i=0}^{x-n}\epsilon_{i}V_{i},\ 其中,\ \epsilon 为\ \epsilon_{i}=\frac{\theta(p_i)}{\sum_{j=i}^{x-n}\theta(p_j)}

最终, 得到用户购买第i个商品的概率为:

.. math::
    \frac{f(\sum_{x=0}^{E-1}(w_{i}^{L_0}aL_{0}(t)+bL_{0}(t))+\sum_{x=0}^{\bar{E}-1}(\bar{w}_{i}^{L_1}\bar{a}_{x}^{(L_1)}+b_{x}^{(L_1)}))}
    {\sum_{x}f(\sum_{x=0}^{E-1}(w_{i}^{L_0}aL_{0}(t)+bL_{0}(t))+\sum_{x=0}^{\bar{E}-1}(\bar{w}_{i}^{L_1}\bar{a}_{x}^{(L_1)}+b_{x}^{(L_1)}))}

- 即使协同过滤(CF)在推荐相关的工作表现得比较好,但是这是建立于历史数据之上,缺乏用户的选择.因此作者提出了用RNN来做推荐的模型.

- challenge:

    - 输入向量大
    - 模型需要对用户实时访问/顺序足够敏感和有效
    - 模型需要在线学习,速度也要足够快

- 作者提出的模型与standard的DRNN不同的是:

    - 模型是用来跟踪(`track`)用户的访问模型
    - 页面序列显示了用户到他所需产品的路径
    - 模型的目的是要缩短用户到其所需产品的距离并要求实时推荐(速度足够快)