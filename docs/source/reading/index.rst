Reading Notes
===================

Paper List
-------------------

- :ref:`Efficient Estimation of Word Representations in Vector Space`

- :ref:`On Availability for Blockchain-Based Systems`

- :ref:`Personal Recommendation Using Deep Recurrent Neural Networks in NetEase`


.. _Efficient Estimation of Word Representations in Vector Space:

Efficient Estimation of Word Representations in Vector Space
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

作者提出了两种新的结构模型,用于在大的数据集中,用向量来表示words.

- 作者的其中一个motivation是想设计一个可以有效的训练更多的数据,但在表示能力上可能没神经网络这么好(`precisely`)

- **CBOW** : 跟标准的词袋(bag-of-words)模型不同的是,该模型用连续的分布来代表上下文

    - 计算复杂度为 Q = N × D + D × log_2(V)

- **Skip-gram** : `It tries to maximize classification of a word based on another word in the sam sentence`

    - 计算复杂度为 Q = C × (D + D × log_2(V))


.. _On Availability for Blockchain-Based Systems:

On Availability for Blockchain-Based Systems
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


.. _Personal Recommendation Using Deep Recurrent Neural Networks in NetEase:

Personal Recommendation Using Deep Recurrent Neural Networks in NetEase
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

- 即使协同过滤(CF)在推荐相关的工作表现得比较好,但是这是建立于历史数据之上,缺乏用户的选择.因此作者提出了用RNN来做推荐的模型.

- challenge:
    - 输入向量大
    - 模型需要对用户实时访问/顺序足够敏感和有效
    - 模型需要在线学习,速度也要足够快

- 作者提出的模型与standard的DRNN不同的是:
    - 模型是用来跟踪(`track`)用户的访问模型
    - 页面序列显示了用户到他所需产品的路径
    - 模型的目的是要缩短用户到其所需产品的距离并要求实时推荐(速度足够快)