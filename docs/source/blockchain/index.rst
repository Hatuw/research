BlockChain Research
===================

Ethereum
--------

初始化
>>>>>>

- 创世块

	初始化创世块： 创世块信息放在 ``genesis.json`` 里面，使用 ``geth --datadir <path/to/data> init <path/of/genesis.json>`` 来初始化区块

.. code-block:: json
	:linenos:

	{ 
		"config": {
			"chainId": 0,
			"homesteadBlock": 0,
			"eip155Block": 0,
			"eip158Block": 0
		},
		"alloc"      : {},
		"coinbase"   : "0x0000000000000000000000000000000000000000",
		"difficulty" : "0x20000",
		"extraData"  : "",
		"gasLimit"   : "0x2fefd8",
		"nonce"      : "0x0000000000000042",
		"mixhash"    : "0x0000000000000000000000000000000000000000000000000000000000000000",
		"parentHash" : "0x0000000000000000000000000000000000000000000000000000000000000000",
		"timestamp"  : "0x00"
	}

- 创建账户

	创建10个账户，使用ansible可以快速列出地址(``ansible eth -m shell -a "geth account list" | grep {.*} -o``)：

.. code-block:: text
	:linenos:

	d04c2aa99a4830570386d73cbfdcb514c26c755b
	553bc297f8ef414623800960323cd9e4c9675ea7
	fd40991efcf56131d9b6772d7ee0c138e9416d93
	01314570bc530c0a260a792940cdf6c45e444a1b
	778cd15f07f4191305caf62b0548547b9729ae74
	3f27320c7952df6c68719d0049808541750f925f
	1468b1bf9a6868e6890783aedcd6f87496aa89f3
	936c87067e6e3a614499ffc54c7b65324ad7c869
	990fcbc3767bd6dd80bc8076101ea21d73136678
	fcf485e29c8364f82cea554021e3fa2771d7466d

启动
>>>>

已知其中一个节点(``ht@172.18.196.2``)的bootnode地址，使用ansible部署：

.. code-block:: bash

	ansible eth -m shell -a "nohup geth --rpc --rpccorsdomain \'*\' --bootnodes \$(curl https://raw.githubusercontent.com/Hatuw/deployBC/master/ethereum/bootnode) >> geth.log" -T 1 -f 10


如果需要挖矿的话可以在后面加上 ``--mine``

issues
>>>>>>

- When using ansible and geth to deploy the Ethereum. It can not run in background.

- The node will connect to the node on ip: `172.18.196.2` first, but it will not connect others nodes after initial. That is to say, the network will turn down if the node in `172.18.196.2` been shutdowned down.

- The Hyperbench project is used to test the performance in a private chain. It test on geth(Ethereum, Party) and Hyperledger Fabric.


Hyperledger
-----------

.. image:: ../assets/tx_hyperledger.png

*Figure 1. hyperledger transaction flow*

MISC
-----------

论文1
>>>>>>>

论文标题：Thunderella：区块链理想条件下的瞬间响应（Thunderella: Blockchains with Optimistic Instant Confirmation）

作者：Rafael Pass，康奈尔大学计算机科学系副教授

内容简介：这篇论文中，作者介绍了一个全新的算法叫做「Thunderella」。与一般状态机的共识原理不同（状态机相当于一个共识机制的抽象，对分布式网络中大量节点的请求进行确认），Thunderella使得状态机可以在实现快速异步处理的同时，在异常时还可以启动回滚机制。如此一来，状态机的相应速度与同步协议无异，在不出现「拜占庭将军问题」（及大多数人都是诚实的）的情况下，可以做到对交易的瞬间响应。

这篇论文中，作者对POW协议下，无需许可和需要许可的设定，提供了一些示例，相应速度可以达到正常上网的体验。不过正如上面所说，这一算法的前提条件是网络中的大部分节点或算力是诚实的，而这里所说的「大部分」指的是不能低于3/4。

原文链接： https://eprint.iacr.org/2017/913.pdf


论文2
>>>>>>>

论文标题：比特币为什么靠得住？关于比特币的理性协议的设计（But Why Does it Work？A Rational Protocol Design Treatment of Bitcoin?）

作者：Vassilis Zikas，英国爱丁堡大学区块链技术实验室副主任、副教授。

内容简介：这篇论文是关于比特币的，作者试图通过实验来验证RPD（Rational Protocol Design）框架作为一个「理性的密码学框架」在比特币分析中的可用性。众所周知，比特币交易的前提是默认大部分算力（矿工）是诚实的，然而在现实世界中，如何向公众证明大多数人是诚实的却是个难题。

在这篇文章中，作者通过对RPD框架部署了一套新的机制，结果显示无论是对矿工添加新的区块进行奖励，还是让他们对挖矿付出代价，诚实的节点总是占大多数。这一结果跟币圈一直以来「矿工总是追求利益最大化」的论调相去甚远。

这篇论文的价值在于，由于比特币交易流程的特殊性，以往密码学家没有一个关于合约是如何工作的常用的模型，需要各自开发安全等级上非常过硬的共识算法。这篇文章却改变了这一现状。

原文链接： https://eprint.iacr.org/2018/138.pdf


论文3
>>>>>>>

论文标题：Ouroboros Praos：一条自适应安全和半同步POS的区块链（Ouroboros Praos: An adaptively-secure, semi-synchronous proof-of-stake blockchain）

作者：Peter Gaži，IOHK研究员

内容简介：这篇文章中，作者提出了一个新的POS协议——Ouroboros Praos，这个协议可以看作是，针对IOHK之前发布的公链Cardano的升级算法。

这一算法第一次实现了半同步条件下对完全自适应腐败（fully-adaptive corruption）的安全保护。尤其是比特币持有者以诚实者占大多数时，黑客可以随时随地对这些人进行「腐化」，这一协议保证了更安全的数字签名，以及对随机函数进行验证的新类别——在而已密钥的生成下，依然保持不可预测。

不过，这个协议仅在随机预言机模型的标准加密假设中有效。

原文链接： https://eprint.iacr.org/2017/573.pdf


论文4
>>>>>>>

论文标题：可持续空间模型（Sustained Space Complexity）

作者：Jo¨el Alwen，奥地利科学与技术学院研究员

内容简介：MHF（Memory-hard functions）是一种函数，它的评测成本受存储成本所控制。在硬件设备（如FPGAs、ASICs）上对MHF进行评测，成本不比在x86设备上便宜。

在这篇文章中，作者引入了一个叫做「持续内存机制」（Sustainedmemory Complexity）。这一机制在平行随机预言机中构建。通过n和O两个函数进行运算，其中n代表步骤，O代表存储，函数式为：O(n/ log(n))。在每个步骤中，一条询问被放入随机预言机，其他算法也可以将其他随机询问放入随机预言机。其存储为：Ω(n/ log(n)) ，步骤为：Ω(n)。

原文链接： https://eprint.iacr.org/2018/147.pdf


EUROCRYPT 2018 最佳论文
>>>>>>>>>>>>>>>>>>>>>>>>>>>>

在区块链论坛入选的4篇论文之外，还有一篇关于区块链的论文入选了「最佳论文」，就是下面这篇。

论文标题：简单的连续工作证明（Simple Proofs of Sequential Work）

作者：Krzysztof Pietrzak，密码学家、奥地利科学与技术学院研究员。Bram Cohen，BitTorrent创始人、Chia CEO

内容简介：试图通过「空间证明」（Proof of Space）来保证比特币及其他加密货币的安全。「空间证明」是Bram Cohen之前提出的一种取代PoW的工作证明方式。

原文链接： https://eprint.iacr.org/2018/183.pdf



Reference
---------
[1] Ansible文档： http://www.ansible.com.cn/docs/
0xdb8086002d43605b7118a3069818bce5212dc60d
