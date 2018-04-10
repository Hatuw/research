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

Hyperledger
-----------

.. image:: ../assets/tx_hyperledger.png

*Figure 1. hyperledger transaction flow*

Reference
---------
[1] Ansible文档： http://www.ansible.com.cn/docs/
