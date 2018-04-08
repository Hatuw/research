BlockChain Research
===================

Ethereum
--------

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


.. image:: ../assets/tx_hyperledger.png

*Figure 1. hyperledger transaction flow*

Reference
---------
[1] Ansible文档： http://www.ansible.com.cn/docs/