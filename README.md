# Per-element Secure Aggregation Protocol

This repository is an implementation of the Per-element SecAgg protocol [1]. This implementation is designed to run on the ABIDE simulator [2, 3]. Please install and run the ABIDE simulator.

The execution can be performed with the following command:
```
python abides.py -c flamingo -n 128 -i 1 -p 1 
```

This implementation is based on Flamingo [4, 5], a committee-based Secure Aggregation protocol.


[1] https://arxiv.org/abs/2508.04285
[2] https://github.com/abides-sim/abides
[3] D. Byrd, M. Hybinette, and T. H. Balch, “ABIDES: Towards high-fidelity multi-agent market simulation,” in Proceedings of the 2020 ACM SIGSIM Conference on Principles of Advanced Discrete Simulation, New York, NY, USA: ACM, June 2020. doi: 10.1145/3384441.3395986.
[4] https://github.com/eniac/flamingo
[5] Y. Ma, J. Woods, S. Angel, A. Polychroniadou, and T. Rabin, “Flamingo: Multi-round single-server secure aggregation with applications to private federated learning,” in 2023 IEEE Symposium on Security and Privacy (SP), IEEE, May 2023, pp. 477–496.
