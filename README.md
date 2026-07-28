# MLP based Latent Factor Model Experiment Library (Implicit Feedback)

```bash
# INSTALL DEPENDENCIES
conda env create -f env/environment.yaml
conda activate recsys-mlp
```

```py
# LOAD PKG
import pointwise, pairwise, listwise
import pointwiseBayes, pairwiseBayes, listwiseBayes
import components
```

## Implementation Repositories

- [`ncf`](https://github.com/jayarnim/RS-NCF) He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).

- [`dmf`](https://github.com/jayarnim/RS-DMF) Xue, H. J., Dai, X., Zhang, J., Huang, S., & Chen, J. (2017, August). Deep matrix factorization models for recommender systems. In IJCAI (Vol. 17, pp. 3203-3209).

- [`deepcf`](https://github.com/jayarnim/RS-DeepCF) Deng, Z. H., Huang, L., Wang, C. D., Lai, J. H., & Yu, P. S. (2019, July). Deepcf: A unified framework of representation learning and matching function learning in recommender system. In Proceedings of the AAAI conference on artificial intelligence (Vol. 33, No. 01, pp. 61-68).

- [`j-ncf`](https://github.com/jayarnim/RS-J-NCF) Chen, W., Cai, F., Chen, H., & Rijke, M. D. (2019). Joint neural collaborative filtering for recommender systems. ACM Transactions on Information Systems (TOIS), 37(4), 1-30.

- [`dncf`](https://github.com/jayarnim/RS-DNCF) He, G., Zhao, D., & Ding, L. (2021). Dual-embedding based neural collaborative filtering for recommender systems. arXiv preprint arXiv:2102.02549.

- [`convncf`](https://github.com/jayarnim/RS-ConvNCF) He, X., Du, X., Wang, X., Tian, F., Tang, J., & Chua, T. S. (2018). Outer product-based neural collaborative filtering. arXiv preprint arXiv:1808.03912.

- [`comet`](https://github.com/jayarnim/RS-COMET) Lin, Z., Feng, L., Guo, X., Zhang, Y., Yin, R., Kwoh, C. K., & Xu, C. (2023). Comet: Convolutional dimension interaction for collaborative filtering. ACM Transactions on Intelligent Systems and Technology, 14(4), 1-18.

- [`dacr`](https://github.com/jayarnim/RS-DACR) Cui, C., Qin, J., & Ren, Q. (2022). Deep collaborative recommendation algorithm based on attention mechanism. Applied Sciences, 12(20), 10594.

- [`drnet`](https://github.com/jayarnim/RS-DRNet) Ji, D., Xiang, Z., & Li, Y. (2020). Dual relations network for collaborative filtering. IEEE Access, 8, 109747-109757.

- [`delf`](https://github.com/jayarnim/RS-DELF) Cheng, W., Shen, Y., Zhu, Y., & Huang, L. (2018, July). DELF: A dual-embedding based deep latent factor model for recommendation. In IJCAI (Vol. 18, pp. 3329-3335).

- [`fism`](https://github.com/jayarnim/RS-FISM) Kabbur, S., Ning, X., & Karypis, G. (2013, August). Fism: factored item similarity models for top-n recommender systems. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 659-667).

- [`nais`](https://github.com/jayarnim/RS-NAIS) He, X., He, Z., Song, J., Liu, Z., Jiang, Y. G., & Chua, T. S. (2018). NAIS: Neural attentive item similarity model for recommendation. IEEE Transactions on Knowledge and Data Engineering, 30(12), 2354-2366.

- [`bpr`](https://github.com/jayarnim/RS-BPR) Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2012). BPR: Bayesian personalized ranking from implicit feedback. arXiv preprint arXiv:1205.2618.

- [`climf`](https://github.com/jayarnim/RS-CLiMF) Shi, Y., Karatzoglou, A., Baltrunas, L., Larson, M., Oliver, N., & Hanjalic, A. (2012, September). Climf: learning to maximize reciprocal rank with collaborative less-is-more filtering. In Proceedings of the sixth ACM conference on Recommender systems (pp. 139-146).