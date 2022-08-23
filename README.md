- Search

```shell
python search.py --dataset cifar10
```

- Retraining

```shell
# set genotype in genotypes.py
python retraining.py 
```
Architecture:

&ensp;&ensp;Genotype(normal=[[('max_pool_3x3', 0), ('sep_conv_3x3', 1)], [('max_pool_3x3', 0), ('dil_conv_5x5', 2)], [('max_pool_3x3', 0), ('skip_connect', 1)], [('skip_connect', 0), ('skip_connect', 1)]], normal_concat=range(2, 6), reduce=[[('sep_conv_3x3', 1), ('dil_conv_3x3', 0)], [('skip_connect', 0), ('avg_pool_3x3', 1)], [('skip_connect', 0), ('dil_conv_5x5', 3)], [('max_pool_3x3', 1), ('skip_connect', 0)]], reduce_concat=range(2, 6))

Accuracy on nasbench 301:

&ensp;&ensp;93.575216

Accuracy after retraining :

&ensp;&ensp;95.48
