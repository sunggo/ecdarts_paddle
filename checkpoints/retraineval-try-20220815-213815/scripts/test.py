# from paddle.io import Dataset, RandomSampler
# import numpy as np
# class RandomDataset(Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples
            
#     def __getitem__(self, idx):
#         image = np.random.random([784]).astype('float32')
#         label = np.random.randint(0, 9, (1, )).astype('int64')
#         return image, label
                
#     def __len__(self):
#         return self.num_samples

# sampler = RandomSampler(data_source=RandomDataset(100),replacement=True,num_samples=6)

# for index in sampler:
#     print(index)
# for index in sampler:
#     print(index)
# import paddle
# test=paddle.randn(shape=[1,100],dtype='float32')
# list=()
# print(list+test)
import numpy as np
import paddle
t=paddle.zeros((2,8))
a=np.zeros((2,8))
a[0][1]=1.0
t=paddle.to_tensor(a)
print("-----")