# PDNet
for RGBD salient object detection


### The pre_trained model
链接: https://pan.baidu.com/s/1rGXSTxtU8iNQa7FPGP83Wg 密码: 9hbq

### Train
Download the dataset, for example, [MSRA10k](http://mmcheng.net/zh/msra10k/), and run the preprocess script
```
python preprocess.py
```

then run
```
python train.py
```

### Test 
Download the pre-trained model and the test picture, run
```
python test.py
```
