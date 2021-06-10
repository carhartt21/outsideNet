# outsideNet - MMSP 2021 (#164)


###
This is the code for the submission __#164__ at __MMSP 2021__. 

We provide the pytorch implementation of our model, pretrained weights for the OUTSIDE15K data set and 30 randomly sampled images for testing.
We also provide high resolution versions of the figures in the paper and additional images in the folder `figures`.

![Network architecture](/figures/outside15k-network-architecture.png)

## Run the code
### Requirements
Python: >3.2

CUDA: 10.2

To install the required python packages run: 
```
pip install -r requirements.txt
```
### Quick start: 
Unzip the pretrained weights:
```
sh unzip_weights.sh
```

Test the network on the sample images:
```
python3 test.py --cfg config/outside15k-resnet50-outsideNet.yaml --imgs test_data/
```

Test the network on costum images or a folder of images:
```
python3 test.py --cfg config/outside15k-resnet50-outsideNet.yaml --imgs $PATH
```

The config file allows testing additional settings, like multi-scale testing.
