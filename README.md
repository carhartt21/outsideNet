# outsideNet - CVPR2021 (#8678)
###
This is the code for the submission #8678 at CVPR 2021. 
We provide the pytorch implementation of our model, pretrained weights for the OUTSIDE15K data set and sample images for testing. 
## Run the code
### Requirements
>>python 3.4
CUDA Version: 10.2
To install the required python packages you can run: 
```
pip install -r requirements.txt
```
### Quick start: 
Unzip the pretrained weights:
```
sh unzip_weights.sh
```

Test the network on the sample images: or a folder of images (```$PATH_IMG```):
```
python3 test.py --cfg config/outside15k-resnet50-outsideNet.yaml --imgs test_data/
```

Test the network on costum images or a folder of images::
```
python3 test.py --cfg $CFG --imgs $PATH
```

The config file allows testing additional settings, like multi-scale testing.