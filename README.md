# 🚨 Important Update Notice

We have made **significant revisions and improvements** to our early version paper **SpikSSD**. The updated paper has been published on arXiv:

📄 **[SpikeDet: Better Firing Patterns for Accurate and Energy-Efficient Object Detection with Spiking Neuron Networks](https://arxiv.org/abs/2501.15151)**

## 🔥 Major Improvements

### 1. **Theoretical Enhancement**

- More comprehensive analysis of how neuron firing patterns affect object detection
- Corrections to errors in the previous version

### 2. **Methodological Upgrades**

Comprehensive improvements to our approach based on the original ideas:

- Optimizing neuron firing patterns by treating the entire model **as a whole**, solving local firing saturation issues, and proposing the **SpikeDet** model - currently the best model in terms of accuracy and energy efficiency trade-off
- Further development of **MDS-ResNet** leading to the proposed **MDSNet** backbone
- Introduction of **SMFM** based on analysis of how feature fusion modules affect neuron firing patterns

### 3. **Performance Improvements**

Through advanced network architectures, improved training pipelines, and engineering optimizations:

#### 📊 Key Results:

- **COCO 2017 dataset**: `51.4% AP`, outperforming previous SNN-based methods by `2.5% AP` while requiring only **half the power consumption**
- **GEN1 event-based dataset**: `47.6% AP`, outperforming previous SNN-based methods by `7.2% AP` with better energy efficiency
- **URPC 2019 underwater dataset**: `50.1% AP` with only `5.5mJ` energy consumption

------

> **Note**: This represents a major evolution from our initial SpikSSD work. We recommend referring to the updated SpikeDet paper for the most current and accurate information about our approach. This repository will be deleted after our new version repository is updated.



## Required Data

To evaluate or train SpikSSD you will need to download the datasets:

|    Dataset Name    |                             Link                             |
| :----------------: | :----------------------------------------------------------: |
| PASCAL VOC Dataset | [Download PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) |
|    GEN1 Dataset    | [Download Gen1 Automotive Detection Dataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) |

## Training

### Training for SpikSSD

#### SpikSSD-S

python object_detection.py -num_workers 4 -test -save_ckpt -backbone mdsresnet-18 -b 32 -augmentation -fusion -decode spiking

#### SpikSSD-L

python object_detection.py -num_workers 4 -test -save_ckpt -backbone mdsresnet-34 -b 32 -augmentation -fusion -decode spiking

## Evaluation

During evaluation, it is necessary to substitute the relevant pretrained model data into the appropriate root directory.

### Evaluation for SpikSSD

#### SpikSSD-S

python object_detection.py -num_workers 4 -test -save_ckpt -backbone mdsresnet-18 -b 32 -augmentation -fusion -decode spiking -pretrained path_to_model -no_train

#### SpikSSD-L

python object_detection.py -num_workers 4 -test -save_ckpt -backbone mdsresnet-34 -b 32 -augmentation -fusion -decode spiking -pretrained path_to_model -no_train

### Code Acknowledgments

This code is based on [SFOD: Spiking Fusion Object Detector](https://github.com/yimeng-fan/SFOD) . Thanks to the contributors of [SFOD: Spiking Fusion Object Detector](https://github.com/yimeng-fan/SFOD) .

```
@inproceedings{fan2024sfod,
  title={SFOD: Spiking Fusion Object Detector},
  author={Fan, Yimeng and Zhang, Wei and Liu, Changsong and Li, Mingyang and Lu, Wenrui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17191--17200},
  year={2024}
}
```

