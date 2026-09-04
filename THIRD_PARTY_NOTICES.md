# Third-Party Notices

The root `LICENSE` applies to VICAL-owned work. The repository also contains code and data-list material imported from or derived from the projects below. Those portions remain governed by their original licenses, reproduced verbatim under `LICENSES/`.

| Upstream project | Use in this repository | Bundled terms | Authoritative source |
|---|---|---|---|
| Classifier-Balancing / Decoupling | ResNet and ResNeXt backbone code under `expert_model/fb_resnets/` and long-tailed dataset conventions | [`LICENSES/CLASSIFIER_BALANCING.txt`](LICENSES/CLASSIFIER_BALANCING.txt) | https://github.com/facebookresearch/classifier-balancing/blob/main/LICENSE |
| Open Long-Tailed Recognition (OLTR) | Portions retained by the Classifier-Balancing backbones and ImageNet-LT data-list conventions | [`LICENSES/OLTR.txt`](LICENSES/OLTR.txt) | https://github.com/zhmiao/OpenLongTailRecognition-OLTR/blob/master/LICENSE |
| LDAM-DRW | Normalized classifiers, CIFAR ResNet code, and loss implementations | [`LICENSES/LDAM_DRW.txt`](LICENSES/LDAM_DRW.txt) | https://github.com/kaidic/LDAM-DRW/blob/master/LICENSE |
| RIDE | Multi-expert backbones, model wrappers, losses, base classes, and utilities | [`LICENSES/RIDE.txt`](LICENSES/RIDE.txt) | https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/LICENSE |
| timm / PyTorch Image Models | The modified `randaugment.py` implementation, originally authored by Ross Wightman and adapted from TensorFlow TPU AutoAugment | [`LICENSES/TIMM_RANDAUGMENT.txt`](LICENSES/TIMM_RANDAUGMENT.txt) | https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE and https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/auto_augment.py |
| Balanced Contrastive Learning | Data pipeline and training-code lineage used in the large-scale release | [`LICENSES/BALANCED_CONTRASTIVE_LEARNING.txt`](LICENSES/BALANCED_CONTRASTIVE_LEARNING.txt) | https://github.com/FlamieZhu/Balanced-Contrastive-Learning/blob/main/LICENSE |

The timm RandAugment source identifies TensorFlow TPU's EfficientNet AutoAugment implementation as an adaptation source: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py. Both the bundled timm terms and that source are under the Apache License 2.0.

Project names belong to their respective owners. Attribution here does not imply endorsement.
