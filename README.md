# IoU-Uniform R-CNN
This repository provides the codes for paper "IoU-Uniform R-CNN: Breaking Through the Limitations of RPN"
- This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection) framework.

## Setup
Please follow official [installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md) and [getting_started](https://github.com/open-mmlab/mmdetection/blob/master/docs/GETTING_STARTED.md) guides.

## Training
``./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [--validate] [other_optional_args]``
Note:
- Config files of IoU-Uniform R-CNN:
  - configs/faster_rcnn_r50_fpn_1x_IoU_reg.py
  - configs/faster_rcnn_r101_fpn_1x_IoU_reg.py
  - configs/cascade_rcnn_r50_fpn_IoU_reg_1x.py
  - configs/cascade_rcnn_r101_fpn_IoU_reg_1x.py
  - configs/pascal_voc/faster_rcnn_r50_fpn_1x_IoU_reg_voc0712.py
  - configs/pascal_voc/faster_rcnn_r50_fpn_1x_reg_separate_voc0712.py
  - configs/pascal_voc/cascade_rcnn_r50_fpn_1x_IoU_reg_voc0712.py

- We train IoU-Uniform R-CNN and accompanied detectors with 2 GPUs and 2 img/GPU. According to the Linear Scaling Rule, you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU.


##  Testing
``./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]``

## TODO
- [x] Release IoU-Uniform R-CNN code base
- [ ] Release trained models
