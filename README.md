## DAMSDet: Dynamic Adaptive Multispectral Detection Transformer with Competitive Query Selection and Adaptive Feature Fusion
## [DAMSDet](https://arxiv.org/pdf/2403.00326)

## Installation
We use PaddlePaddle2.5(Stable) with the CUDA11.7 Linux version and our python version is 3.8. Please refer to the official guide of [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop) for installation guide.

## Data Preparation
We provide annotated JSON files and dataset partitioning files for each dataset in **dataset** folder, so you only need to download each dataset images from internet (**M3FD**, **FLIR_align**, **LLVIP**, **VEDAI**). Then, you need to put each dataset imgs in the **dataset/coco_xxx** directory according to the **train.txt** and **val.txt**.

## Pretrained weights
You can download coco pretrained weights on [coco_pretrain_weights](https://drive.google.com/file/d/13IfjgrLvoUQq8CCoMDdZ3skUmmHHWLcu/view?usp=sharing).

You can download **M3FD** pretrained weights on [M3FD_pretrain_weights](https://drive.google.com/file/d/1V1ohLT2YeUyX_AcMogO8tnWFiHqMgLvH/view?usp=sharing)

You can download **FLIR_align** pretrained weights on [FLIR_pretrain_weights](https://drive.google.com/file/d/107rDvGqXT0MkDABM6WAB9Qim1srcqP0P/view?usp=sharing)

You can download **LLVIP** pretrained weights on [LLVIP_pretrain_weights](https://drive.google.com/file/d/1y3_q7lSQvq_NZy0NfehKf9vMtQ7R0CEh/view?usp=sharing)

You can download **VEDAI** pretrained weights on [VEDAI_pretrain_weights](https://drive.google.com/file/d/1iCGCxTjIUpB6nWWemyPR-f7ujUAgb39H/view?usp=sharing)

## Train
train on **M3FD** 

    python tools/train.py -c configs/damsdet/damsdet_r50vd_m3fd.yml -o pretrain_weights=coco_pretrain_weights.pdparams --eval

train on **FLIR** 

    python tools/train.py -c configs/damsdet/damsdet_r50vd_flir.yml -o pretrain_weights=coco_pretrain_weights.pdparams --eval

train on **LLVIP** 

    python tools/train.py -c configs/damsdet/damsdet_r50vd_llvip.yml -o pretrain_weights=coco_pretrain_weights.pdparams --eval

train on **VEDAI** 

    python tools/train.py -c configs/damsdet/damsdet_r50vd_vedai.yml -o pretrain_weights=coco_pretrain_weights.pdparams --eval


## Evaluate
 evaluation on **M3FD** 

    python tools/eval.py -c configs/damsdet/damsdet_r50vd_m3fd.yml --classwise -o weights=output/M3FD/damsdet_r50vd_m3fd/best_model

evaluation on **FLIR**  

    python tools/eval.py -c configs/damsdet/damsdet_r50vd_flir.yml --classwise -o weights=output/FLIR/damsdet_r50vd_flir/best_model

evaluation on **LLVIP**  

    python tools/eval.py -c configs/damsdet/damsdet_r50vd_llvip.yml --classwise -o weights=output/LLVIP/damsdet_r50vd_llvip/best_model

evaluation on **VEDAI**

    python tools/eval.py -c configs/damsdet/damsdet_r50vd_vedai.yml --classwise -o weights=output/VEDAI/damsdet_r50vd_vedai/best_model


## Inference
inference on **M3FD** 

    python tools/multi_infer.py -c configs/damsdet/damsdet_r50vd_m3fd.yml --infer_vis_dir=dataset/coco_m3fd/val_vis_img/ --infer_ir_dir=dataset/coco_m3fd/val_ir_img --output_dir=(detection saved path) -o weights=output/M3FD/damsdet_r50vd_m3fd/best_model

inference on **FLIR** 

    python tools/multi_infer.py -c configs/damsdet/damsdet_r50vd_flir.yml --infer_vis_dir=dataset/coco_FLIR_align/val_imgs/vis_imgs --infer_ir_dir=dataset/coco_FLIR_align/val_imgs/ir_imgs --output_dir=(detection saved path) -o weights=output/M3FD/damsdet_r50vd_m3fd/best_model

inference on **LLVIP** 

    python tools/multi_infer.py -c configs/damsdet/damsdet_r50vd_llvip.yml --infer_vis_dir=dataset/coco_LLVIP/val_imgs/vis_imgs --infer_ir_dir=dataset/coco_LLVIP/val_imgs/ir_imgs --output_dir=(detection saved path) -o weights=output/LLVIP/damsdet_r50vd_llvip/best_model

inference on **VEDAI** 

    python tools/multi_infer.py -c configs/damsdet/damsdet_r50vd_vedai.yml --infer_vis_dir=dataset/coco_VEDAI/val_imgs/vis_imgs --infer_ir_dir=dataset/coco_VEDAI/val_imgs/ir_imgs --output_dir=(detection saved path) -o weights=output/LLVIP/damsdet_r50vd_llvip/best_model


## Acknowledgement
For the implementation, we rely heavily on [Paddle](https://github.com/PaddlePaddle/Paddle) and [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)


## Reference

    @article{guo2024damsdet,
      title={DAMSDet: Dynamic Adaptive Multispectral Detection Transformer with Competitive Query Selection and Adaptive Feature Fusion},
      author={Guo, Junjie and Gao, Chenqiang and Liu, Fangcen and Meng, Deyu and Gao, Xinbo},
      journal={arXiv e-prints},
      pages={arXiv--2403},
      year={2024}
    }
