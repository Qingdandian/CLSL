
### Data Preparation
1. Download dataset and organize them as follow:
```
|datasets
|---- MSCOCO
|-------- annotations
|-------- train2014
|-------- val2014
|---- NUS-WIDE
|-------- Flickr
|-------- Groundtruth
|-------- ImageList
|-------- NUS_WID_Tags
|-------- Concepts81.txt
|---- VOC2007
|-------- Annotations
|-------- ImageSets
|-------- JPEGImages
|-------- SegmentationClass
|-------- SegmentationObject
```

2. Preprocess using following commands:
```bash
python scripts/mscoco.py
python scripts/nuswide.py
python scripts/voc2007.py
python scripts/embedding.py --data [mscoco, nuswide, voc2007]
python scripts/embedding.py --data voc2007
python scripts/embedding.py --data nuswide
python scripts/embedding.py --data mscoco

e.g.
--data_dir /public/home/hpc2307070100001/Datasets/voc/datasets/VOC2007
--save_dir /public/home/hpc2307070100001/Datasets/voc/data/voc2007
--glove_dir /public/home/hpc2307070100001/PreModelEncoder/Glove/glove.840B.300d.txt
--bert_dir /public/home/hpc2307070100001/PreModelEncoder/Bert/bert-base-uncased
```

### Requirements
```
torch >= 1.12.0
torchvision >= 0.13.0
```

### Training
One can use following commands to train model and reproduce the results reported in paper.
```bash
VOC2007
python train.py --model baseline --arch resnet101 --data voc2007 --loss asl --partial=0.9 --img-size=448 --batch-size 64 --lr 0.00009 --ema-decay 0.9983 --pos --num_workers 6
python train.py --model sarlpwoJLS --arch resnet101 --data voc2007 --loss asl --partial=0.9 --img-size=448 --batch-size 64 --lr 0.00009 --ema-decay 0.9983 --pos --num_workers 6
python train.py --model sarlpwoSRFL --arch resnet101 --data voc2007 --loss asl --partial=0.9 --img-size=448 --batch-size 64 --lr 0.00009 --ema-decay 0.9983 --pos --num_workers 6
python train.py --model sarlp_clip_lora --data voc2007 --loss asl --partial=0.8 --img-size=448 --batch-size 32 --lr 0.000024 --ema-decay 0.9983 --pos --num-layers 1 --num_workers 6 \
 --clip_model_path /public/home/hpc2307070100001/PreModelEncoder/clip/RN101.pt
python train.py --model sarlp --arch resnet101 --data voc2007 --loss asl --partial=0.1 --img-size=448 --batch-size 64 --lr 0.00009 --ema-decay 0.9983 --pos --num_workers 6

MS-COCO
python train.py --model baseline --arch resnet101 --data mscoco --loss asl --partial=0.9 --img-size=448 --batch-size 52 --lr 0.00005 --pos --num_workers 6
python train.py --model sarlpwoJLS --arch resnet101 --data mscoco --loss asl --partial=0.9 --img-size=448 --batch-size 52 --lr 0.00005 --pos --num_workers 6
python train.py --model sarlpwoSRFL --arch resnet101 --data mscoco --loss asl --partial=0.9 --img-size=448 --batch-size 52 --lr 0.00005 --pos --num_workers 6
python train.py --model sarlp_clip_lora --data mscoco --loss asl --partial=0.8 --img-size=448 --batch-size 32 --lr 0.00005 --pos --num_workers 6  \
--clip_model_path /public/home/hpc2307070100001/PreModelEncoder/clip/RN101.pt
python train.py --model sarlp --arch resnet101 --data mscoco --loss asl --partial=0.1 --img-size=448 --batch-size 52 --lr 0.00005 --pos --num_workers 6

NUS-WIDE
python train.py --model baseline --arch resnet101 --data nuswide --loss asl --partial=0.9 --img-size=224 --batch-size 128 --lr 0.00009 --pos --num_workers 6
python train.py --model sarlpwoJLS --arch resnet101 --data nuswide --loss asl --partial=0.9 --img-size=224 --batch-size 128 --lr 0.00009 --pos --num_workers 6
python train.py --model sarlpwoSRFL --arch resnet101 --data nuswide --loss asl --partial=0.9 --img-size=224 --batch-size 128 --lr 0.00009 --pos --num_workers 6
python train.py --model sarlp_clip_lora --data nuswide --loss asl --partial=0.9 --img-size=224 --batch-size 128 --lr 0.00009 --pos --num_workers 6 --ema-decay 0.9983 \
--clip_model_path /public/home/hpc2307070100001/PreModelEncoder/clip/RN101.pt
python train.py --model sarlp --arch resnet101 --data nuswide --loss asl --partial=0.1 --img-size=224 --batch-size 128 --lr 0.00009 --pos --num_workers 6 --ema-decay 0.9983
```
### Evaluation

```bash
python evaluate.py --exp-dir experiments/sarlp_voc2007/exp58/checkpoints/best_ema_model.pth # evaluation for ResNet101 on VOC2007
```

### Visualization
To visualize the attention heatmaps in the paper, run the following command and visualization results are saved in the `visualization` folder of the corresponding experiment.
```bash
python infer.py --exp-dir experiments/sarlp_voc2007/exp58
```

### Citation
```
@misc{he2025collaborativelearningsemanticawarefeature,
      title={Collaborative Learning of Semantic-Aware Feature Learning and Label Recovery for Multi-Label Image Recognition with Incomplete Labels}, 
      author={Zhi-Fen He and Ren-Dong Xie and Bo Li and Bin Liu and Jin-Yan Hu},
      year={2025},
      eprint={2510.10055},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.10055}, 
}

```

## Acknowledgement
We would like to thank Xuelin Zhu for providing code for [SGRE](https://github.com/jasonseu/SGRE). We borrowed and refactored a large portion of his code in the implementation of our work.
