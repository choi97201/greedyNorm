# Adversarial-Attention-Model

## Dataset
- [**ATR dataset**](https://drive.google.com/drive/folders/0BzvH3bSnp3E9ZW9paE9kdkJtM3M)
- [**PASCAL VOC 2012**](http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/VOCtrainval_11-May-2012.tar)
- [**Augmented PASCAL VOC 2012**](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)  
- [**Cityscapes**](https://www.cityscapes-dataset.com/)  

## How to Use Augmented PASCAL VOC 2012 made by SBD(Semantic Boundaries Dataset and Benchmark)
- [**How to use 10,582 trainaug images on DeeplabV3 code?**](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/)  
- Bharath Hariharan, Pablo Arbelaez, Lubomir Bourdev, Subhransu Maji and Jitendra Malik, *"Semantic Contours from Inverse Detectors"*, in International Conference on Computer Vision, 2011.  

1. PASCAL VOC 2012와 Augmented PASCAL VOC 2012를 위의 링크에서 다운받고, dataset 폴더에 넣어준다.  
2. ```convert_voc2012_aug.sh``` 를 실행한다.


## How to train this model
기본적인 학습은 ```python main.py train --version 1```로 실행할 수 있다.  
아래는 usage message이다.
```
usage: main.py [-h] [--model FCN / Unet / UnetBN / AttentionUnet / v3]
               [--dataset voc / cityscapes / atr]
               [--method normal / adversarial] [--epochs EPOCHS]
               [--input-shape INPUT_SHAPE] [--learning-rate LEARNING_RATE]
               [--batch-size BATCH_SIZE] [--l1] [--l2] [--multigpu MULTIGPU]
               [--loss-fn crossentropy / focal] [--version VERSION]
               [--checkpoint /path/to/weights/hdf5] [--init-epoch INIT_EPOCH]
               [--multiprocessing MULTIPROCESSING] [--workers WORKERS]
               <command>

positional arguments:
  <command>             'train' or 'test'

optional arguments:
  -h, --help            show this help message and exit
  --model FCN / Unet / UnetBN / AttentionUnet / v3
                        Choose your model
  --dataset voc / cityscapes / atr
                        Choose dataset which will be used
  --method normal / adversarial
                        Choose training method you will use
  --epochs EPOCHS       Enter epochs you want
  --input-shape INPUT_SHAPE
                        Enter input shape you want
  --learning-rate LEARNING_RATE
                        Enter learning rate you want
  --batch-size BATCH_SIZE
                        Enter batch size you want
  --l1                  Set L1 Loss weight
  --l2                  Set L2 Loss weight
  --multigpu MULTIGPU   Set number of gpus you will use
  --loss-fn crossentropy / focal
                        Choose your loss function
  --version VERSION     Enter version
  --checkpoint /path/to/weights/hdf5
                        Path to weight .hdf5 file
  --init-epoch INIT_EPOCH
                        If you use trained weight
  --multiprocessing MULTIPROCESSING
                        If you want to use multiprocessing when training
  --workers WORKERS     Workers you want

```
