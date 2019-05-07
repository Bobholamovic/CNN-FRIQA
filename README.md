# CNN-FRIQA
Convulutional Neural Network for Full-Reference color Image Quality Assessment  

This is a lightweight version of [deepIQA](https://arxiv.org/abs/1612.01697). The original project of deepIQA is [here](https://github.com/dmaniry/deepIQA) and the paper is [here](https://arxiv.org/abs/1612.01697). 
  
  
## Environment and Dependencies
> Ubuntu 16.04 64-bit, Visual Studio Code, Python 3.5.2, Pytorch 0.4.0

`requirements.txt` is not included yet.   
  
  
  
## Usage

### Data Preparation
Now the `.json` files are utilized for the data lists. The relative paths of the distorted images and the reference images to `data-dir` and the quality scores (ground-truth values) are stored in three arrays of a `json` object, with the fields specified as `img`, `ref`, and `score`, respectively. For example, `train_data.json` may look like this:

```
{
  "img":
    [
      "distorted/img11_2_4.bmp", 
      "distorted/img6_3_3.bmp"
    ], 
  "ref":
    [
      "images/img11.bmp", 
      "distorted/img6.bmp"
    ], 
  "score":
    [
      0.5503, 
      0.4312
    ]
}
```
(this has been prettified as everthing actually on one line)

Also, there are `val_data.json` for validation subset and `test_data.json` for test subset. The lists are expected to be found at `list-dir`, which will be set to `data-dir` if not specified. 

The scripts for data preparation on `Waterloo` and `TID2013` are provided. 

### Running Code
Start from the root directory of this project, 
```bash
cd src/
```

For training, try
```bash
python iqa.py train --resume pretrianed_model_path --data-dir DIR_OF_DATASET
```

If `pretrained_model_path` is not correctly specified, the model will learn from scratch. 

Use
```bash
python iqa.py train --resume pretrianed_model_path | tee train.log
```
to dump logs. 
  
For evaluation, try
```bash
python iqa.py train --evaluate --resume pretrained_model_path
```
  
For testing, try
```bash
python iqa.py test --resume pretrained_model_path
```
  
The code of testing the model on a single image is desired, yet to be provided. 

As the patches are randomly extracted, there should be a random noise in the output of the model, which explains the slight difference of the performances upon different attempts. 

Some pertrained models ~~and the script to make filename lists~~ are to be uploaded later.   
  
  
  
## Experiment and Performance
Roughly, the `SROCC` value reaches `0.95` or higher under the best condition.

The experiment results are to be added here.  
  
  
  
## Acknowledgement
+ The design of the model is based on [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment](https://arxiv.org/abs/1612.01697)
+ Torch version of `MS-SSIM` from [lizhengwei1992/MS_SSIM_pytorch](https://github.com/lizhengwei1992/MS_SSIM_pytorch.git)
+ Part of the code layout from [fyu/drn](https://github.com/fyu/drn)

With best thanks!  

  
