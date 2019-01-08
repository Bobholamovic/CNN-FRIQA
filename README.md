# CNN-FRIQA
Convulutional Neural Network for Full-Reference color Image Quality Assessment  
  
  
  
## Environment and Dependencies
> Ubuntu 16.04 64-bit, Visual Studio Code, Python 3.5.2, Pytorch 0.1.0

```requirements.txt``` is not included yet.   
  
  
  
## Usage
The first thing to do that split the datasets and create several ```.txt``` files to separately include the data lists for the three phases, namely training, testing and validation. The ```.txt``` files should contain the absolute or relative paths of both the scores (subjective IQA scores like ```MOS``` or ```DMOS```), the reference images (labels), and the distorted images. For example, if you are using the ```TID2013``` database, there must be ```9``` ```txt```s. And each item in the list file should be put in a separate line. Hence, the content of ```train_images.txt``` may look like this

> distorted_images/I01_01_1.bmp  
  distorted_images/i01_01_2.bmp  
  distorted_images/i01_01_3.bmp  
  distorted_images/i01_01_4.bmp  
  distorted_images/i01_01_5.bmp  
  distorted_images/i01_02_1.bmp  
  distorted_images/i01_02_2.bmp  
  ...

And that in ```train_labels.txt```
> reference_images/I01.BMP  
  reference_images/I01.BMP  
  reference_images/I01.BMP  
  reference_images/I01.BMP  
  reference_images/I01.BMP  
  reference_images/I01.BMP  
  reference_images/I01.BMP  
  ...
  
 In ```train_scores```
 > 5.51429  
  5.56757  
  4.94444  
  4.37838  
  3.86486  
  6.02857  
  6.10811  
  ...
  
  
Note that the names of the data list files have to be specified as ```train_images.txt```, ```train_scores.txt```, ```train_labels.txt```, ```val_images.txt```, ```val_scores.txt```, ```val_labels.txt```, ```test_images.txt```, ```test_labels.txt```, and ```test_scores.txt```. 
  
  
For training, try
```bash
python iqa.py train --resume pretrianed_model_path
```
If ```pretrained_model_path``` is not specified, the model will learn from scratch. 
  
  
For evaluation, try
```bash
python iqa.py train --evaluate --resume pretrained_model_path
```
  
  
For testing, try
```bash
python iqa.py test --resume pretrained_model_path
```
  
  
The code of testing the model on a single image is desired, yet to be provided. 

As the patches are randomly extracted, there should be a random noise in the output of the model, which explains the slight difference of performance upon different attempts. 

Some pertrained models and the script to make filename lists are to be uploaded later.   
  
  
  
## Experiment and Performance
Roughly, the ```SROCC``` value reaches ```0.95``` or higher under the best condition.

The experiment results are to be added here.  
  
  
  
## Acknowledgement
+ Torch version of ```SSIM``` from [Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)
+ Part of the code layout from [fyu/drn](https://github.com/fyu/drn)

With best thanks!  
  
  
  
