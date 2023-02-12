# Eva8_s7_Advanced_Training_Concepts

### To Do:
1. Your assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:
    1. pull your Github code to google colab (don't copy-paste code)
    2. prove that you are following the above structure 
    3. that the code in your google collab notebook is NOTHING.. barely anything. There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files
    4. your colab file must:
        1. train resnet18 for 20 epochs on the CIFAR10 dataset
        2. show loss curves for test and train datasets
        3. show a gallery of 10 misclassified images
        4. show gradcam Links to an external site.output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬
        6. Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. 
        7. Train for 20 epochs
        8. Get 10 misclassified images
        9. Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class)
            Apply these transforms while training:
            1. RandomCrop(32, padding=4)
            2. CutOut(16x16)

### Solution

1. Please refer to the following GitHub Link to access the ```EVA8_codebase``` that has been used to train/test the model.
[EVA8_codebase](https://github.com/Rohithmarktricks/eva8_codebase)

2. Please refer the following link. It's jupyter notebook clones the above GitHub repository and uses the modules and packages for this assignment.
[Jupyter Notebook](/Google_colab_work.ipynb)

#### Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
```

#### Training and Validation Logs
```
Epoch: 0
 Loss=1.312723994255066 Batch_id=390 Accuracy=43.76: 100%|██████████| 391/391 [00:10<00:00, 36.59it/s]

Test set: Average loss: 0.0108, Accuracy: 5410/10000 (54.10%)

Epoch: 1
 Loss=1.026256799697876 Batch_id=390 Accuracy=61.36: 100%|██████████| 391/391 [00:09<00:00, 41.08it/s]

Test set: Average loss: 0.0076, Accuracy: 6704/10000 (67.04%)

Epoch: 2
 Loss=0.9139658212661743 Batch_id=390 Accuracy=69.12: 100%|██████████| 391/391 [00:09<00:00, 40.39it/s]

Test set: Average loss: 0.0060, Accuracy: 7385/10000 (73.85%)

Epoch: 3
 Loss=0.8577181100845337 Batch_id=390 Accuracy=73.82: 100%|██████████| 391/391 [00:09<00:00, 42.21it/s]

Test set: Average loss: 0.0050, Accuracy: 7852/10000 (78.52%)

Epoch: 4
 Loss=0.778190016746521 Batch_id=390 Accuracy=76.82: 100%|██████████| 391/391 [00:09<00:00, 40.69it/s]

Test set: Average loss: 0.0051, Accuracy: 7841/10000 (78.41%)

Epoch: 5
 Loss=0.6471050977706909 Batch_id=390 Accuracy=79.55: 100%|██████████| 391/391 [00:09<00:00, 41.42it/s]

Test set: Average loss: 0.0051, Accuracy: 7880/10000 (78.80%)

Epoch: 6
 Loss=0.3535439372062683 Batch_id=390 Accuracy=81.92: 100%|██████████| 391/391 [00:09<00:00, 40.78it/s]

Test set: Average loss: 0.0049, Accuracy: 8012/10000 (80.12%)

Epoch: 7
 Loss=0.45419448614120483 Batch_id=390 Accuracy=83.78: 100%|██████████| 391/391 [00:09<00:00, 40.39it/s]

Test set: Average loss: 0.0039, Accuracy: 8451/10000 (84.51%)

Epoch: 8
 Loss=0.29096391797065735 Batch_id=390 Accuracy=85.51: 100%|██████████| 391/391 [00:09<00:00, 40.77it/s]

Test set: Average loss: 0.0037, Accuracy: 8450/10000 (84.50%)

Epoch: 9
 Loss=0.330562949180603 Batch_id=390 Accuracy=87.22: 100%|██████████| 391/391 [00:09<00:00, 39.35it/s]

Test set: Average loss: 0.0038, Accuracy: 8404/10000 (84.04%)

Epoch: 10
 Loss=0.192952960729599 Batch_id=390 Accuracy=88.42: 100%|██████████| 391/391 [00:09<00:00, 40.71it/s]

Test set: Average loss: 0.0039, Accuracy: 8413/10000 (84.13%)

Epoch: 11
 Loss=0.4632284641265869 Batch_id=390 Accuracy=90.03: 100%|██████████| 391/391 [00:09<00:00, 40.46it/s]

Test set: Average loss: 0.0040, Accuracy: 8532/10000 (85.32%)

Epoch: 12
 Loss=0.14388705790042877 Batch_id=390 Accuracy=91.14: 100%|██████████| 391/391 [00:09<00:00, 40.59it/s]

Test set: Average loss: 0.0039, Accuracy: 8582/10000 (85.82%)

Epoch: 13
 Loss=0.32736510038375854 Batch_id=390 Accuracy=92.15: 100%|██████████| 391/391 [00:09<00:00, 39.99it/s]

Test set: Average loss: 0.0050, Accuracy: 8298/10000 (82.98%)

Epoch: 14
 Loss=0.08685078471899033 Batch_id=390 Accuracy=93.08: 100%|██████████| 391/391 [00:09<00:00, 40.96it/s]

Test set: Average loss: 0.0039, Accuracy: 8588/10000 (85.88%)

Epoch: 15
 Loss=0.11374463140964508 Batch_id=390 Accuracy=93.80: 100%|██████████| 391/391 [00:09<00:00, 40.49it/s]

Test set: Average loss: 0.0043, Accuracy: 8573/10000 (85.73%)

Epoch: 16
 Loss=0.1282021552324295 Batch_id=390 Accuracy=94.20: 100%|██████████| 391/391 [00:09<00:00, 41.19it/s]

Test set: Average loss: 0.0045, Accuracy: 8515/10000 (85.15%)

Epoch: 17
 Loss=0.16123245656490326 Batch_id=390 Accuracy=94.95: 100%|██████████| 391/391 [00:09<00:00, 41.00it/s]

Test set: Average loss: 0.0044, Accuracy: 8581/10000 (85.81%)

Epoch: 18
 Loss=0.0928918644785881 Batch_id=390 Accuracy=95.31: 100%|██████████| 391/391 [00:09<00:00, 40.63it/s]

Test set: Average loss: 0.0050, Accuracy: 8512/10000 (85.12%)

Epoch: 19
 Loss=0.10367634147405624 Batch_id=390 Accuracy=95.77: 100%|██████████| 391/391 [00:09<00:00, 40.11it/s]

Test set: Average loss: 0.0049, Accuracy: 8600/10000 (86.00%)
```

#### Loss (Training and Testing)
[Loss](/images/Loss.png)

#### Accuracy (Training and Testing)
[Accuracy](/images/Acc.png)

#### Misclassified Images
[Misclassified Images](/images/misclassified_images.png)

#### CAMs of Misclassified Images
[CAM1](/images/maps1.png)
[CAM2](/images/maps2.png)
[CAM3](/images/maps3.png)
[CAM4](/images/maps4.png)
[CAM5](/images/maps5.png)
