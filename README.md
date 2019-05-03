# Implementing Billion-scale semi-supervised learning for image classification using Pytorch
- [Billion-scale semi-supervised learning for image classification paper](https://arxiv.org/abs/1905.00546)
- Author: I. Zeki Yalniz, Herve J ´ egou, Kan Chen, Manohar Paluri, Dhruv Mahajan (Facebook AI)

## Network Architecture
![캡처](https://user-images.githubusercontent.com/22078438/57149837-99763c80-6e07-11e9-8090-48003f2e3242.PNG)

- Step 1:
  - We train on the labeled data to get an initial teacher model
- Step 2:
  - For each class/label, we use the predictions of this teacher model to rank the unlabeled images and pick top-K images to construct a new training data
- Step 3:
  - We use this data to train a student model, which typically differs from the teacher model: hence we can target to reduce the complexity at test time
- Step 4:
  - finally, pre-trained student model is fine-tuned on the initial labeled data to circumvent potential labeling errors.

## Progress
- We Implementing Step 3.

## Reference
- [ResNet 50 Network github](https://github.com/weiaicunzai/pytorch-cifar100)
  - Thank you :)
