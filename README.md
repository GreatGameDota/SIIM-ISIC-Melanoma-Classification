# Melanoma-Classification

My 759th place solution to the SIIM-ISIC Melanoma Classification Competition hosted on Kaggle by SIIM and ISIC.

## Overview

My final solution was an ensemble and TTA of 8 models all trained differently. I used [@cdeotte](https://www.kaggle.com/cdeotte)'s [Triple Stratified Leak-Free KFold CV data](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526) with external data from past competitions and other places as well as upsampling malignant samples.

## Model

There were two different types of models: Image only model and Image + Meta Data model.

Image only model was a simple pretrained efficientnet with a simple double linear layer head.

Meta + Image model concatenates meta data features, after the tabular data goes though a double linear layer head, with the output from the pretrained efficientnet. Then the combined features go through two more linear layers.

Both pretrained models were loaded from [PytorchCV](https://github.com/osmr/imgclsmob).

The models were trained with either BCE or focal loss and with or without label smoothing.

Most models also used Pytorch's new amp mixed precision for training but some models trained without it ended up in the final blend.

## Input and Augmentation

I used Chris Deotte's [Triple Stratified Leak-Free KFold CV](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526) data as input. I mostly trained with an image size of 512x512 but trained a few models with an image size of 384x384.

I also mixed what external data and upsamling I used and the final blend is a mix of models with differing setups. External data included images from the 2018 and 2019 Melanoma competitions as well as unique malignant samples collected by Chris.

For augmentation I used [Albumentations](https://albumentations.readthedocs.io/en/latest/api/augmentations.html) and used SSR, Affine augmentations, Random flips, Random Brightness Contrast, and Cutout. I also used custom augmentations like [Advanced Hair Augmentation](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159176) and [Microscope](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159476#900271) augmentation in some training configurations.

## Training

Training was simple, train for 15 epochs with a batch size of 8-16, depending on the model and configuration, Adam optimizer and ReduceLrOnPlateau scheduler.

## Ensemble and TTA

Ensembling and TTA were the big boosts in LB score. For TTA I did 10 rounds using the same augmentation as training and then enembled all the predictons using the mean.

## Final Submission

For my final submission I ensembled all 8 models that performed the best on public LB or had the smallest CV to LB gap. The ensemble gave the highest public LB score of .9544 and private score .930.

This submission made me jump over 200 positions on the private leaderboard tho I was still only in the top 23%.

Very glad to see that my ensemble of just models with low CV to LB discrepency performed second best. As well has the one five fold ensemble, I tried right as the compeition ended, did not shakeup very much in the private LB.

## What didn't work

- Other models besides efficient nets
- Efficient nets bigger than B4

Did not experiment much so list is rather short

## Final Thoughts

I joined this competition very late with only around 2-3 weeks left. I am content with what I achieved as I didn't shakeup very much.

My previous competition: [PANDA Challenge](https://github.com/GreatGameDota/PANDA-Challenge-Solution)

My next competition: [placeholder]
