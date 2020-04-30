# Similitude Attentive Relation Network for Click-Through Rate Prediction

This is the code for Similitude Attentive Relation Network forClick-Through Rate Prediction.
It is an example for Amazon Electronics dataset and you can try other datasets easily.

## prepare data
#### split_by_user

``sh prepare_data_split_by_user_electronics.sh``

## train model

``python script/train.py data/Electronics_split_by_user/``

## Requirements:
Python 2.7

Tensorflow 1.4

Note: the code is heavily dependent on [DIEN](https://github.com/mouna99/dien) code in the year of 2018.
