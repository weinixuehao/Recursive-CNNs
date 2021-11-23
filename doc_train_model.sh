#!/bin/bash

python train_model.py --name NameOfExperiment -i dataset/generated_dataset/selfCollectedData_DocCyclic dataset/generated_dataset/smartdocData_DocTrainC --lr 0.5 --schedule 20 30 35 -v dataset/generated_dataset/smartDocData_DocTestC --batch-size 16 --model-type resnet --loader ram