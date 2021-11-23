#!/bin/bash

python train_model.py --name CornerModel -i dataset/generated_dataset/cornerTrain64 --lr 0.5 --schedule 20 30 35 -v dataset/generated_dataset/selfCollectedData_CornDetec --batch-size 16 --model-type resnet --loader ram --dataset corner