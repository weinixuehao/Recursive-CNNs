#!/bin/bash

set -e # exit when any command fails

if [[ "$1" != "" ]]; then
    epochs="$1"
else
    epochs=40 # The number of epochs is 40.
fi

echo $epochs

python train_model.py --name DocModel -i dataset/selfCollectedData_DocCyclic dataset/smartdocData_DocTrainC dataset/my_doc_train \
--lr 0.5 --schedule 20 30 35 -v dataset/smartDocData_DocTestC dataset/my_doc_test --batch-size 16 --model-type resnet --loader ram --epochs $epochs

python train_model.py --name CornerModel -i dataset/cornerTrain64 dataset/my_corner_train \
--lr 0.5 --schedule 20 30 35 -v dataset/selfCollectedData_CornDetec dataset/my_corner_test --batch-size 16 --model-type resnet --loader ram --dataset corner --epochs $epochs