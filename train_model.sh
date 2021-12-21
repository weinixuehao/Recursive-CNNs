#!/bin/bash

set -e # exit when any command fails

if [ "$1" == "dev" ]; then
    doc_dataset_train="dataset/selfCollectedData_DocCyclic"
    doc_dataset_test="dataset/my_doc_test"

    corner_dataset_train="dataset/my_corner_train"
    corner_dataset_test="dataset/my_corner_test"
else
    doc_dataset_train="dataset/selfCollectedData_DocCyclic dataset/smartdocData_DocTrainC dataset/my_doc_train dataset/sythetic_doc_train"
    doc_dataset_test="dataset/smartDocData_DocTestC dataset/my_doc_test dataset/sythetic_doc_test"

    corner_dataset_train="dataset/cornerTrain64 dataset/my_corner_train dataset/sythetic_corner_train"
    corner_dataset_test="dataset/selfCollectedData_CornDetec dataset/my_corner_test dataset/sythetic_corner_test"
fi

# echo "doc_dataset_train=$doc_dataset_train corner_dataset_test=$corner_dataset_test"

# 1、resnet model

python train_model.py --name DocModel -i $doc_dataset_train \
--lr 0.5 --schedule 20 30 35 -v $doc_dataset_test --batch-size 32 --model-type resnet --loader ram

python train_model.py --name CornerModel -i $corner_dataset_train \
--lr 0.5 --schedule 20 30 35 -v $corner_dataset_test --batch-size 32 --model-type resnet --loader ram --dataset corner

# 2、mobile_net model

# python train_model.py --name DocModel -i $doc_dataset_train \
# --lr 0.5 --schedule 20 30 35 -v $doc_dataset_test --batch-size 16 --model-type shallow --loader ram

# python train_model.py --name CornerModel -i $corner_dataset_train \
# --lr 0.5 --schedule 20 30 35 -v $corner_dataset_test --batch-size 16 --model-type shallow --loader ram --dataset corner