# Star_Formation

Each `.ipynb` notebook specifies what ML classification algorithm is being tested.

The targets and four IRAC bands as pulled from Cornu and Montillaud (2021) (https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/647/A116).

**The final project report for PHYS 555 is located in `Final_Project.ipynb`. This report summarizes the work and includes example runs of all algorithms and the MLP, and can be run with only `custom_dataloader.py`, `NN_Defs.py`, and the data files `Input_Class_AllClasses_Sep.npy` and `Target_Class_AllClasses_Sep.npy`.**

## custom_dataloader.py

This file contains a custom dataloader that takes in multiple lists of integers, representing how many of each class in the original data is to be included for training/validation and testing sets.

## network_runner.py

A basic script used to run certain hyperparameters through the BaseMLP class neural network built in `NN_Defs.py`.

## NN_Defs.py

This script contains basic functions used for the training and validation setps in the neural network used. Additionally contained within, is the BaseMLP class used to build all instances of the neural network used in this project.
