import time
tic = time.perf_counter()

import numpy as np
# import random
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
# import torch.utils.data as data_utils
import torch.optim as optim
import multiprocess as mp
from functools import partial
from NN_Defs import TwoLayerMLP, BaseMLP, bootstrap

device = torch.device("cpu")

# data load
X = np.load("Input_Class_AllClasses_Sep.npy")
Y = np.load("Target_Class_AllClasses_Sep.npy")

# CM21 Split
amounts_train = [300,300,300,300,27,70,300]
amounts_val = [82, 531, 104, 278, 6, 17, 4359]



if __name__=="__main__":

    BaseNN = BaseMLP(8, 20, 7, weight_initialize=True)
    optimizer = optim.SGD(BaseNN.parameters(), lr=4e-1, momentum=0.9)

    iters = [(BaseNN, 10000, optimizer, X, Y, amounts_train, amounts_val, device)] * 100
    with mp.Pool(4) as pool:
        ans = pool.starmap(bootstrap, iters)

    scoresR = list(map(list, zip(*ans)))[0]
    scoresP = list(map(list, zip(*ans)))[1]
    scoresA = list(map(list, zip(*ans)))[2]


    scoresR = list(map(list, zip(*scoresR)))
    scoresP = list(map(list, zip(*scoresP)))


    estA = np.mean(scoresA)*100.
    stderrA = np.std(scoresA)*100.

    estR = [np.mean(scoresR[0])*100.,np.mean(scoresR[1])*100.,np.mean(scoresR[2])*100.,np.mean(scoresR[3])*100.,np.mean(scoresR[4])*100.,np.mean(scoresR[5])*100.,np.mean(scoresR[6])*100.]
    stderrR = [np.std(scoresR[0])*100.,np.std(scoresR[1])*100.,np.std(scoresR[2])*100.,np.std(scoresR[3])*100.,np.std(scoresR[4])*100.,np.std(scoresR[5])*100.,np.std(scoresR[6])*100.]

    estP = [np.mean(scoresP[0])*100.,np.mean(scoresP[1])*100.,np.mean(scoresP[2])*100.,np.mean(scoresP[3])*100.,np.mean(scoresP[4])*100.,np.mean(scoresP[5])*100.,np.mean(scoresP[6])*100.]
    stderrP = [np.std(scoresP[0])*100.,np.std(scoresP[1])*100.,np.std(scoresP[2])*100.,np.std(scoresP[3])*100.,np.std(scoresP[4])*100.,np.std(scoresP[5])*100.,np.std(scoresP[6])*100.]
    
    classes = ["Class I", "Class II", "Galaxies", "AGNs", "Shocks", "PAHs", "Stars"]
    f = open("PRAScores_1LayerMLP_7Classes.txt", "w")
    f.write("OneLayerMLP & Recall & Precision & Accuracy//\n")
    for i, cl in enumerate(classes):
        if i==3:
            f.write(cl+"& $"+"{:.1f}".format(estR[i])+"\pm"+"{:.1f}".format(stderrR[i])+"$ & $"+
                "{:.1f}".format(estP[i])+"\pm"+"{:.1f}".format(stderrP[i])+"$ & $"+"{:.1f}".format(estA)+"\pm"+"{:.1f}".format(stderrA)+"$ // \n")
        else:
            f.write(cl+"& $"+"{:.1f}".format(estR[i])+"\pm"+"{:.1f}".format(stderrR[i])+"$ & $"+
                "{:.1f}".format(estP[i])+"\pm"+"{:.1f}".format(stderrP[i])+"$&// \n")

    f.close()

toc = time.perf_counter()
print(f"Completed in {(toc - tic)/60:0.1f} minutes")