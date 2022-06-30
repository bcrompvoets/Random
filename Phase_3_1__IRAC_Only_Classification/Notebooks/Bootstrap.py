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

# YSO_EG_Stars Train
X_tr = np.load("../Data_and_Results/c2d_Inputs_CLOUDS_Train.npy") # Load input data
Y_tr = np.load("../Data_and_Results/c2d_Targets_CLOUDS_Train.npy") # Load target data
inp_tr = np.float32(X_tr)
tar_tr = np.float32(Y_tr)

# YSO_EG_Stars Valid
X_va = np.load("../Data_and_Results/c2d_Inputs_CORES_Valid.npy") # Load input data
Y_va = np.load("../Data_and_Results/c2d_Targets_CORES_Valid.npy") # Load target data
inp_va = np.float32(X_va)
tar_va = np.float32(Y_va)

# YSO_EG_Stars Test
X_te = np.load("../Data_and_Results/Rap_Inputs_Test.npy") # Load input data
Y_te = np.load("../Data_and_Results/Rap_Targets_Test.npy") # Load target data
inp_te = np.float32(X_te)
tar_te = np.float32(Y_te)


if __name__=="__main__":

    BaseNN = TwoLayerMLP(9, 20, 3, weight_initialize=True)
    optimizer = optim.SGD(BaseNN.parameters(), lr=0.01, momentum=0.9)

    iters = [(BaseNN, 3000, optimizer, inp_tr, tar_tr, inp_va, tar_va, inp_te, tar_te, device)] * 100
    with mp.Pool(6) as pool:
        ans = pool.starmap(bootstrap, iters)

    scoresR = list(map(list, zip(*ans)))[0]
    scoresP = list(map(list, zip(*ans)))[1]
    scoresA = list(map(list, zip(*ans)))[2]


    scoresR = list(map(list, zip(*scoresR)))
    scoresP = list(map(list, zip(*scoresP)))


    estA = np.mean(scoresA)*100.
    stderrA = np.std(scoresA)*100.

    estR = [np.mean(scoresR[0])*100.,np.mean(scoresR[1])*100.,np.mean(scoresR[2])*100.]
    stderrR = [np.std(scoresR[0])*100.,np.std(scoresR[1])*100.,np.std(scoresR[2])*100.]

    estP = [np.mean(scoresP[0])*100.,np.mean(scoresP[1])*100.,np.mean(scoresP[2])*100.]
    stderrP = [np.std(scoresP[0])*100.,np.std(scoresP[1])*100.,np.std(scoresP[2])*100.]
    
    classes = ["YSO", "EG", "Star"]
    f = open("PRAScores_2LayerMLP.txt", "w")
    f.write("TwoLayerMLP & Recall & Precision & Accuracy//\n")
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