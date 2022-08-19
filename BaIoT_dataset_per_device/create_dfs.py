import pandas as pd
import os

path = "D:/Documents/TalTech/detection_of_IoT_botnet_attacks_N_BaIoT/"

for i in os.listdir(path):
    if os.path.exists("D:/Documents/TalTech/BaIoT_ai_lab/"+i+"/attacks.csv"):
        os.remove("D:/Documents/TalTech/BaIoT_ai_lab/"+i+"/attacks.csv")
    f = pd.DataFrame()

    path_1 = os.path.join(path, i)
    if len(os.listdir(path_1)) == 2:
        path_1_1 = os.path.join(path_1, "attacks/")
        for j in os.listdir(path_1_1):
            f1 = pd.read_csv(os.path.join(path_1_1, j))
            f = f.append(f1)  # f1[:100]
        f['Malware'] = 1
        f.to_csv("D:/Documents/TalTech/BaIoT_ai_lab/"+i+"/attacks.csv", index=False)

    df = pd.read_csv("D:/Documents/TalTech/BaIoT_ai_lab/"+i+"/attacks.csv")
    # print(df.head())
    # print(df.shape)

for i in os.listdir(path):
    path_2 = os.path.join(path, i)
    f1 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
    if 'Malware' not in f1.columns:
        f1['Malware'] = 0
        f1.to_csv("D:/Documents/TalTech/BaIoT_ai_lab/"+i+"/benign_traffic.csv", index=False)
    df1 = pd.read_csv("D:/Documents/TalTech/BaIoT_ai_lab/"+i+"/benign_traffic.csv")
    # print(df1.head())

