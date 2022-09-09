import pandas as pd
import os

path = "D:/Documents/TalTech/detection_of_IoT_botnet_attacks_N_BaIoT/"

list_col = ["Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell", "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera", "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
        "SimpleHome_XCS7_1002_WHT_Security_Camera", "SimpleHome_XCS7_1003_WHT_Security_Camera"]

for i in list_col:
    if os.path.exists("D:/Documents/TalTech/BaIoT_new_2/50_50/"+i+"/attacks.csv"):
        os.remove("D:/Documents/TalTech/BaIoT_new_2/50_50/"+i+"/attacks.csv")
for i in list_col:
    if os.path.exists("D:/Documents/TalTech/BaIoT_new_2/50_50/"+i+"/benign_traffic.csv"):
        os.remove("D:/Documents/TalTech/BaIoT_new_2/50_50/"+i+"/benign_traffic.csv")

# Danmini_Doorbell
path_1 = os.path.join(path, "Danmini_Doorbell/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1_1 = f1.sample(4900)
        f1_2 = f1.sample(420)
    f1_1['Malware'] = 1  # create the malware column
    f1_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Danmini_Doorbell/attacks.csv", index=False)
    f1_2['Malware'] = 1  # create the malware column
    f1_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Danmini_Doorbell/attacks.csv", index=False)

path_2 = os.path.join(path, "Danmini_Doorbell/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2_1 = f2.sample(4900)
f2_2 = f2.sample(3780)
f2_1['Malware'] = 0  # create the malware column
f2_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Danmini_Doorbell/benign_traffic.csv", index=False)
f2_2['Malware'] = 0  # create the malware column
f2_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Danmini_Doorbell/benign_traffic.csv", index=False)

# Ecobee_Thermostat
path_1 = os.path.join(path, "Ecobee_Thermostat/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1_1 = f1.sample(1300)
        f1_2 = f1.sample(111)
    f1_1['Malware'] = 1  # create the malware column
    f1_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Ecobee_Thermostat/attacks.csv", index=False)
    f1_2['Malware'] = 1  # create the malware column
    f1_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Ecobee_Thermostat/attacks.csv", index=False)

path_2 = os.path.join(path, "Ecobee_Thermostat/")
f2_1 = f2.sample(1300)
f2_2 = f2.sample(1004)
f2_1['Malware'] = 0  # create the malware column
f2_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Ecobee_Thermostat/benign_traffic.csv", index=False)
f2_2['Malware'] = 0  # create the malware column
f2_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Ecobee_Thermostat/benign_traffic.csv", index=False)

# Ennio_Doorbell
path_1 = os.path.join(path, "Ennio_Doorbell/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1_1 = f1.sample(3900)
        f1_2 = f1.sample(334)
    f1_1['Malware'] = 1  # create the malware column
    f1_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Ennio_Doorbell/attacks.csv", index=False)
    f1_2['Malware'] = 1  # create the malware column
    f1_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Ennio_Doorbell/attacks.csv", index=False)

path_2 = os.path.join(path, "Ennio_Doorbell/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2_1 = f2.sample(3900)
f2_2 = f2.sample(3009)
f2_1['Malware'] = 0  # create the malware column
f2_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Ennio_Doorbell/benign_traffic.csv", index=False)
f2_2['Malware'] = 0  # create the malware column
f2_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Ennio_Doorbell/benign_traffic.csv", index=False)

# Philips_B120N10_Baby_Monitor
path_1 = os.path.join(path, "Philips_B120N10_Baby_Monitor/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1_1 = f1.sample(17500)
        f1_2 = f1.sample(1500)
    f1_1['Malware'] = 1  # create the malware column
    f1_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Philips_B120N10_Baby_Monitor/attacks.csv", index=False)
    f1_2['Malware'] = 1  # create the malware column
    f1_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Philips_B120N10_Baby_Monitor/attacks.csv", index=False)

path_2 = os.path.join(path, "Philips_B120N10_Baby_Monitor/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2_1 = f2.sample(17500)
f2_2 = f2.sample(13500)
f2_1['Malware'] = 0  # create the malware column
f2_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Philips_B120N10_Baby_Monitor/benign_traffic.csv", index=False)
f2_2['Malware'] = 0  # create the malware column
f2_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Philips_B120N10_Baby_Monitor/benign_traffic.csv", index=False)

# Provision_PT_737E_Security_Camera
path_1 = os.path.join(path, "Provision_PT_737E_Security_Camera/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1_1 = f1.sample(6200)
        f1_2 = f1.sample(531)
    f1_1['Malware'] = 1  # create the malware column
    f1_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Provision_PT_737E_Security_Camera/attacks.csv", index=False)
    f1_2['Malware'] = 1  # create the malware column
    f1_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Provision_PT_737E_Security_Camera/attacks.csv", index=False)

path_2 = os.path.join(path, "Provision_PT_737E_Security_Camera/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2_1 = f2.sample(6200)
f2_2 = f2.sample(4784)
f2_1['Malware'] = 0  # create the malware column
f2_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Provision_PT_737E_Security_Camera/benign_traffic.csv", index=False)
f2_2['Malware'] = 0  # create the malware column
f2_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Provision_PT_737E_Security_Camera/benign_traffic.csv", index=False)

# Provision_PT_838_Security_Camera
path_1 = os.path.join(path, "Provision_PT_838_Security_Camera/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1_1 = f1.sample(9800)
        f1_2 = f1.sample(840)
    f1_1['Malware'] = 1  # create the malware column
    f1_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Provision_PT_838_Security_Camera/attacks.csv", index=False)
    f1_2['Malware'] = 1  # create the malware column
    f1_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Provision_PT_838_Security_Camera/attacks.csv", index=False)

path_2 = os.path.join(path, "Provision_PT_838_Security_Camera/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2_1 = f2.sample(9800)
f2_2 = f2.sample(7560)
f2_1['Malware'] = 0  # create the malware column
f2_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Provision_PT_838_Security_Camera/benign_traffic.csv", index=False)
f2_2['Malware'] = 0  # create the malware column
f2_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Provision_PT_838_Security_Camera/benign_traffic.csv", index=False)

# Samsung_SNH_1011_N_Webcam
path_1 = os.path.join(path, "Samsung_SNH_1011_N_Webcam/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1_1 = f1.sample(5200)
        f1_2 = f1.sample(445)
    f1_1['Malware'] = 1  # create the malware column
    f1_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Samsung_SNH_1011_N_Webcam/attacks.csv", index=False)
    f1_2['Malware'] = 1  # create the malware column
    f1_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Samsung_SNH_1011_N_Webcam/attacks.csv", index=False)

path_2 = os.path.join(path, "Samsung_SNH_1011_N_Webcam/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2_1 = f2.sample(5200)
f2_2 = f2.sample(4013)
f2_1['Malware'] = 0  # create the malware column
f2_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/Samsung_SNH_1011_N_Webcam/benign_traffic.csv", index=False)
f2_2['Malware'] = 0  # create the malware column
f2_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/Samsung_SNH_1011_N_Webcam/benign_traffic.csv", index=False)

# SimpleHome_XCS7_1002_WHT_Security_Camera
path_1 = os.path.join(path, "SimpleHome_XCS7_1002_WHT_Security_Camera/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1_1 = f1.sample(4600)
        f1_2 = f1.sample(394)
    f1_1['Malware'] = 1  # create the malware column
    f1_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/SimpleHome_XCS7_1002_WHT_Security_Camera/attacks.csv", index=False)
    f1_2['Malware'] = 1  # create the malware column
    f1_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/SimpleHome_XCS7_1002_WHT_Security_Camera/attacks.csv", index=False)

path_2 = os.path.join(path, "SimpleHome_XCS7_1002_WHT_Security_Camera/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2_1 = f2.sample(4600)
f2_2 = f2.sample(3549)
f2_1['Malware'] = 0  # create the malware column
f2_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/SimpleHome_XCS7_1002_WHT_Security_Camera/benign_traffic.csv", index=False)
f2_2['Malware'] = 0  # create the malware column
f2_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/SimpleHome_XCS7_1002_WHT_Security_Camera/benign_traffic.csv", index=False)

# SimpleHome_XCS7_1003_WHT_Security_Camera
path_1 = os.path.join(path, "SimpleHome_XCS7_1003_WHT_Security_Camera/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1_1 = f1.sample(1900)
        f1_2 = f1.sample(162)
    f1_1['Malware'] = 1  # create the malware column
    f1_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/SimpleHome_XCS7_1003_WHT_Security_Camera/attacks.csv",
                index=False)
    f1_2['Malware'] = 1  # create the malware column
    f1_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/SimpleHome_XCS7_1003_WHT_Security_Camera/attacks.csv",
                index=False)

path_2 = os.path.join(path, "SimpleHome_XCS7_1003_WHT_Security_Camera/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2_1 = f2.sample(1900)
f2_2 = f2.sample(1467)
f2_1['Malware'] = 0  # create the malware column
f2_1.to_csv("D:/Documents/TalTech/BaIoT_new_2/50_50/SimpleHome_XCS7_1003_WHT_Security_Camera/benign_traffic.csv", index=False)
f2_2['Malware'] = 0  # create the malware column
f2_2.to_csv("D:/Documents/TalTech/BaIoT_new_2/90_10/SimpleHome_XCS7_1003_WHT_Security_Camera/benign_traffic.csv", index=False)
