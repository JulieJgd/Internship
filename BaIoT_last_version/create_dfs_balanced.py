import pandas as pd
import os

path = "D:/Documents/TalTech/detection_of_IoT_botnet_attacks_N_BaIoT/"

list_col = ["Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell", "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera", "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
        "SimpleHome_XCS7_1002_WHT_Security_Camera", "SimpleHome_XCS7_1003_WHT_Security_Camera"]

for i in list_col:
    if os.path.exists("D:/Documents/TalTech/BaIoT_new/"+i+"/attacks.csv"):
        os.remove("D:/Documents/TalTech/BaIoT_new/"+i+"/attacks.csv")
for i in list_col:
    if os.path.exists("D:/Documents/TalTech/BaIoT_new/"+i+"/benign_traffic.csv"):
        os.remove("D:/Documents/TalTech/BaIoT_new/"+i+"/benign_traffic.csv")

# Danmini_Doorbell
path_1 = os.path.join(path, "Danmini_Doorbell/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1 = f1.sample(4900)
    f1['Malware'] = 1  # create the malware column
    f1.to_csv("D:/Documents/TalTech/BaIoT_new/Danmini_Doorbell/attacks.csv", index=False)

path_2 = os.path.join(path, "Danmini_Doorbell/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2 = f2.sample(4900)
f2['Malware'] = 0  # create the malware column
f2.to_csv("D:/Documents/TalTech/BaIoT_new/Danmini_Doorbell/benign_traffic.csv", index=False)

# Ecobee_Thermostat
path_1 = os.path.join(path, "Ecobee_Thermostat/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1 = f1.sample(1300)
    f1['Malware'] = 1  # create the malware column
    f1.to_csv("D:/Documents/TalTech/BaIoT_new/Ecobee_Thermostat/attacks.csv", index=False)

path_2 = os.path.join(path, "Ecobee_Thermostat/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2 = f2.sample(1300)
f2['Malware'] = 0  # create the malware column
f2.to_csv("D:/Documents/TalTech/BaIoT_new/Ecobee_Thermostat/benign_traffic.csv", index=False)

# Ennio_Doorbell
path_1 = os.path.join(path, "Ennio_Doorbell/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1 = f1.sample(3900)
    f1['Malware'] = 1  # create the malware column
    f1.to_csv("D:/Documents/TalTech/BaIoT_new/Ennio_Doorbell/attacks.csv", index=False)

path_2 = os.path.join(path, "Ennio_Doorbell/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2 = f2.sample(3900)
f2['Malware'] = 0  # create the malware column
f2.to_csv("D:/Documents/TalTech/BaIoT_new/Ennio_Doorbell/benign_traffic.csv", index=False)

# Philips_B120N10_Baby_Monitor
path_1 = os.path.join(path, "Philips_B120N10_Baby_Monitor/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1 = f1.sample(17500)
    f1['Malware'] = 1  # create the malware column
    f1.to_csv("D:/Documents/TalTech/BaIoT_new/Philips_B120N10_Baby_Monitor/attacks.csv", index=False)

path_2 = os.path.join(path, "Philips_B120N10_Baby_Monitor/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2 = f2.sample(17500)
f2['Malware'] = 0  # create the malware column
f2.to_csv("D:/Documents/TalTech/BaIoT_new/Philips_B120N10_Baby_Monitor/benign_traffic.csv", index=False)

# Provision_PT_737E_Security_Camera
path_1 = os.path.join(path, "Provision_PT_737E_Security_Camera/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1 = f1.sample(6200)
    f1['Malware'] = 1  # create the malware column
    f1.to_csv("D:/Documents/TalTech/BaIoT_new/Provision_PT_737E_Security_Camera/attacks.csv", index=False)

path_2 = os.path.join(path, "Provision_PT_737E_Security_Camera/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2 = f2.sample(6200)
f2['Malware'] = 0  # create the malware column
f2.to_csv("D:/Documents/TalTech/BaIoT_new/Provision_PT_737E_Security_Camera/benign_traffic.csv", index=False)

# Provision_PT_838_Security_Camera
path_1 = os.path.join(path, "Provision_PT_838_Security_Camera/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1 = f1.sample(9800)
    f1['Malware'] = 1  # create the malware column
    f1.to_csv("D:/Documents/TalTech/BaIoT_new/Provision_PT_838_Security_Camera/attacks.csv", index=False)

path_2 = os.path.join(path, "Provision_PT_838_Security_Camera/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2 = f2.sample(9800)
f2['Malware'] = 0  # create the malware column
f2.to_csv("D:/Documents/TalTech/BaIoT_new/Provision_PT_838_Security_Camera/benign_traffic.csv", index=False)

# Samsung_SNH_1011_N_Webcam
path_1 = os.path.join(path, "Samsung_SNH_1011_N_Webcam/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1 = f1.sample(5200)
    f1['Malware'] = 1  # create the malware column
    f1.to_csv("D:/Documents/TalTech/BaIoT_new/Samsung_SNH_1011_N_Webcam/attacks.csv", index=False)

path_2 = os.path.join(path, "Samsung_SNH_1011_N_Webcam/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2 = f2.sample(5200)
f2['Malware'] = 0  # create the malware column
f2.to_csv("D:/Documents/TalTech/BaIoT_new/Samsung_SNH_1011_N_Webcam/benign_traffic.csv", index=False)

# SimpleHome_XCS7_1002_WHT_Security_Camera
path_1 = os.path.join(path, "SimpleHome_XCS7_1002_WHT_Security_Camera/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1 = f1.sample(4600)
    f1['Malware'] = 1  # create the malware column
    f1.to_csv("D:/Documents/TalTech/BaIoT_new/SimpleHome_XCS7_1002_WHT_Security_Camera/attacks.csv", index=False)

path_2 = os.path.join(path, "SimpleHome_XCS7_1002_WHT_Security_Camera/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2 = f2.sample(4600)
f2['Malware'] = 0  # create the malware column
f2.to_csv("D:/Documents/TalTech/BaIoT_new/SimpleHome_XCS7_1002_WHT_Security_Camera/benign_traffic.csv", index=False)

# SimpleHome_XCS7_1003_WHT_Security_Camera
path_1 = os.path.join(path, "SimpleHome_XCS7_1003_WHT_Security_Camera/")
if len(os.listdir(path_1)) == 2:
    path_1_1 = os.path.join(path_1, "attacks/")
    for j in os.listdir(path_1_1):
        f1 = pd.read_csv(os.path.join(path_1_1, j))
        f1 = f1.sample(1900)
    f1['Malware'] = 1  # create the malware column
    f1.to_csv("D:/Documents/TalTech/BaIoT_new/SimpleHome_XCS7_1003_WHT_Security_Camera/attacks.csv", index=False)

path_2 = os.path.join(path, "SimpleHome_XCS7_1003_WHT_Security_Camera/")
f2 = pd.read_csv(os.path.join(path_2, "benign_traffic.csv"))
f2 = f2.sample(1900)
f2['Malware'] = 0  # create the malware column
f2.to_csv("D:/Documents/TalTech/BaIoT_new/SimpleHome_XCS7_1003_WHT_Security_Camera/benign_traffic.csv", index=False)
