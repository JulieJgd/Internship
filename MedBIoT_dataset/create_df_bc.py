import pandas as pd
import glob
import os

path_normal = "D:/Documents/TalTech/medbiot/fine-grained/structured_dataset/normal/"
joined_normal_files = os.path.join(path_normal, "*.csv")
joined_normal_list = glob.glob(joined_normal_files)
df_normal = pd.concat(map(pd.read_csv, joined_normal_list), ignore_index=True)
df_normal = df_normal.sample(5000)
df_normal['Malware'] = 0  # Adding the Malware column
df_normal.to_csv("df_normal_bc.csv", index=False)
# print(df_normal)

path_malware = "D:/Documents/TalTech/medbiot/fine-grained/structured_dataset/malware/"
joined_malware_files = os.path.join(path_malware, "*.csv")
joined_malware_list = glob.glob(joined_malware_files)
df_malware = pd.concat(map(pd.read_csv, joined_malware_list), ignore_index=True)
df_malware = df_malware.sample(5000)  # Adding the Malware column
df_malware['Malware'] = 1
df_malware.to_csv("df_malware_bc.csv", index=False)

df_bc = pd.concat(map(pd.read_csv, ['df_normal_bc.csv', 'df_malware_bc.csv']), ignore_index=True).sample(frac=1)
df_bc.to_csv("df_bc.csv", index=False)
print(df_bc)


