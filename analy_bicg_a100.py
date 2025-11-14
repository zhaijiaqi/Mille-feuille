import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import re
import matplotlib.ticker as mtick
pd.options.mode.chained_assignment = None
sns.set()
sns.set_style("whitegrid")

df_syncfree = pd.read_csv(
    "bicg_AD_new.csv",
    header=None,
    names=["name","iter", "time_spmv", "time_bicg","nnzR", "l2_norm","norm"])
for i in range(0,len(df_syncfree)):
    df_syncfree["name"][i]=df_syncfree["name"][i].split("/")[2]
dict_time = {}
dict_nnz = {}
for i in range(0,len(df_syncfree["name"])):
    dict_time.setdefault(df_syncfree["name"][i], []).append(df_syncfree["time_bicg"][i])
    dict_nnz.setdefault(df_syncfree["name"][i], []).append(df_syncfree["nnzR"][i])
name_syncfree=[]
time_syncfree=[]
nnz_syncfree=[]
for arr in dict_time:
    min_e=100000
    for ele in dict_time[arr]:
        if ele<min_e:
            min_e=ele
    name_syncfree.append(arr)
    time_syncfree.append(min_e)
    nnz_syncfree.append(dict_nnz[arr][0])
df_syncfree_new = pd.DataFrame({"name": name_syncfree, "time_bicg": time_syncfree, "nnzR":nnz_syncfree})


df_cusparse = pd.read_csv(
    "bicg_cusparse_a100.csv",
    header=None,
    names=["name","iter", "time_spmv", "time_bicg","nnzR", "l2_norm","time_dot","norm"])
for i in range(0,len(df_cusparse)):
    df_cusparse["name"][i]=df_cusparse["name"][i].split("/")[2]
merged_df = pd.merge(df_syncfree_new, df_cusparse, on='name')
merged_df["time_bicg"]=merged_df["time_bicg_y"]*5
merged_df["speed_up"]=merged_df["time_bicg"]/merged_df["time_bicg_x"]
for i in range(0,len(merged_df)):
    if merged_df["speed_up"][i]<1:
        print(merged_df["name"][i])
df_ginkgo = pd.read_csv(
    "ginkgo_bicg_a100.csv",
    header=None,
    names=["iter","time_ginkgo_bicg","time_precond", "name","nnzR","time_all"])
for i in range(0,len(df_ginkgo)):
    df_ginkgo["name"][i]=df_ginkgo["name"][i].split("/")[2]
df_ns = pd.read_csv(
    "mat2757new.csv",
    header=None,
    names=["name","n"]
)

for i in range(0,len(df_ginkgo)):
    df_ginkgo["time_ginkgo_bicg"][i]-=df_ginkgo["time_precond"][i]
    df_ginkgo["time_all"][i]-=df_ginkgo["time_precond"][i]
    if df_ginkgo["iter"][i]==0:
        df_ginkgo["iter"][i]=1
df_ginkgo["time_ginkgo_bicg"]=(df_ginkgo["time_ginkgo_bicg"]/df_ginkgo["iter"])*50
df_ginkgo["time_all"]=(df_ginkgo["time_all"]/df_ginkgo["iter"])*50
merged_ginkgo = pd.merge(merged_df, df_ginkgo, on='name')
n = [] 
for i in range(len(merged_ginkgo["name"])):
    for j in range(len(df_ns["name"])):
        if(merged_ginkgo["name"][i].split(".")[0] == df_ns["name"][j]):
            n.append(df_ns["n"][j])
            break
merged_ginkgo["n"] = [0] * len(merged_ginkgo)
for i in range(len(merged_ginkgo)):
    merged_ginkgo["n"][i] = n[i]
merged_ginkgo["speed_up"]=merged_ginkgo["time_ginkgo_bicg"]/merged_ginkgo["time_bicg_x"]
merged_ginkgo["Glops"]=(4 * merged_ginkgo["nnzR_x"] + 7 * 2 * merged_ginkgo["n"]) / ((merged_ginkgo["time_ginkgo_bicg"]/ 50) * 1e6)
merged_ginkgo["Glops_sync"]=(4 * merged_ginkgo["nnzR_x"] + 7 * 2 * merged_ginkgo["n"]) / ((merged_ginkgo["time_bicg_x"]/ 50) * 1e6)
df_petsc = pd.read_csv(
    "petsc_bicg_a100.csv",
    header=None,
    names=["name","nnzR_petsc", "n","norm","iter","time_solve","time_pre"])
for i in range(0,len(df_petsc)):
    df_petsc["name"][i]=df_petsc["name"][i].split("/")[2]
for i in range(0,len(df_petsc["iter"])):
    if df_petsc["iter"][i]==0:
        df_petsc["iter"][i]=1
df_petsc["time_solve"]=(df_petsc["time_solve"]/df_petsc["iter"])*50
merged_petsc = pd.merge(df_petsc, merged_ginkgo, on='name')
merged_petsc["Gflops_solve"]=(4 * merged_petsc["nnzR_x"] + 7 * 2 * merged_petsc["n_x"]) / ((merged_petsc["time_solve"]/ 50) * 1e6)
merged_petsc["speed_up_pet"]=merged_petsc["time_solve"]/merged_petsc["time_bicg_x"]
bicg_petsc_ginkgo = pd.DataFrame({"name_petsc":merged_petsc["name"],"name_ginkgo":merged_ginkgo["name"],"nnzR": merged_petsc["nnzR_x"],"nnzR_x":merged_ginkgo["nnzR_x"],"Gflops_petsc":merged_petsc["Gflops_solve"],"Glops_ginkgo":merged_ginkgo["Glops"],"Glops_syncfree":merged_ginkgo["Glops_sync"],
                            "speed_up_ginkgo":merged_ginkgo["speed_up"],"speed_up_petsc":merged_petsc["speed_up_pet"]})
bicg_petsc_ginkgo.to_csv("bicg_petsc_ginkgo.csv", index=False,header=None)

df_ns = pd.read_csv(
    "mat2757new.csv",
    header=None,
    names=["name","n"]
)
n = [] #维度
for i in range(len(merged_df["name"])):
    for j in range(len(df_ns["name"])):
        if(merged_df["name"][i].split(".")[0] == df_ns["name"][j]):
            n.append(df_ns["n"][j])
            break
merged_df["n"] = [0] * len(merged_df)
for i in range(len(merged_df)):
    merged_df["n"][i] = n[i]

merged_df["Glops"]=(4 * merged_df["nnzR_x"] + 7 * 2 * merged_df["n"]) / ((merged_df["time_bicg"]/ 50) * 1e6)
merged_df["Glops_x"]=(4 * merged_df["nnzR_x"] + 7 * 2 * merged_df["n"]) / ((merged_df["time_bicg_x"]/ 50) * 1e6)
bicg_sover_a100 = pd.DataFrame({"name":merged_df["name"],"nnzR": merged_df["nnzR_x"],"Gflops_cusparse":merged_df["Glops"],"Glops_syncfree":merged_df["Glops_x"],"speed_up":merged_df["speed_up"]})
bicg_sover_a100.to_csv("bicg_solver_a100.csv", index=False,header=None)