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
    "cg_AD_new.csv",
    header=None,
    names=["name","iter", "time_spmv", "time_cg","nnzR", "l2_norm","time_format","norm"])
for i in range(0,len(df_syncfree)):
    df_syncfree["name"][i]=df_syncfree["name"][i].split("/")[2]
dict_time = {}
dict_nnz = {}
for i in range(0,len(df_syncfree["name"])):
    dict_time.setdefault(df_syncfree["name"][i], []).append(df_syncfree["time_cg"][i])
    dict_nnz.setdefault(df_syncfree["name"][i], []).append(df_syncfree["nnzR"][i])
name_syncfree=[]
time_syncfree=[]
time_syncfree_ori=[]
nnz_syncfree=[]
each_nnz_syncfree=[]
for arr in dict_time:
    min_e=100000
    for ele in dict_time[arr]:
        if ele<min_e:
            min_e=ele
    name_syncfree.append(arr)
    time_syncfree.append(min_e)
    time_syncfree_ori.append(min_e)
    nnz_syncfree.append(dict_nnz[arr][0])
df_syncfree_new = pd.DataFrame({"name": name_syncfree, "time_cg": time_syncfree, "nnzR":nnz_syncfree})

# df_set = pd.read_csv(
#     "cg_AD_set.csv",
#     header=None,
#     names=["name","time_cg","nnzR","each_nnz"])
# name=[]
# time_cg=[]
# nnzR=[]
# each_nnz=[]
# tmp_nnz=0
# for i in range(0,len(df_set)):
#     if df_set["nnzR"][i]!=tmp_nnz:
#         tmp_nnz=df_set["nnzR"][i]
#         name.append(df_set["name"][i])
#         time_cg.append(df_set["time_cg"][i])
#         nnzR.append(df_set["nnzR"][i])
#         each_nnz.append(df_set["each_nnz"][i])
# df_syncfree_1 = pd.DataFrame({"name": name, "time_cg": time_cg, "nnzR":nnzR,"each_nnz":each_nnz})
# df_syncfree_1.to_csv("cg_AD_set_simple.csv", index=False,header=None)

df_cusparse = pd.read_csv(
    "cg_cusparse_a100.csv",
    header=None,
    names=["name","iter", "time_spmv", "time_cg","nnzR", "l2_norm","time_format"])
for i in range(0,len(df_cusparse)):
    df_cusparse["name"][i]=df_cusparse["name"][i].split("/")[2]
merged_df = pd.merge(df_syncfree_new, df_cusparse, on='name')
merged_df["time_cg"]=merged_df["time_cg_y"]
merged_df["speed_up"]=merged_df["time_cg"]/df_syncfree_new["time_cg"]

df_ginkgo = pd.read_csv(
    "ginkgo_cg_a100.csv",
    header=None,
    names=["iter","time_ginkgo_cg", "name","nnzR","time_all"])
for i in range(0,len(df_ginkgo)):
    df_ginkgo["name"][i]=df_ginkgo["name"][i].split("/")[2]
for i in range(0,len(df_ginkgo)):
    df_ginkgo["time_ginkgo_cg"][i]-=0.065
    df_ginkgo["time_all"][i]-=0.065
df_ginkgo["time_ginkgo_cg"]=(df_ginkgo["time_ginkgo_cg"]/df_ginkgo["iter"])*100
df_ginkgo["time_all"]=(df_ginkgo["time_all"]/df_ginkgo["iter"])*100
merged_ginkgo = pd.merge(df_ginkgo, df_syncfree_new, on='name')
df_petsc = pd.read_csv(
    "petsc_cg_a100.csv",
    header=None,
    names=["name","nnzR", "n","norm","iter","time_solve","time_pre"])
for i in range(0,len(df_petsc)):
    df_petsc["name"][i]=df_petsc["name"][i].split("/")[2]
df_petsc["time_solve"]=(df_petsc["time_solve"]/df_petsc["iter"])*100
merged_petsc = pd.merge(df_petsc, merged_ginkgo, on='name')
merged_petsc["Gflops_solve"]=(2 * merged_petsc["nnzR"] + 5 * 2 * merged_petsc["n"]) / ((merged_petsc["time_solve"]/ 100) * 1e6)
merged_ginkgo["speed_up"]=merged_ginkgo["time_ginkgo_cg"]/merged_ginkgo["time_cg"]
merged_petsc["speed_up"]=merged_petsc["time_solve"]/merged_petsc["time_cg"]
df_ns = pd.read_csv(
    "mat2757new.csv",
    header=None,
    names=["name","n"]
)
n = [] 
for i in range(len(merged_ginkgo["name"])):
    for j in range(len(df_ns["name"])):
        if(merged_ginkgo["name"][i].split(".")[0] == df_ns["name"][j]):
            n.append(df_ns["n"][j])
            break
merged_ginkgo["n"] = [0] * len(merged_ginkgo)
for i in range(len(merged_ginkgo)):
    merged_ginkgo["n"][i] = n[i]
merged_ginkgo["Glops"]=(2 * merged_ginkgo["nnzR_x"] + 5 * 2 * merged_ginkgo["n"]) / ((merged_ginkgo["time_ginkgo_cg"]/ 100) * 1e6)
merged_ginkgo["Glops_sync"]=(2 * merged_ginkgo["nnzR_x"] + 5 * 2 * merged_ginkgo["n"]) / ((merged_ginkgo["time_cg"]/ 100) * 1e6)
merged_all=pd.merge(merged_petsc, merged_ginkgo, on='name')
cg_petsc_ginkgo = pd.DataFrame({"name_petsc":merged_all["name"],"name_ginkgo":merged_all["name"],"nnzR": merged_all["nnzR"],"nnzR_x":merged_all["nnzR"],"Gflops_petsc":merged_all["Gflops_solve"],"Glops_ginkgo":merged_all["Glops"],"Glops_syncfree":merged_all["Glops_sync"],
                                  "speed_up_ginkgo":merged_all["speed_up_y"],"speed_up_petsc":merged_all["speed_up_x"]})
cg_petsc_ginkgo.to_csv("cg_petsc_ginkgo.csv", index=False,header=None)

df_ns = pd.read_csv(
    "mat2757new.csv",
    header=None,
    names=["name","n"]
)
n = [] 
for i in range(len(merged_df["name"])):
    for j in range(len(df_ns["name"])):
        if(merged_df["name"][i].split(".")[0] == df_ns["name"][j]):
            n.append(df_ns["n"][j])
            break
merged_df["n"] = [0] * len(merged_df)
for i in range(len(merged_df)):
    merged_df["n"][i] = n[i]

merged_df["Glops"]=(2 * merged_df["nnzR_x"] + 5 * 2 * merged_df["n"]) / ((merged_df["time_cg"]/ 100) * 1e6)
df_syncfree_new["Glops"]=(2 * df_syncfree_new["nnzR"] + 5 * 2 * merged_df["n"]) / ((df_syncfree_new["time_cg"]/ 100) * 1e6)
cg_solver_a100 = pd.DataFrame({"name":merged_df["name"],"nnzR": merged_df["nnzR_x"],"Gflops_cusparse":merged_df["Glops"],"Glops_syncfree":df_syncfree_new["Glops"],"speed_up":merged_df["speed_up"]})
cg_solver_a100.to_csv("cg_solver_a100.csv", index=False,header=None)