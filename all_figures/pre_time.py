import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np
pd.options.mode.chained_assignment = None
sns.set()
sns.set_style("whitegrid")
sns.set_style("ticks")

by_syncfree=pd.read_csv(
    "pre_time.csv",
    header=None,
    names=["mat", "iter", "time_spmv","time_cg","nnzR","norm","time_format","time_format1","time_format2","time_shared"]
    )
by_syncfree["time_spmv"]=by_syncfree["time_cg"]-by_syncfree["time_format1"]-by_syncfree["time_format2"]-by_syncfree["time_shared"]-by_syncfree["time_format"]
by_syncfree["time_pre_all"] = (by_syncfree["time_format1"]+by_syncfree["time_format2"]+by_syncfree["time_shared"]+by_syncfree["time_format"]) + by_syncfree["time_spmv"]
by_syncfree["pre_to_spmv_ratio"] = by_syncfree["time_spmv"] / by_syncfree["time_pre_all"]
by_syncfree["pre_to_pre_ratio"] = (by_syncfree["time_format1"]+by_syncfree["time_format2"]+by_syncfree["time_shared"]) / by_syncfree["time_pre_all"]
pre_new=[]
spmv_new=[]
for i in range(0,len(by_syncfree["pre_to_pre_ratio"])):
    if by_syncfree["pre_to_pre_ratio"][i]>0 and by_syncfree["pre_to_spmv_ratio"][i]>0.9:
            pre_new.append(by_syncfree["pre_to_pre_ratio"][i])
            spmv_new.append(by_syncfree["pre_to_spmv_ratio"][i])

plt.figure(figsize=(6.1,2.6),dpi=400)
bar_width = 1
color_pattern = sns.color_palette()
cnt=0
max_i=0
for i in pre_new:
    cnt=cnt+1
    max_i=max(i,max_i)
plt.bar(np.arange(len(spmv_new)), 
        np.ones(len(spmv_new)), 
        label="Preprocessing", 
        width=bar_width, 
        edgecolor='none', 
        color=color_pattern[0])
plt.bar(np.arange(len(spmv_new)), 
        spmv_new, 
        label="100 iterations of CG and BiCGSTAB", 
        width=bar_width, 
        edgecolor='none', 
        color=color_pattern[1])

plt.legend(ncol=2, bbox_to_anchor=(-0.0145, 1.21), loc='upper left',fontsize=11)

ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
plt.xticks([])

plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],labels=['0%', '20%', '40%', '60%', '80%', '100%'])
plt.tick_params(direction='out')

plt.xlabel("Matrices", labelpad=4,fontsize=15)
plt.ylabel("Preprocessing Ratio",fontsize=15)

plt.xlim(-0.5, len(spmv_new)-0.5)
plt.ylim(0, 1)

plt.savefig("Fig 13.pdf", dpi= 600, format = "pdf", bbox_inches='tight', pad_inches=0.0)