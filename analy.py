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
df_cg_a100=pd.read_csv(
    "cg_solver_a100.csv",
    header=None,
    names=["name","nnzR","Gflops_cusparse", "Glops_syncfree", "speed_up"])
df_bicg_a100=pd.read_csv(
    "bicg_solver_a100.csv",
    header=None,
    names=["name","nnzR","Gflops_cusparse", "Glops_syncfree", "speed_up"])
df_bicg_pg=pd.read_csv(
    "bicg_petsc_ginkgo.csv",
    header=None,
    names=["name_petsc","name_ginkgo","nnzR","nnzR_x","Gflops_petsc", "Glops_ginkgo", "Glops_syncfree","speed_up_ginkgo","speed_up_petsc"])
df_cg_pg=pd.read_csv(
    "cg_petsc_ginkgo.csv",
    header=None,
    names=["name_petsc","name_ginkgo","nnzR","nnzR_x","Gflops_petsc", "Glops_ginkgo", "Glops_syncfree","speed_up_ginkgo","speed_up_petsc"])
plt.figure(figsize=(6.5, 6), dpi=400)
legend_size=11
label_size=15
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.subplot(2,2,1)
plt.title("CG on A100",fontsize=18)
#size = 20
size = 10
plt.scatter((np.log10(df_cg_pg["nnzR"])), (np.log10(df_cg_pg["Gflops_petsc"])), label="PETSc", s=size, c='r')
plt.scatter((np.log10(df_cg_pg["nnzR_x"])), (np.log10(df_cg_pg["Glops_ginkgo"])), label="Ginkgo", s=size, c='#3366CC')
plt.scatter((np.log10(df_cg_pg["nnzR_x"])), (np.log10(df_cg_pg["Glops_syncfree"])), label="Mille-feuille\n(this work)", s=size, c='teal')
plt.legend(loc='lower right',fontsize=legend_size)
plt.ylim(-3.2, 2.5)
plt.xlim(1, 8)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
plt.ylabel("Gflops ($log10$ scale)",fontsize=label_size)
plt.subplot(2,2,2)
plt.title("BiCGSTAB on A100",fontsize=18)
plt.scatter((np.log10(df_bicg_pg["nnzR"])), (np.log10(df_bicg_pg["Gflops_petsc"])), label="PETSc", s=size, c='r')
plt.scatter((np.log10(df_bicg_pg["nnzR_x"])), (np.log10(df_bicg_pg["Glops_ginkgo"])), label="Ginkgo", s=size, c='#3366CC')
plt.scatter((np.log10(df_bicg_pg["nnzR_x"])), (np.log10(df_bicg_pg["Glops_syncfree"])), label="Mille-feuille\n(this work)", s=size, c='teal')
plt.legend(loc='lower right',fontsize=legend_size)
plt.ylim(-3.2, 2.5)
plt.xlim(0, 8)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.subplot(4,2,5)
plt.scatter((np.log10(df_cg_pg["nnzR"])), np.log10(df_cg_pg["speed_up_petsc"]), label="vs. PETSc", s=size, c="SlateBlue")
#print(np.average(df_cg_pg["speed_up_petsc"]))
ssize=15
plt.legend(loc='upper right',fontsize=legend_size)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylim(-0.8, 2)
plt.xlim(1, 8)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
# plt.subplot(4,2,6)
# plt.scatter((np.log10(df_cg_pg["nnzR"])), np.log10(df_cg_pg["speed_up_petsc"]), label="Petsc", s=size, c="SlateBlue")
plt.subplot(4,2,7)
plt.scatter((np.log10(df_cg_pg["nnzR_x"])), np.log10(df_cg_pg["speed_up_ginkgo"]), label="vs. Ginkgo", s=size, c="#993399")
#print(np.average(df_cg_pg["speed_up_ginkgo"]))
plt.ylim(-0.8, 2)
plt.xlim(1, 8)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("#nonzeros ($log10$ scale)",fontsize=15)
plt.legend(loc='upper right',fontsize=legend_size)
#plt.ylabel("               Speed up\n               ($log10$ scale)",fontsize=label_size)
#plt.ylabel("Speed up ($log10$ scale)",fontsize=label_size)
plt.ylabel("                  Speedup ($log10$ scale)",fontsize=label_size)
plt.subplot(4,2,6)
plt.scatter((np.log10(df_bicg_pg["nnzR"])), np.log10(df_bicg_pg["speed_up_petsc"]), label="vs. PETSc", s=size, c="SlateBlue")
#print(np.average(df_bicg_pg["speed_up_petsc"]))
#print(np.max(df_bicg_pg["speed_up_petsc"]))
plt.xlim(0, 8)
plt.legend(loc='upper right',fontsize=legend_size)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.ylim(-0.8, 2)
plt.subplot(4,2,8)
plt.scatter((np.log10(df_bicg_pg["nnzR_x"])), np.log10(df_bicg_pg["speed_up_ginkgo"]), label="vs. Ginkgo", s=size, c="#993399")
#print(np.average(df_bicg_pg["speed_up_ginkgo"]))
#print(np.max(df_bicg_pg["speed_up_ginkgo"]))
ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])
plt.ylim(-0.8, 2)
plt.xlim(0, 8)
#plt.legend(loc='upper right',fontsize=legend_size,bbox_to_anchor=(0.5,0.,0.5,0.5))
plt.legend(loc='upper right',fontsize=legend_size)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("#nonzeros ($log10$ scale)",fontsize=15)
#plt.subplots_adjust(wspace=0.1)
plt.savefig("Fig 9.pdf", dpi= 400, format = "pdf", bbox_inches='tight', pad_inches=0.0)

plt.figure(figsize=(12, 6), dpi=400)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.subplot(2,4,1)
petsc_color = 'indigo'
trilinos_color = 'teal'
title_size=18
size=10
plt.title("CG on A100",fontsize=title_size)
plt.scatter((np.log10(df_cg_a100["nnzR"])), np.log10(df_cg_a100["Gflops_cusparse"]), label="cuSPARSE", s=size, c=petsc_color)
plt.scatter((np.log10(df_cg_a100["nnzR"])), np.log10(df_cg_a100["Glops_syncfree"]), label="Mille-feuille\n(this work)", s=size, c=trilinos_color)
legend_size=12
label_size=15
plt.legend(loc='lower right',fontsize=legend_size)
plt.ylim(-3.2, 2.5)
plt.xlim(1, 8)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Gflops\n($log10$ scale)",fontsize=label_size)
plt.subplot(2,4,2)
plt.title("BiCGSTAB on A100",fontsize=title_size)
plt.scatter((np.log10(df_bicg_a100["nnzR"])), np.log10(df_bicg_a100["Gflops_cusparse"]), label="cuSPARSE", s=size, c=petsc_color)
plt.scatter((np.log10(df_bicg_a100["nnzR"])), np.log10(df_bicg_a100["Glops_syncfree"]), label="Mille-feuille\n(this work)", s=size, c=trilinos_color)
plt.legend(loc='lower right',fontsize=legend_size)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.subplot(4,4,9)
plt.scatter((np.log10(df_cg_a100["nnzR"])), np.log10(df_cg_a100["speed_up"]), label="vs. cuSPARSE", s=size, c="b")
plt.ylabel("Speedup\n($log10$ scale)",fontsize=label_size)
plt.ylim(-0.5, 1.5)
plt.xlim(1, 8)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(loc='upper right',fontsize=legend_size)
plt.xlabel("#nonzeros ($log10$ scale)",fontsize=label_size)
plt.subplot(4,4,10)
plt.scatter((np.log10(df_bicg_a100["nnzR"])), np.log10(df_bicg_a100["speed_up"]), label="vs. cuSPARSE", s=size, c="b")
ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])
plt.ylim(-0.5, 1.5)
plt.xlim(0, 8)
plt.xlabel("#nonzeros ($log10$ scale)",fontsize=label_size)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(loc='upper right',fontsize=legend_size)
plt.savefig("Fig 8.pdf", dpi= 400, format = "pdf",bbox_inches='tight', pad_inches=0.0)