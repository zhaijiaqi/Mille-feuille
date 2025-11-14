import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def syncfree():
    sns.set()
    sns.set_style("whitegrid")

    df_syncfree_mix_precision = pd.read_csv(
        "mix_cg.csv",
        header=None,
        names=["name", "iter", "time_spmv", "time_cg", "nnzR", "l2_norm", "time_format", "index", "each_nnz", "norm"])

    dict_time = {}
    for i in range(0, len(df_syncfree_mix_precision["name"])):
        dict_time.setdefault(df_syncfree_mix_precision["name"][i], []).append(df_syncfree_mix_precision["time_spmv"][i])
    name_syncfree_mixed = []
    time_syncfree_mixed = []
    for arr in dict_time:
        min_e = 100000
        for ele in dict_time[arr]:
            if ele < min_e:
                min_e = ele
        name_syncfree_mixed.append(arr)
        time_syncfree_mixed.append(min_e)
    print(len(name_syncfree_mixed))


    df_syncfree = pd.read_csv(
        "syncfree_cg.csv",
        header=None,
        names=["name", "iter", "time_spmv", "time_cg", "nnzR", "l2_norm", "time_format", "index", "each_nnz", "norm"])
    dict_time = {}
    for i in range(0, len(df_syncfree["name"])):
        dict_time.setdefault(df_syncfree["name"][i], []).append(df_syncfree["time_spmv"][i])
    name_syncfree = []
    time_syncfree_ori = []

    for arr in dict_time:
        min_e = 100000
        for ele in dict_time[arr]:
            if ele < min_e:
                min_e = ele
        name_syncfree.append(arr)
        time_syncfree_ori.append(min_e)
    for i in range(0, len(name_syncfree)):
        name_syncfree[i] = name_syncfree[i].split('/')[6].split('.')[0]


    df_ratio_mix = pd.read_csv(
        "cg_ratio.csv",
        header=None,
        names=["name", "time_cg", "nnzR", "fp32", "fp16", "fp8", "fp64", "bypass"])

    fp64_ratio = []
    fp32_ratio = []
    fp16_ratio = []
    fp8_ratio = []
    bypass_ratio = []
    name_new = []
    name = [
        'Muu', 'qa8fm', 'thermomech_TC',
        'thermal', 'tmt_sym', 't3dh_e',
        'bp_600', 'fv1', 'cz2548',
        'shallow_water1', 't2em', 'rajat24'
    ]


    name_for_x_label=[
    'Muu','qa8f...','ther...',
    'ther...','tmt_...','t3dh...',
    'bp_6...','fv1','cz25...',
    'shall...','t2em','raja...'
    ]

    for name_matrix in name:
        for i in range(0, len(df_ratio_mix)):
            if (df_ratio_mix["name"][i].split('/')[6].split('.')[0] == name_matrix):
                name_new.append(name_matrix)
                fp64_ratio.append(df_ratio_mix["fp64"][i])
                fp32_ratio.append(df_ratio_mix["fp32"][i])
                fp16_ratio.append(df_ratio_mix["fp16"][i])
                fp8_ratio.append(df_ratio_mix["fp8"][i])
                bypass_ratio.append(df_ratio_mix["bypass"][i])

    x = np.arange(len(name_new))
    bar_width = 0.4
    spacing = 0.0

    c1 = '#3399FF'
    c2 = '#FF9900'
    c3 = '#66A438'
    c4 = '#FF0000'
    c5 = '#9400cc'
    plt.figure(figsize=(10, 8), dpi=400)
    plt.subplots_adjust(hspace=0.10, wspace=0.03)


    ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=3, colspan=2)

    ax1.bar(x, fp64_ratio, bar_width, color=c1, label='FP64', edgecolor='black')
    ax1.bar(x, fp32_ratio, bar_width, bottom=fp64_ratio,
            color=c2, label='FP32', edgecolor='black')
    ax1.bar(x, fp16_ratio, bar_width, bottom=[sum(x) for x in zip(fp64_ratio, fp32_ratio)],
            color=c3, label='FP16', edgecolor='black')
    ax1.bar(x, fp8_ratio, bar_width, bottom=[sum(x) for x in zip(fp64_ratio, fp32_ratio, fp16_ratio)],
            color=c4, label='FP8', edgecolor='black')
    ax1.bar(x, bypass_ratio, bar_width, bottom=[sum(x) for x in zip(fp64_ratio, fp32_ratio, fp16_ratio, fp8_ratio)],
            color=c5, label='Bypass', edgecolor='black')
    x1 = [0] * len(name_new)
    for i in range(len(name_new)):
        x1[i] = i

    ax1 = plt.gca()

    ax1.set_ylim(0, 1)
    labels_y = ['0%', '20%', '40%', '60%', '80%', '100%']
    ax1.set_xticks([])
    ax1.set_yticks(range(0, 105, 20), labels=labels_y, fontsize=25)
    ax1.set_ylabel("Precision distribution", fontsize=25)
    ax1.legend(bbox_to_anchor=(1.015, 1.16), loc='upper center', ncol=5, fontsize=16)

    data_mix = []
    for matrix_name in name_new:
        for i in range(0, len(name_syncfree_mixed)):
            if name_syncfree_mixed[i].split('/')[6].split('.')[0] == matrix_name:
                data_mix.append(time_syncfree_mixed[i])
    data_sync = []
    for matrix_name in name_new:
        for i in range(0, len(name_syncfree)):
            if name_syncfree[i] == matrix_name:
                data_sync.append(time_syncfree_ori[i])
    speedup_data = []
    for i in range(len(data_sync)):
        speedup_data.append(data_sync[i] / data_mix[i])
        print(data_sync[i] / data_mix[i])
    x = np.arange(len(speedup_data))
    width = 0.1


    ax2 = plt.subplot2grid((5, 4), (3, 0), rowspan=2, colspan=2)

    ax2.bar(x, speedup_data, bar_width, color='#386AE8', label='double', edgecolor='black', zorder=2)
    for i, val in enumerate(speedup_data):
        ax2.text(x[i], speedup_data[i] + 0.05, f'{val:.2f}', fontsize=20, ha='center', rotation=90, zorder=3,
                 fontweight='bold')

    ax2 = plt.gca()
    ax2.set_xticks(x)
    ax2.set_xticklabels(name_for_x_label, rotation=270, fontsize=25)
    ax2.xaxis.labelpad = 8
    labels_y = ['1.0', '1.5', '1.8']
    ax2.set_yticks((1.0, 1.5, 1.8), labels=labels_y, fontsize=25)
    ax2.set_ylim(1, 1.8)
    ax2.set_xlabel('Matrices in CG', fontsize=25)
    ax2.set_ylabel("Speedup", fontsize=25)


    df_ratio_mix = pd.read_csv(
        "bicg_ratio.csv",
        header=None,
        names=["name", "time_cg", "nnzR", "fp32", "fp16", "fp8", "fp64", "bypass"])

    fp64_ratio = []
    fp32_ratio = []
    fp16_ratio = []
    fp8_ratio = []
    bypass_ratio = []
    name_new = []

    name = [
        'swang2', 'chipcool1', 'nemeth08',
        'swang1', 'Zhao1', 'wang3',
        't2dal_a', 'mesh3e1', 'torso2',
        'nemeth07', 'Pd', 'cz308'
    ]

    name_for_x_label1=[
    'swan...','chip...','neme...',
    'swan...','Zhao...','wang...',
    't2da...','mesh...', 'tors...',
    'neme...','Pd','cz30...'
    ]

    for name_matrix in name:
        for i in range(0, len(df_ratio_mix)):
            if (df_ratio_mix["name"][i].split('/')[6].split('.')[0] == name_matrix):
                name_new.append(name_matrix)
                fp64_ratio.append(df_ratio_mix["fp64"][i])
                fp32_ratio.append(df_ratio_mix["fp32"][i])
                fp16_ratio.append(df_ratio_mix["fp16"][i])
                fp8_ratio.append(df_ratio_mix["fp8"][i])
                bypass_ratio.append(df_ratio_mix["bypass"][i])

    x = np.arange(len(name_new))
    bar_width = 0.4
    spacing = 0.0

    c1 = '#3399FF'
    c2 = '#FF9900'
    c3 = '#66A438'
    c4 = '#FF0000'
    c5 = '#9400cc'

    plt.subplots_adjust(hspace=0.27, wspace=0.03)


    ax3 = plt.subplot2grid((5, 4), (0, 2), rowspan=3, colspan=2)

    ax3.bar(x, fp64_ratio, bar_width, color=c1, label='FP64', edgecolor='black')
    ax3.bar(x, fp32_ratio, bar_width, bottom=fp64_ratio,
            color=c2, label='FP32', edgecolor='black')
    ax3.bar(x, fp16_ratio, bar_width, bottom=[sum(x) for x in zip(fp64_ratio, fp32_ratio)],
            color=c3, label='FP16', edgecolor='black')
    ax3.bar(x, fp8_ratio, bar_width, bottom=[sum(x) for x in zip(fp64_ratio, fp32_ratio, fp16_ratio)],
            color=c4, label='FP8', edgecolor='black')
    ax3.bar(x, bypass_ratio, bar_width, bottom=[sum(x) for x in zip(fp64_ratio, fp32_ratio, fp16_ratio, fp8_ratio)],
            color=c5, label='Bypass', edgecolor='black')
    x1 = [0] * len(name_new)
    for i in range(len(name_new)):
        x1[i] = i
    ax3 = plt.gca()

    ax3.set_ylim(0, 1)
    labels_y = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
    ax3.set_xticks([])
    ax3.set_yticks(range(0, 105, 10), labels=[], fontsize=25)

    df_ratio_mix_bicg = pd.read_csv(
        "test2.csv",
        header=None,
        names=["sync", "mix", "up"])
    speedup_data = []
    for i in range(len(df_ratio_mix_bicg["up"])):
        speedup_data.append(df_ratio_mix_bicg["up"][i])
    x = np.arange(len(speedup_data))
    width = 0.1


    ax4 = plt.subplot2grid((5, 4), (3, 2), rowspan=2, colspan=2)

    ax4.bar(x, speedup_data, bar_width, color='#E3409A', label='', edgecolor='black', zorder=2)

    for i, val in enumerate(speedup_data):
        ax4.text(x[i], speedup_data[i] + 0.05, f'{val:.2f}', fontsize=20, ha='center', rotation=90, zorder=3,
                 fontweight='bold')

    ax4 = plt.gca()
    ax4.set_xticks(x)

    ax4.set_xticklabels(name_for_x_label1, rotation=270, fontsize=25)
    ax4.xaxis.labelpad = 5
    labels_y = ['1.0', '1.5', '1.8']
    plt.yticks((1.0, 1.5, 1.8), labels=[], fontsize=25)
    ax4.set_ylim(1, 1.8)
    ax4.set_xlabel('Matrices in BiCGSTAB', fontsize=25)
    plt.savefig("Fig 11.pdf", dpi=400, format="pdf", bbox_inches='tight', pad_inches=0.0)


def iteration():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    sns.set()
    sns.set_style("whitegrid")
    df_mesh2e1_initial1 = pd.read_csv(
        "poisson3Da_initial.csv",
        header=None,
        names=["iter", "norm"])
    df_mesh2e1_1e_131 = pd.read_csv(
        "poisson3Da_mix.csv",
        header=None,
        names=["iter", "norm"])

    plt.figure(figsize=(28, 10), dpi=400)
    plt.subplots_adjust(wspace=0.07, hspace=0.5)

    ax1 = plt.subplot(133)
    ax1.plot(df_mesh2e1_initial1['iter'], np.log10(df_mesh2e1_initial1['norm']), linestyle='-', label='FP64',
             linewidth=4)
    ax1.plot(df_mesh2e1_1e_131['iter'], np.log10(df_mesh2e1_1e_131['norm']), linestyle='-', label='Mixed Precision',
             linewidth=4)

    new_y_ticks = [r'', r'', r'', r'', r'', r'', r'']
    ax1.set_yticks([1, -1, -3, -5, -7, -9, -11], new_y_ticks, fontsize=45)

    ax1.set_xlabel('Iteration', fontsize=70)
    ax1.set_xlim(0, len(df_mesh2e1_1e_131['iter']))
    ax1.set_xticklabels(np.arange(0, len(df_mesh2e1_1e_131['iter']), 50), fontsize=45)
    ax1.set_title("poisson3Da", fontsize=70, pad=15)
    plt.grid(True)
    ax1.legend(prop={'size': 33}, loc="lower left", bbox_to_anchor=(-0.03, -0.03, 0.5, 0.3))
    df_mesh2e1_initial2 = pd.read_csv(
        "m3plates_initial.csv",
        header=None,
        names=["iter", "norm"])
    df_mesh2e1_1e_132 = pd.read_csv(
        "m3plates_mix.csv",
        header=None,
        names=["iter", "norm"])

    ax2 = plt.subplot(132)
    ax2.plot(df_mesh2e1_initial2['iter'], np.log10(df_mesh2e1_initial2['norm']), linestyle='-', label='FP64',
             linewidth=4)
    ax2.plot(df_mesh2e1_1e_132['iter'], np.log10(df_mesh2e1_1e_132['norm']), linestyle='-', label='Mixed Precision',
             linewidth=4)

    new_y_ticks = [r'', r'', r'', r'', r'', r'', r'']
    ax2.set_yticks([1, -1, -3, -5, -7, -9, -11], new_y_ticks, fontsize=45)
    ax2.set_xlabel('Iteration', fontsize=70)
    ax2.set_xlim(0, 45)
    ax2.set_title("m3plates", fontsize=70, pad=15)
    ax2.set_xticklabels(np.arange(0, len(df_mesh2e1_1e_132['iter']), 5), fontsize=45)
    plt.grid(True)

    ax2.legend(prop={'size': 33}, loc="lower left", bbox_to_anchor=(-0.03, -0.03, 0.5, 0.3))

    df_mesh2e1_initial3 = pd.read_csv(
        "min.csv",
        header=None,
        names=["iter", "norm"])
    df_mesh2e1_1e_133 = pd.read_csv(
        "minAP.csv",
        header=None,
        names=["iter", "norm"])

    ax3 = plt.subplot(131)
    ax3.plot(df_mesh2e1_initial3['iter'], np.log10(df_mesh2e1_initial3['norm']), linestyle='-', label='FP64',
             linewidth=4)
    ax3.plot(df_mesh2e1_1e_133['iter'], np.log10(df_mesh2e1_1e_133['norm']), linestyle='-', label='Mixed Precision',
             linewidth=4)


    new_y_ticks = [r'$10^{1}$', r'$10^{-1}$', r'$10^{-3}$', r'$10^{-5}$', r'$10^{-7}$', r'$10^{-9}$', r'$10^{-11}$']
    ax3.set_yticks([1, -1, -3, -5, -7, -9, -11], new_y_ticks, fontsize=45)

    ax3.set_xlabel('Iteration', fontsize=70)
    ax3.set_ylabel('Relative error', fontsize=70)
    ax3.set_xticklabels(np.arange(0, 110, 20), fontsize=45)
    ax3.set_title("minsurfo", fontsize=70, pad=15)
    plt.grid(True)
    ax3.legend(prop={'size': 33}, loc="lower left", bbox_to_anchor=(-0.03, -0.03, 0.5, 0.3))
    plt.savefig("Fig 10.pdf", dpi=400, format="pdf", bbox_inches='tight', pad_inches=0.02)
    
syncfree()
iteration()