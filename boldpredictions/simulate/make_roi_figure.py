import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def make_roi_figure(roi_list, results, random_results, filename):

    roi_names = ['{0} [{1},{2},{3}]'.format(c.name,c.x,c.y,c.z) for c in roi_list]
    subject_names = results.keys()

    ROIs = dict([(subject_name,
                [results[subject_name][idx_r]["mean"]*100
                 for idx_r, roi_name in enumerate(roi_names)])
                 for subject_name in subject_names])


    fig, axs = plt.subplots(9,1, figsize=(6,13),sharex=True,sharey=False)

    sns.set(font_scale=1.2)

    colors = sns.hls_palette(6, l=.4, s=.8)+sns.hls_palette(6, l=.4, s=.8)+\
             sns.hls_palette(6, l=.4, s=.8)+sns.hls_palette(4, l=.4, s=.8)
    markers = ['ro','ro','ro','ro','ro','ro','rd','rd','rd',
               'rd','rd','rd','r^','r^','r^','r^','rd','rd',
               'rd','rd',]

    subject_names_legend = ['subject 1', 'subject 2',
                            'subject 3', 'subject 4', 'subject 5',
                            'subject 6', 'subject 7','subject 8',
                            'MNI combination']

    lines = [""]*len(roi_names)
    print(random_results[subject_names[0]].shape)
    for idx,subject_name in enumerate(subject_names):

        tmp = random_results[subject_name][:,1]*100
        tmp[-1] = 120
        tmp[-2] = -1
        #h = sns.kdeplot(tmp, shade=True, ax = axs[idx],color='olivedrab')
        h = sns.distplot(tmp, bins  =25, kde = True,hist=True, rug=False,norm_hist=False, ax=axs[idx],hist_kws=dict(alpha=0),
                        kde_kws = dict(bw = 0.5, shade = True, color = 'olivedrab'))

        for idx_r,roi_name in enumerate(roi_names):
            markerline, stemlines, baseline = axs[idx].stem([ROIs[subject_name][idx_r]],[0.0], #0.04
                                                        linefmt='r-', markerfmt=markers[idx_r], basefmt=" ",)
            #rname = roi_name
            stemlines[0].set(color=colors[idx_r],label = roi_name)
            markerline.set(markersize=15,markerfacecolor=colors[idx_r],markeredgecolor = 'white',
                          markeredgewidth = 1)
            lines[idx_r] = markerline
        axs[idx].set_xlim([0,70])
        current_ylim = axs[idx].get_ylim()[1]
        axs[idx].set_ylim([0-current_ylim/6, current_ylim])
        axs[idx].set_yticks([0,current_ylim/2] )
        ps = [90,95,99]
        for p in ps:
            x_p = np.percentile(tmp, p)
            axs[idx].plot([x_p,x_p],axs[idx].get_ylim(),'--',linewidth = 1, color = 'gray')
        axs[idx].set_ylabel('pdf')
        axs[idx].set_title(subject_names_legend[idx])
        axs[idx].set_xlabel('overlap (percentage of voxels)')
        axs[idx].set_xlim([0,axs[idx].get_xlim()[1]])
    fig.tight_layout()

    roi_names2 = ['density of randomly\nsampled ROIs']
    for roi_name in roi_names:
        if 1:
            a = roi_name.split(' ')
            cnt = 0
            tmp1 = '\n'
            for i in a:
                cnt = cnt+1
                if cnt%4 == 0 or i[0] == '[' or i[0] == '(':
                    tmp1 +='\n'
                else:
                    tmp1 +=' '
                if i =='g':
                    i = 'gyrus'
                tmp1 += i
        roi_names2.append(tmp1)
    lines = [h.get_lines()[0]] + lines
    fig.legend(lines,roi_names2, loc='upper left',fontsize = 13,ncol = 1,bbox_to_anchor=(0.98, 0.95))
    plt.savefig(filename, bbox_inches='tight')
    return fig