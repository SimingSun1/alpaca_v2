import numpy as np
from matplotlib import rc
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)
import matplotlib.pyplot as plt
import time

    
def sinusoid_plot(freq,phase,amp,x_list,sigma_list,y_list,X_update, Y_update,sampling_density=101,legend_labels=['Ours', 'True']):
    """
    x,y,sigma should be lists
    """

    #plot given data
    conf_list = [1.96*np.sqrt(s) for s in sigma_list]
    upper = [y + c for y,c in zip(y_list,conf_list)]
    lower = [y - c for y,c in zip(y_list,conf_list)]
    plt.fill_between(x_list, upper, lower, alpha=.5)
    plt.plot(x_list,y_list)
    
    
    
    #plot true sinusoid
    yr_list = [amp*np.sin(freq*x + phase) for x in x_list]
    plt.plot(x_list,yr_list,color='r')
    
    # plot update points
    plt.plot(X_update[0,:,0],Y_update[0,:,0],'+',color='k',markersize=10)
    plt.xlim([np.min(x_list), np.max(x_list)])
    # set by ssm
    plt.ylim(-6,6)
    #legend
    if legend_labels:
        plt.legend(legend_labels + ['sampled points'])
    
def gen_sin_fig(agent, X,Y,freq,phase,amp,upper_x=5,lower_x=-5,point_every=0.1, label=None):
    y_list = []
    x_list = []
    s_list = []
    for p in np.arange(lower_x,upper_x,0.1):
        y, s = agent.test(X, Y, np.array([[[p]]]))
        y_list.append(y[0,0,0])
        x_list.append(p)
        if s:
            s_list.append(s[0,0,0,0])
        else:
            s_list.append(0)
    legend_labels = None
    if label:
        legend_labels = [label, 'True']
    sinusoid_plot(freq,phase,amp,x_list,s_list,y_list,X,Y, legend_labels=legend_labels)

