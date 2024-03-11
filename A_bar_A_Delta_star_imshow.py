import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import time

time_start = time.time()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im,fraction=0.046, pad=0.04, ax=ax,**cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=0,fontsize=25, va="bottom")
    cbar.ax.tick_params(labelsize=25)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels,fontsize=25)
    ax.set_yticklabels(row_labels,fontsize=25)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            if data[i,j] != 0.0:
                text = im.axes.text(j, i, valfmt(data[i, j], None),fontsize=18, **kw)
                texts.append(text)

    return texts

def fun_dotx(theta_temp,A_matrix,omega_array):
    k_temp = omega_array+np.sum(np.multiply(A_matrix,np.sin(theta_temp.T-theta_temp)),axis=1)
    return k_temp

N=9
A_matrix=np.array([[0,1,1,0,0,1,0,0,0],[1,0,1,0,1,0,0,0,0],[1,1,0,1,0,0,0,0,0],[0,0,1,0,1,0,0,1,1],[0,1,0,1,0,1,1,0,1],
                   [1,0,0,0,1,0,1,1,0],[0,0,0,0,1,1,0,1,0],[0,0,0,1,0,1,1,0,1],[0,0,0,1,1,0,0,1,0]])*1.0
omega_array = np.array([[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[2.0],[2.0],[2.0]])*1.0
m=4
B_intra = np.matrix([[-1,0,0,0,0],[1,0,0,0,0],[0,-1,0,0,0],[0,1,0,0,0],[0,0,-1,0,0],[0,0,1,0,0],[0,0,0,-1,0],[0,0,0,1,-1],[0,0,0,0,1]])*1.0
B_inter = np.zeros((N,m-1))*1.0
#B_inter[2,0] = 1.0; B_inter[3,0] = -1.0
V_P = np.zeros((N,m))*1.0
V_P[0:2,0] = 1.0; V_P[2:4,1] = 1.0
V_P[4:6,2] = 1.0; V_P[6:9,3] = 1.0
H = np.ones((N,N))
H = H - np.eye(N)
for k in range(0,8):
    H[k,k+1] = 0.0
    H[k+1,k] = 0.0
H_c = np.ones((N,N))*1.0-H
bar_A = A_matrix-np.multiply(A_matrix,np.dot(V_P,V_P.T))
Z_0 = bar_A
T_0 = 0.0
Q_0 = 0.0
#T_0 = np.zeros((N,N))*1.0
#Q_0 = np.zeros((N,N))*1.0
#ZT_sum = (Z_0+T_0)
ZT_sum = ((Z_0+T_0)+np.abs(Z_0+T_0))/2.0
Y_0 = np.multiply(H,ZT_sum)+np.multiply(H_c,bar_A)
T_1 = Z_0+T_0-Y_0
matrix_temp = np.dot(B_intra,np.dot(np.linalg.inv(np.dot(B_intra.T,B_intra)),np.dot(B_intra.T,np.dot(Y_0+Q_0,np.dot(V_P,np.dot(np.linalg.inv(np.dot(V_P.T,V_P)),V_P.T))))))
Z_1 = Y_0+Q_0-matrix_temp-np.sum(Y_0+Q_0-bar_A-2.0*matrix_temp)*np.ones((N,N))/(1.0*N**2)
Q_1 = Y_0+Q_0-Z_1
while np.max(np.abs(Z_1-Z_0))>10**(-10):
    Z_0 = Z_1
    T_0 = T_1
    Q_0 = Q_1
    ZT_sum = ((Z_0 + T_0) + np.abs(Z_0 + T_0)) / 2.0
    Y_0 = np.multiply(H, ZT_sum) + np.multiply(H_c, bar_A)
    T_1 = Z_0 + T_0 - Y_0
    matrix_temp = np.dot(B_intra,np.dot(np.linalg.inv(np.dot(B_intra.T,B_intra)),np.dot(B_intra.T,np.dot(Y_0+Q_0,np.dot(V_P,np.dot(np.linalg.inv(np.dot(V_P.T,V_P)),V_P.T))))))
    Z_1 = Y_0+Q_0-matrix_temp-np.sum(Y_0+Q_0-bar_A-2.0*matrix_temp)*np.ones((N,N))/(1.0*N**2)
    Q_1 = Y_0 + Q_0 - Z_1
    print(np.max(np.abs(Z_1-Z_0)))


Delta_star = Z_1-bar_A



fig, ax = plt.subplots()
row_label = [1,2,3,4,5,6,7,8,9]
col_label = [1,2,3,4,5,6,7,8,9]
A_matrix_4_cluster=A_matrix+Delta_star
A_matrix_4_cluster=np.around(A_matrix_4_cluster,decimals=2)
#Delta_star_0=np.around(Delta_star,decimals=2)
#matrix_show = Delta_star_0
matrix_show = A_matrix_4_cluster

im, cbar = heatmap(matrix_show, row_label, col_label, ax=ax,
                   cmap="YlGn",vmin=-1.0,vmax=1.2)
texts = annotate_heatmap(im, valfmt="{x:.2f} ")
#plt.colorbar(im,fraction=0.046, pad=0.04)
fig_0 = plt.gcf()
fig_0.set_size_inches(10.0,8.5)
fig.tight_layout()

#plt.savefig('F:\Latex_work\work_3_2019_9_18\Figures\Delta_star_3to4.pdf')
#plt.savefig('F:\Latex_work\work_3_2019_9_18\Figures\Adjacency_4_3to4.pdf')
plt.show()



time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")


