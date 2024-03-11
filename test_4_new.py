import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import time

time_start = time.time()
N=200
m_range = np.arange(2,9)
num_k = 20
H_numerical_1_range = np.zeros((num_k,np.size(m_range)))
H_num_intra_range = np.zeros((num_k,np.size(m_range)))
H_num_inter_range = np.zeros((num_k,np.size(m_range)))
lambda_2_Js_Pr_range = np.zeros(np.size(m_range))
N_2 = int(N/2)
A_matrix = np.zeros((N,N))
A_matrix[0:N_2,0:N_2] = np.eye(N_2,k=1)+np.eye(N_2,k=2)+np.eye(N_2,k=N_2-2)
A_matrix[0,N_2-1] = 1.0
A_matrix[0:N_2,N_2:N] = np.fliplr(np.eye(N_2))
A_matrix[N_2:N,N_2:N] = A_matrix[0:N_2,0:N_2]
A_matrix += A_matrix.T-np.diag(A_matrix.diagonal())
degree_range = np.zeros((N,np.size(m_range)))
degree_range[:,0] = np.sum(A_matrix,axis=1)

A_matrix_range = np.zeros((N,N,np.size(m_range)))
A_matrix_range[:,:,0] = A_matrix
delta_matrix_range = np.zeros((N,N,np.size(m_range)-1))
delta_matrix_range_2 = np.zeros((N,N,np.size(m_range)-1))
F_norm_range = np.zeros(np.size(m_range)-1)
F_norm_range_2 = np.zeros(np.size(m_range)-1)

data_path = 'F:/python_work_PyCharm/work_2_2019_22/'
for m in m_range:

    H_numerical_1_range[:,m-2] = np.load(data_path+"H_num_1_"+str(m)+"_clusters_N_200.npy")
    H_num_intra_range[:,m-2] = np.load(data_path+"H_num_intra_"+str(m)+"_clusters_N_200.npy")
    H_num_inter_range[:,m-2] = np.load(data_path+"H_num_inter_"+str(m)+"_clusters_N_200.npy")
    lambda_2_Js_Pr_range[m-2] = np.real(np.load(data_path+"lambda_2_Js_Pr_"+str(m)+"_clusters_N_200.npy"))

    if m>2:
        A_matrix_range[:,:,m-2] = np.load(data_path+"A_matrix_modify_"+str(m)+"_clusters_N_200.npy")
        degree_range[:,m-2] = np.sum(A_matrix_range[:,:,m-2],axis=1)
        delta_matrix_range[:,:,m-3] = A_matrix_range[:,:,m-2]-A_matrix
        delta_matrix_range_2[:,:,m-3] = np.load(data_path+"A_matrix_modify_"+str(m-1)+"_to_"+str(m)+"_clusters_N_200.npy.npy") - A_matrix
        F_norm_range[m-3] = np.sum(np.power(delta_matrix_range[:,:,m-3],2),axis=(0,1))
        F_norm_range_2[m-3] = np.sum(np.power(delta_matrix_range_2[:,:,m-3],2),axis=(0,1))

H_numerical_1_mean = np.mean(H_numerical_1_range,axis=0)
H_num_intra_mean = np.mean(H_num_intra_range,axis=0)
H_num_inter_mean = np.mean(H_num_inter_range,axis=0)
H_numerical_1_std = np.std(H_numerical_1_range,axis=0)
H_num_intra_std = np.std(H_num_intra_range,axis=0)
H_num_inter_std = np.std(H_num_inter_range,axis=0)

time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")

print(np.shape(degree_range))
plt.figure(1,figsize=[6.4,4.8])
plt.plot(m_range,np.std(degree_range,axis=0),'-o', markersize=None, linewidth=2, color="red",alpha=0.8)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r"$m$", fontsize=20)
plt.ylabel(r"$\sigma$", fontsize=20)
#plt.savefig('F:\Latex_work\work_3_2019_9_18\Figures\Fig_degree_std_clusters_N_200.pdf')


plt.figure(2,figsize=[6.4,4.8])
plt.plot(m_range,H_numerical_1_mean,'-o', markersize=None, linewidth=2, color="red",alpha=0.8)
plt.fill_between(m_range, H_numerical_1_mean-H_numerical_1_std, H_numerical_1_mean+H_numerical_1_std,color='red',linewidth=2,alpha=0.2)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r"$m$",fontsize=20)
plt.ylabel(r"$H$",fontsize=20)
plt.savefig('F:\Latex_work\work_3_2019_9_18\Figures\Fig_H_m_variations_200_clusters.pdf')

# plt.figure(3,figsize=[6.4*1.7,4.8*1.6])
plt.figure(3,figsize=[6.4,4.8])
plt.plot(m_range,H_num_intra_mean,'-o', markersize=None, linewidth=2, color="red", alpha=0.8,label=r'$H_{\mathrm{intra}}$')
plt.fill_between(m_range, H_num_intra_mean-H_num_intra_std, H_num_intra_mean+H_num_intra_std,color='red',linewidth=2,alpha=0.2)
plt.plot(m_range,H_num_inter_mean,'-o', markersize=None, linewidth=2, color="blue", alpha=0.8,label=r'$H_{\mathrm{inter}}$')
plt.fill_between(m_range, H_num_inter_mean-H_num_inter_std, H_num_inter_mean+H_num_inter_std,color='blue',linewidth=2,alpha=0.2)
plt.legend(loc='upper left',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r"$m$",fontsize=20)
#plt.ylabel(r"$H_{intra}$",fontsize=20)
plt.savefig('F:\Latex_work\work_3_2019_9_18\Figures\Fig_H_intra_m_variations_200_clusters.pdf')


# plt.figure(4,figsize=[6.4*1.9,4.8*1.6])
plt.figure(4,figsize=[6.4,4.8])
plt.plot(m_range,lambda_2_Js_Pr_range,'-o', markersize=None, linewidth=2, color="red",alpha=0.8)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r"$m$",fontsize=20)
plt.ylabel(r"$\lambda_2$",fontsize=20)
plt.savefig('F:\Latex_work\work_3_2019_9_18\Figures\Fig_lambda_2_Js_Pr_m_variations_200_clusters.pdf')

plt.figure(5,figsize=[6.4,4.8])
plt.plot(m_range[1:],F_norm_range,'-o', markersize=None, linewidth=2, color="red", alpha=0.8)
#plt.plot(m_range[1:],F_norm_range_2,'-o',color="blue",linewidth=2, alpha=0.8)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r"$m$",fontsize=20)
plt.ylabel(r"$\Vert\bar\Delta^*\Vert_F^2$",fontsize=20)
plt.savefig('F:\Latex_work\work_3_2019_9_18\Figures\Fig_F_norm_m_variations_200_clusters.pdf')

plt.show()
print()


