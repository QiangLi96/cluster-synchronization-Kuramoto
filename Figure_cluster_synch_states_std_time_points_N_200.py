import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import matplotlib.ticker as ticker
import networkx as nx
import math
import time


time_start = time.time()

def fun_dotx(theta_temp,A_matrix,omega_array):
    k_temp = omega_array+5.0*np.sum(np.multiply(A_matrix,np.sin(theta_temp.T-theta_temp)),axis=1,keepdims=True)
    return k_temp

N=200
m_range = np.arange(2,9)

"""
#create a graph is consisted of 2 layers ring-like graph, A_matrix is the adjacency matrix of the graph
A_matrix = np.zeros((N,N))
A_matrix[0:int(N/2),0:int(N/2)] = np.eye(int(N/2),k=1)+np.eye(int(N/2),k=2)+np.eye(int(N/2),k=int(N/2)-2)
A_matrix[0,int(N/2)-1] = 1.0
A_matrix[0:int(N/2),int(N/2):N] = np.fliplr(np.eye(int(N/2)))
A_matrix[int(N/2):N,int(N/2):N] = A_matrix[0:int(N/2),0:int(N/2)]
A_matrix += A_matrix.T-np.diag(A_matrix.diagonal())
"""
omega_array = np.vstack((np.ones((int(N/2),1))*(-1.0), np.ones((int(N/2),1))*1.0))

theta_star_std = np.zeros(np.size(m_range))
theta_time_star = np.zeros(np.size(m_range))


#A_matrix = np.load("/home1/LiQiang/python_work/work_2_2020_2_15/A_matrix_modify_"+str(m)+"_clusters_N_200.npy")


step_size=0.002
Total_step = 20000

for m in m_range:
    if m==2:
        #create a graph is consisted of 2 layers ring-like graph, A_matrix is the adjacency matrix of the graph
        A_matrix = np.zeros((N,N))
        A_matrix[0:int(N/2),0:int(N/2)] = np.eye(int(N/2),k=1)+np.eye(int(N/2),k=2)+np.eye(int(N/2),k=int(N/2)-2)
        A_matrix[0,int(N/2)-1] = 1.0
        A_matrix[0:int(N/2),int(N/2):N] = np.fliplr(np.eye(int(N/2)))
        A_matrix[int(N/2):N,int(N/2):N] = A_matrix[0:int(N/2),0:int(N/2)]
        A_matrix += A_matrix.T-np.diag(A_matrix.diagonal())
        theta_matrix = np.zeros((N, Total_step))
        np.random.seed(0)
        theta_matrix[:, 0] = np.random.uniform(-0.1, 0.1, N)

        for i in range(0, Total_step - 1, 1):
            if i == 0:
                k1 = step_size * fun_dotx(theta_matrix[:, [i]], A_matrix, omega_array)
                k2 = step_size * fun_dotx(theta_matrix[:, [i]] + k1 / 2.0, A_matrix, omega_array)
                k3 = step_size * fun_dotx(theta_matrix[:, [i]] + k2 / 2.0, A_matrix, omega_array)
                k4 = step_size * fun_dotx(theta_matrix[:, [i]] + k3, A_matrix, omega_array)
                theta_matrix[:, [i + 1]] = theta_matrix[:, [i]] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                omega_max = np.max(np.abs(theta_matrix[:, [i + 1]] - theta_matrix[:, [i]])) / step_size
            else:
                k1 = step_size * fun_dotx(theta_matrix[:, [i]], A_matrix, omega_array)
                k2 = step_size * fun_dotx(theta_matrix[:, [i]] + k1 / 2.0, A_matrix, omega_array)
                k3 = step_size * fun_dotx(theta_matrix[:, [i]] + k2 / 2.0, A_matrix, omega_array)
                k4 = step_size * fun_dotx(theta_matrix[:, [i]] + k3, A_matrix, omega_array)
                theta_matrix[:, [i + 1]] = theta_matrix[:, [i]] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                omega_max_temp = omega_max
                omega_max = np.max(np.abs(theta_matrix[:, [i + 1]] - theta_matrix[:, [i]])) / step_size
                if np.abs(omega_max - omega_max_temp) < 10**-8:
                    print(i + 1, np.abs(omega_max - omega_max_temp))
                    break

        print(m)
        theta_star = theta_matrix[:, i + 1]
        theta_star_std[m-2] = np.std(theta_star)
        theta_time_star[m-2] = (i+1)*step_size
    else:
        A_matrix = np.load("A_matrix_modify_" + str(m) + "_clusters_N_200.npy")
        theta_matrix = np.zeros((N,Total_step))
        np.random.seed(0)
        theta_matrix[:,0] = np.random.uniform(-0.1,0.1,N)

        for i in range(0,Total_step-1,1):
            if i == 0:
                k1 = step_size*fun_dotx(theta_matrix[:,[i]],A_matrix,omega_array)
                k2 = step_size*fun_dotx(theta_matrix[:,[i]]+k1/2.0,A_matrix,omega_array)
                k3 = step_size * fun_dotx(theta_matrix[:,[i]] + k2 / 2.0, A_matrix,omega_array)
                k4 = step_size * fun_dotx(theta_matrix[:,[i]]+k3, A_matrix,omega_array)
                theta_matrix[:,[i+1]] = theta_matrix[:,[i]]+(k1+2.0*k2+2.0*k3+k4)/6.0
                omega_max = np.max(np.abs(theta_matrix[:,[i+1]]-theta_matrix[:,[i]]))/step_size
            else:
                k1 = step_size*fun_dotx(theta_matrix[:,[i]],A_matrix,omega_array)
                k2 = step_size*fun_dotx(theta_matrix[:,[i]]+k1/2.0,A_matrix,omega_array)
                k3 = step_size * fun_dotx(theta_matrix[:,[i]] + k2 / 2.0, A_matrix,omega_array)
                k4 = step_size * fun_dotx(theta_matrix[:,[i]]+k3, A_matrix,omega_array)
                theta_matrix[:,[i+1]] = theta_matrix[:,[i]]+(k1+2.0*k2+2.0*k3+k4)/6.0
                omega_max_temp = omega_max
                omega_max = np.max(np.abs(theta_matrix[:,[i+1]]-theta_matrix[:,[i]]))/step_size
                if np.abs(omega_max-omega_max_temp)<10**-8:
                    print(i + 1, np.abs(omega_max - omega_max_temp))
                    break

        print(m)
        theta_star = theta_matrix[:,i+1]
        theta_star_std[m-2] = np.std(theta_star)
        theta_time_star[m-2] = (i+1)*step_size

time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")



plt.figure(1, figsize=[6.4*1.7,4.8*1.6])
plt.plot(m_range,theta_star_std, '-o', markersize=12, linewidth=4, color='red', alpha=0.8)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$m$",fontsize=35)
plt.ylabel(r"$\sigma_{\theta^*}$",fontsize=35)
plt.savefig('F:\Latex_work\work_3_2019_9_18\Figures\Fig_sigma_theta_star_N_200.pdf')


plt.figure(2, figsize=[6.4*1.7,4.8*1.6])
plt.plot(m_range,theta_time_star, '-o', markersize=12, linewidth=4, color='red', alpha=0.8)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$m$",fontsize=35)
plt.ylabel(r"$t^*$",fontsize=35)
plt.savefig('F:\Latex_work\work_3_2019_9_18\Figures\Fig_theta_time_star_N_200.pdf')

plt.show()

