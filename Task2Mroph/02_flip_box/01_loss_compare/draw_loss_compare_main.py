import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




def smooth2nd(x,M): 
    K = round(M/2-0.1) 
    lenX = len(x)
    if lenX<2*K+1:
        print('The data length is less than the number of smooth points')
    else:
        y = np.zeros(lenX)
        for NN in range(0,lenX,1):
            startInd = max([0,NN-K])
            endInd = min(NN+K+1,lenX)
            y[NN] = np.mean(x[startInd:endInd])
    y[0]=x[0]       
    y[-1]=x[-1]   
    return(y)

if __name__ == '__main__':

    # path="result_new/"
    path="#result_-27_27_-4_4_2_9_seed=0-19_reference1#/"  
    
    # smooth=5
    M = 5
    line_len = 4
    iter_num = 150
    our = "choose_good"
    N = 20 
    
    var_n = 10
    
    filename_ls = os.listdir(path)
    loss_diff=[]
    loss_our=[]
    loss_diff_c=[]
    loss_our_c=[]
    
    for file in filename_ls:
        data = pd.read_csv(path + file,index_col=0)
        smoth_data = smooth2nd(data['Loss'].tolist(),M)
        
        if "diffhand" not in file and "control" not in file:
            print(file)
            loss_our.append(smoth_data)
        elif "diffhand" not in file and "control" in file:
            loss_our_c.append(smoth_data)
            
        elif "diffhand" in file and "control" not in file:
            loss_diff.append(smoth_data)
        elif "diffhand" in file and "control" in file:
            loss_diff_c.append(smoth_data)   
    
    all_loss = [loss_our,loss_diff,loss_our_c,loss_diff_c]   
    all_mean=[]
    all_var=[]
    
    for ls in all_loss:
        loss_mean=[]
        loss_var=[]
        for i in range(iter_num):
            loss_cur = []
            for j in range(N):
                if i < len(ls[j]):
                    loss_cur.append(ls[j][i])
                else:
                    loss_cur.append(ls[j][-1])
            loss_mean.append(sum(loss_cur)/len(loss_cur))
            loss_var.append(np.std(loss_cur,ddof=1)/var_n)
            
        all_mean.append(loss_mean)
        all_var.append(loss_var)
            
    plt.title('Flip Box')  
    
    iters=np.array(range(0,iter_num))
    labels_ls = ['Task2Morph ','DiffHand ','Task2Morph (Control Only)','DiffHand (Control Only)']
    colors = ['#D62627','#1D76B3','#FF8113','#2A9F2A']
    
    for i in range(line_len):
        plt.plot(iters, all_mean[i], zorder=9,c = colors[i],linewidth=1.5,label = labels_ls[i])  # 实线
        np1=np.array(all_mean[i])
        np2=np.array(all_var[i])
        plt.fill_between(iters, np1 + np2, np1 - np2,color=colors[i],alpha=0.3,zorder=3)
    plt.legend()  
    plt.xlabel("Episode") 
    plt.ylabel("Loss")
    plt.savefig("pic/flip_loss_cmopare.jpg", dpi=300)
    plt.show()
    
    

    
    
    
    
