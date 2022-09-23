

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager


if __name__ == '__main__':
    result_name = "flip_contrast_only_control.csv"
    
    df = pd.read_csv('results/'+result_name)
    
    diff_loss = df['diffhand_loss'].values
    mod_loss = df['model_loss'].values
    
    TASK_NUMS = 100
    model_name = "ours"

    
    good_d = []
    good_m = []
    bad_d = []
    bad_m = []
    
    good_d_task = []
    good_m_task = []
    bad_d_task = []
    bad_m_task = []
    
    
    for diff,mod in zip(diff_loss,mod_loss):
        if mod<diff:
            good_d.append(diff)
            good_m.append(mod)
        else:
            bad_d.append(diff)
            bad_m.append(mod)

    plt.title('Flip Box')  
    plt.plot([0,8000],[0,8000])
    plt.scatter(good_d,good_m,label="good")
    plt.scatter(bad_d,bad_m,label="bad")
    plt.legend(loc = 'best',\
               labels = ['Loss_D = Loss_T','Loss_T is smaller','Loss_D is smaller'])
    plt.xlabel("Loss of Diffhand (Loss_D)") 
    plt.ylabel("Loss of Task2Mroph (Loss_T)")
    plt.savefig('flip_only_contrast.jpg',dpi=300)
    plt.show()    
      


# =============================================================================
# import os
# import sys
# 
# example_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# sys.path.append(example_base_dir)
# 
# from parameterization_torch import Design as Design
# from parameterization import Design as Design_np
# 
# from renderer import SimRenderer
# import numpy as np
# import scipy.optimize
# import redmax_py as redmax
# import os
# import argparse
# import time
# from common import *
# import torch
# import matplotlib.pyplot as plt
# import datetime
# from modify_xml import modify_flip_box_pos,modify_flip_box_size
# import random
# 
# from flip_constrast import diffhand
# from flip_use_model import use_model
# 
# torch.set_default_dtype(torch.double)
# 
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('')
#     parser.add_argument("--model", type = str, default = 'flip_box')
#     parser.add_argument('--record', action = 'store_true')
#     parser.add_argument('--record-file-name', type = str, default = 'flip_box')
#     parser.add_argument('--seed', type=int, default = 0)
#     parser.add_argument('--save-dir', type=str, default = './results/tmp/')
#     parser.add_argument('--no-design-optim', action='store_true', help = 'whether control-only')
#     parser.add_argument('--visualize', type=str, default='False', help = 'whether visualize the simulation')
#     parser.add_argument('--load-dir', type = str, default = None, help = 'load optimized parameters')
#     parser.add_argument('--verbose', default = False, action = 'store_true', help = 'verbose output')
#     parser.add_argument('--test-derivatives', default = False, action = 'store_true')
#     
#     asset_folder = os.path.abspath(os.path.join(example_base_dir, '../../', 'assets'))
#     args = parser.parse_args()
#     
#     model_name = "choose_good"
#     model_route = "../model/"
#     if model_name == "choose_good":
#         # model_route += "flip_model_parameter_reference_1.pkl"
#         # model_route += "flip_model_parameter_reference_2.pkl"
#         model_route += "flip_model_parameter_reference_3.pkl"
# 
#     TASK_NUMS = 100
#     only_control = True
#     print("only_control",only_control)
#     
#     good_d = []
#     good_m = []
#     bad_d = []
#     bad_m = []
#     
#     good_d_task = []
#     good_m_task = []
#     bad_d_task = []
#     bad_m_task = []
#     
#     
#     result_data = []
#     
#     for i in range(TASK_NUMS):
#         print(" ============= task= "+str(i)+" ==================")
#         if i!=0:
#             print("Good loss :",len(good_d)," Bad loss :",len(bad_d)," : %.3f" % (len(good_d)/i))
#             print("Good Task :",len(good_d_task)," Bad Task :",len(bad_d_task)," : %.3f" % (len(good_d_task)/i))
#             
#         taski_result = []
#         
#         box_x = random.uniform(-27, 27)
#         box_y = random.uniform(-4,4)    
#         box_x = round(box_x,3)
#         box_y = round(box_y,3)
#         new_str_pos = str(box_x)+" "+str(box_y)+" -10"
#         print("new_str_pos:",new_str_pos)
#     
#         # change box size
#         size_a = round(random.uniform(2, 9),3)
#         size_b = round(random.uniform(2, 9),3)
#         size_c = round(random.uniform(2, 9),3)
#         print("new_size:",size_a,size_b,size_c)
#         
#         
#         taski_result.extend([box_x,box_y,size_a,size_b,size_c])
#         
#         # diffhand_result
#         
#         d_loss,d_flip,d_times,d_iteri = diffhand(i,box_x,box_y,size_a,size_b,size_c,only_control)
#         print("diffhand:",d_loss,d_flip)
#         taski_result.append(d_loss)
#         taski_result.append(d_flip)
#         taski_result.append(d_times)
#         taski_result.append(d_iteri)
#         
#             
#         
#         loss,flip,times,iteri = use_model(model_route,model_name,i,box_x,box_y,size_a,size_b,size_c,only_control)
#         print(model_name+":",loss,flip)
#         taski_result.append(loss)
#         taski_result.append(flip)
#         taski_result.append(times)
#         taski_result.append(iteri)
#         
#         if loss<d_loss:
#             good_d.append(d_loss)
#             good_m.append(loss)
#         else:
#             bad_d.append(d_loss)
#             bad_m.append(loss)
#             
#         if flip<d_flip:
#             good_d_task.append(d_flip)
#             good_m_task.append(flip)
#         else:
#             bad_d_task.append(d_flip)
#             bad_m_task.append(flip)
#         
#         result_data.append(taski_result)
#     
#         
#     import pandas as pd
#     columns=['box_x','box_y','size_a','size_b','size_c','diffhand_loss','diffhand_flip','diffhand_t','diffhand_i'\
#              ,'model_loss','model_flip','model_t','model_i']
#     result_data_pd = pd.DataFrame(result_data)
#     result_data_pd.columns=columns
#     result_name = 'flip_contrast_'+str(TASK_NUMS)+"_mod="+str(model_name)+'.csv'
#     result_data_pd.to_csv('results/'+result_name,header=True,index=True)
# 
# 
# =============================================================================
















    
    
    


