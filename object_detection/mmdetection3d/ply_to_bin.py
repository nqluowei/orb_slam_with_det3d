# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
import os.path

import numpy as np
import pandas as pd
from plyfile import PlyData
 
 
def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # read file
    data = plydata.elements[0].data  # read data
    data_pd = pd.DataFrame(data)  # convert to DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # initialize array to store data
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(
            property_names):  # read data by property
        data_np[:, i] = data_pd[name]
        
    print("data_np=",data_np.shape)
    data_np.astype(np.float32).tofile(output_path)
    
 
 
if __name__ == '__main__':
    for i in range(0,2000):

        ply_file = '/home/luowei/ai/object_detection/multi_view_3d_detection/output/%d.ply'%(i)
        bin_file = '/home/luowei/ai/object_detection/multi_view_3d_detection/output/%d.bin'%(i)

        if not os.path.exists(ply_file):
            continue

        print(i, end=" ")
        convert_ply(ply_file, bin_file)
