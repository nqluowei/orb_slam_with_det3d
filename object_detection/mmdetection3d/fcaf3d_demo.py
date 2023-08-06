# Copyright (c) OpenMMLab. All rights reserved.
import os.path
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model, show_result_meshlab


#              0       1        2       3        4        5       6              7            8            9
#cls_names = ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub']
cls_names = ['床',   '小桌',  '沙发',  '椅子  ',  '马桶',   '办公桌', '梳妆台',   '床头柜',       '书架',      '浴缸']
cls_colors= [(0,0,0),(0,1,0),(0,0,0),(1,0,0),(0,0,0),  (1,1,0), (0,0,0),     (0,0,1),      (0,0,0),    (0,0,0)]
from plyfile import PlyData
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as Rota

import torch
import numpy as np
# 关闭numpy中的科学计数输出
np.set_printoptions(precision=4, suppress=True)
# 关闭pytorch中的科学计数输出
torch.set_printoptions(precision=4, sci_mode=False)


def read_ply(input_path):
    plydata = PlyData.read(input_path)  # read file
    data = plydata.elements[0].data  # read data
    data_pd = pd.DataFrame(data)  # convert to DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float32)  # initialize array to store data
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(
            property_names):  # read data by property
        data_np[:, i] = data_pd[name]

    return data_np


args_checkpoint = "checkpoints/fcaf3d_8x2_sunrgbd-3d-10class_20220805_165017.pth"
args_config = "configs/fcaf3d/fcaf3d_8x2_sunrgbd-3d-10class.py"
args_device = "cuda:0"
args_pcd = ""
args_out_dir = "demo/outdir"
#args_score_thr = 0.1 在_base_配置文件里直接过滤掉
args_show = True
args_snapshot = True


def model_init():
    model = init_model(args_config, args_checkpoint, device=args_device)
    return model



def transform_box_sunrgbd_to_footprint(pred_bboxes,euler_rad=[0, 0, -np.pi/2],translate=[[0],[0],[0.326921]]):
    output_bboxes = []
    for x,y,z,dx,dy,dz,angle in pred_bboxes:
        center = np.array([[x],[y], [z]])
        r = Rota.from_euler('xyz', euler_rad, degrees=False)
        R = r.as_matrix()
        #x,y,z = center @ R + np.array(translate) #已经到地面上了
        x,y,z = R @ center + translate
        x=x[0]
        y=y[0]
        z=z[0]

        angle += euler_rad[2]

        output_bboxes.append([x,y,z,dx,dy,dz,angle])

    return output_bboxes


def infer(model,pcd, data_np,SHOW = False, transform = True): #默认要做变换到地面上

    if SHOW:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1400, height=1000, left=0, top=0)

        vis.clear_geometries()  # 清除所有物体


    result, data = inference_detector(model,args_pcd,data_np) #args_pcd已无效，因为从第三个变量加载数据

    # show_result_meshlab(
    #     data,
    #     result,
    #     args_out_dir,
    #     args_score_thr,
    #     show=args_show,
    #     snapshot=args_snapshot,
    #     task='det')

    pred_bboxes = result[0]['boxes_3d'].tensor.cpu().numpy()
    pred_scores = result[0]['scores_3d'].cpu().numpy()
    pred_labels = result[0]['labels_3d'].cpu().numpy()  # luowei add

    pred_pnums = []
    for x, y, z, dx, dy, dz, angle in pred_bboxes:
        z += dz / 2  # 移动到真的中心
        R_box = pcd.get_rotation_matrix_from_axis_angle(np.array([0, 0, angle]).T)
        boundingbox = o3d.geometry.OrientedBoundingBox(np.array([x, y, z]),
                                                       R_box,
                                                       np.array([dx, dy, dz])
                                                       )
        indices = boundingbox.get_point_indices_within_bounding_box(pcd.points)
        pred_pnums.append(len(indices))


    # pred_bboxes = [(1,2,-0.3,0.5,1,1,-np.pi/6)]
    # pred_scores = [0.2]
    # pred_labels = [1]

    if transform:
        pred_bboxes_transform = transform_box_sunrgbd_to_footprint(pred_bboxes) #转换到x向前，并且地面和坐标轴对齐
    else:
        pred_bboxes_transform = pred_bboxes

    if SHOW:
        coordinate0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=(0, 0, 0))

        vis.add_geometry(coordinate0)

        vis.add_geometry(pcd)

    out_res = []
    print("\n检测结果数目:",len(pred_bboxes_transform))
    for i,(x,y,z,dx,dy,dz,angle) in enumerate(pred_bboxes_transform): #这里输出的center_z表示的是底部中心的高度
        z += dz/2 #移动到真的中心

        score = pred_scores[i]
        label_id = pred_labels[i]
        pnums = pred_pnums[i]
        cls_name = cls_names[label_id]

        out_res.append( [x,y,z,dx,dy,dz,angle,label_id,score,pnums] )

        print("检测结果:%d  类别:%s 位置xyz:(%.2f %.2f %.2f) 尺寸dxyz:(%.2f %.2f %.2f) 朝向:%.2f°"%(i+1,cls_name,x,y,z,dx,dy,dz,np.degrees(angle)))

        if SHOW:
            #for i in range(5):
            if 1:
                r = Rota.from_euler('xyz', [0, 0, angle], degrees=False)
                R_box = r.as_matrix()
                #R_box = pcd.get_rotation_matrix_from_axis_angle(np.array([0, 0, angle]).T)  # 本身就要输入弧度
                boundingbox = o3d.geometry.OrientedBoundingBox(np.array([x, y, z]),
                                                               R_box,
                                                               np.array([dx, dy, dz])
                                                               )

                boundingbox.color = np.array(cls_colors[label_id])
                #boundingboxs.append(boundingbox)
                vis.add_geometry(boundingbox)
    
    print("")

    # o3d.visualization.draw_geometries([coordinate0, pcd]+boundingboxs, window_name="Open3D")

    if SHOW:
        para = vis.get_view_control().convert_to_pinhole_camera_parameters()
        #R = pcd.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0, 0]).T)
        r = Rota.from_euler('xyz', [np.pi, 0, 0], degrees=False)
        R = r.as_matrix()
        para.extrinsic = np.mat([
            [R[0, 0], R[0, 1], R[0, 2], 0],
            [R[1, 0], R[1, 1], R[1, 2], 0],
            [R[2, 0], R[2, 1], R[2, 2], 10],
            [0, 0, 0, 1]
        ])
        vis.get_view_control().convert_from_pinhole_camera_parameters(para)

        vis.poll_events()
        vis.update_renderer()

        vis.run()


    return pred_bboxes,np.array(out_res)

def main():
    index = 450 #120 #450
    input_file_name = "/home/luowei/ai/object_detection/multi_view_3d_detection/output/%d.ply" % (index)
    if not os.path.exists(input_file_name):
        return

    data_np = read_ply(input_file_name)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_np[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(data_np[:,3:]/255.0)


    model = model_init()
    #model = None
    infer(model,pcd,data_np,SHOW=True,transform=False)

#再对类别之间做一次nms luowei add
if __name__ == '__main__':
    main()
