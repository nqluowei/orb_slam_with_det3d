import os
os.putenv("LD_LIBRARY_PATH","/home/luowei/ai/test_env_ws/devel/lib")
import sys
sys.path.append("/opt/ros/melodic/lib/python2.7/dist-packages")
sys.path.append("/home/luowei/ai/test_env_ws/devel/lib/python3/dist-packages")

import time

import ros_numpy
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry,Path
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point
import tf
import cv2
import platform
from scipy.spatial.transform import Rotation as Rota


import open3d as o3d
import numpy as np
import fcaf3d_demo as demo


from sort import *
import numpy as np
import cv2

import numpy as np
# 关闭numpy中的科学计数输出
np.set_printoptions(precision=4, suppress=True)

HEI = 480
WID = 640


#ROS
fx,fy,cx,cy = 546.0500834384593, 545.7495208147751, 318.26543549764034, 235.530988776277
K = np.mat([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])


pcd_update_flag = False
points = None
rgb = None
# Trans_c2w = None
# Rota_c2w = None
# Rota_euler = None

Trans_c2f = None
Rota_c2f = None

color_img = None
obj_paths = []

class ObjPath():
  def __init__(self):
    self.trans = None
    self.r_matrix = None
    self.z_euler = None

    self.trks = None


import message_filters

class PointCloudSubscriber(object):
    def __init__(self):
        # self.sub = rospy.Subscriber("cur_cloud", PointCloud2, self.callback, queue_size=10)
        #
        # self.listener = tf.TransformListener()

        cur_cloud = message_filters.Subscriber("cur_cloud", PointCloud2)
        #odom = message_filters.Subscriber("odom", Odometry)
        color = message_filters.Subscriber("slam_frame", Image)
        slam_path = message_filters.Subscriber("slam_path", Path)
        cur_cloud_color_slam_path = message_filters.TimeSynchronizer([cur_cloud, color,slam_path], 1)  # 绝对时间同步
        self.bridge = CvBridge()
        cur_cloud_color_slam_path.registerCallback(self.callback)

    def callback(self, msg, msg2, msg3):
        global pcd_update_flag
        #global points,rgb,Trans_c2w,Rota_c2w,Trans_c2f,Rota_c2f,Rota_euler,color_img,obj_paths #注意，必须写全局变量！！！
        global points, rgb, Trans_c2f, Rota_c2f, color_img, obj_paths  # 注意，必须写全局变量！！！

        if pcd_update_flag==True: #是True表示主线程还没检测完成，不进行新加入
            return

        assert isinstance(msg, PointCloud2)
        assert isinstance(msg2, Image)
        assert isinstance(msg3, Path)

        color_img = self.bridge.imgmsg_to_cv2(msg2, 'bgr8')

        #print("msg3=",msg3)

        #points = point_cloud2.read_points_list(msg, field_names=("x", "y", "z","rgb"))

        pc = ros_numpy.numpify(msg)
        pc = ros_numpy.point_cloud2.split_rgb_field(pc)
        points = np.zeros((pc.shape[0], 3))
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        rgb = np.zeros((pc.shape[0], 3))
        rgb[:, 0] = pc['r']
        rgb[:, 1] = pc['g']
        rgb[:, 2] = pc['b']

        #print("points=",points)

        #(const std::string& target_frame, const std::string& source_frame,const ros::Time& time, const ros::Duration timeout)
        #(Trans_c2w, rota_c2w) = self.listener.lookupTransform('/camera_optical', '/map', rospy.Time(0)) #camera_optical->map，可用于像素坐标转世界坐标

        #map->footprint
        # try:
        #     (Trans_c2w, rota_c2w) = self.listener.lookupTransform('/map', '/camera_footprint', rospy.Time(0)) #camera_optical->map，可用于像素坐标转世界坐标
        # except:
        #     print("lookupTransform error")
        #     return

        # p = msg1.pose.pose.position
        # q = msg1.pose.pose.orientation
        # Trans_c2w = np.array([p.x,p.y,p.z]) #xyz
        # rota_c2w = [q.x,q.y,q.z,q.w]  #四原数
        # r = Rota.from_quat(np.array(rota_c2w))
        # Rota_c2w = r.as_matrix()
        # Rota_euler = float(r.as_euler('xyz', degrees=False)[2]) #z轴欧拉角

        for i in range(len(msg3.poses)):
            p = msg3.poses[i].pose.position
            q = msg3.poses[i].pose.orientation
            trans = np.array([p.x, p.y, p.z])  # xyz
            quan = [q.x, q.y, q.z, q.w]  # 四原数
            r = Rota.from_quat(np.array(quan))
            r_matrix = r.as_matrix()
            z_euler = float(r.as_euler('xyz', degrees=False)[2])  # z轴欧拉角

            if i>=len(obj_paths): #
                obj_paths.append(ObjPath())

            #更新位姿
            obj_paths[i].trans = trans
            obj_paths[i].r_matrix = r_matrix
            obj_paths[i].z_euler = z_euler



        #camera->footprint
        # try:
        #     (Trans_c2f, rota_c2f) = self.listener.lookupTransform('/camera_optical', '/camera_footprint', rospy.Time(0)) #camera_optical->map，可用于像素坐标转世界坐标
        # except:
        #     print("lookupTransform error")
        #     return
        #
        # r = Rota.from_quat(rota_c2f)
        # Rota_c2f = r.as_matrix()
        #
        # print("Trans_c2f,Rota_c2f=",Trans_c2f, Rota_c2f)

        Trans_c2f = np.array([0, 0.326921, 0])
        Rota_c2f = np.array([[-0.,- 1.,- 0.],
                            [-0.,0.,- 1.],
                            [1.,- 0.,- 0.]])

        
        #print("Rota_euler=",Rota_euler)

        print("recv points num=",len(points))

        pcd_update_flag = True

#              0       1        2       3        4        5       6              7            8            9
cls_names = ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub']


def box_to_corners(box):
    # generate clockwise corners and rotate it clockwise
    # 顺时针方向返回角点位置
    cx, cy, cz, dx, dy, dz, angle,cls, conf,pnums = box

    a_cos = np.cos(-angle) #注意这里角度yao jia fu hao
    a_sin = np.sin(-angle)
    corners_x = [-dx / 2, -dx / 2, dx / 2, dx / 2]
    corners_y = [-dy / 2, dy / 2, dy / 2, -dy / 2]

    WZ_bottom = cz - dz/2
    WZ_top =    cz + dz/2

    corners = []
    for i in range(4):
        X = a_cos * corners_x[i] + \
                     a_sin * corners_y[i] + cx
        Y = -a_sin * corners_x[i] + \
                     a_cos * corners_y[i] + cy
        corners.append([X, Y, WZ_bottom])
        corners.append([X, Y, WZ_top])

    return corners #返回8个顶点坐标

line_indexs = [[0, 1], [2, 3], [4, 5], [6, 7],
         [0, 2], [2, 4], [4, 6], [6, 0],
         [1, 3], [3, 5], [5, 7], [7, 1]]

def pub_marker(det):

    # path_index = det.path_index #读取索引号
    # RT = np.zeros((4, 4), np.float32)
    # RT[:3, :3] = obj_paths[path_index].r_matrix  # Rota_c2w
    # RT[:3, 3] = obj_paths[path_index].trans  # Trans_c2w
    # RT[3, 3] = 1
    #
    # # 用新的RT做一次转换
    # cx, cy, cz = footprint_to_world(det.x_raw, det.y_raw, det.z_raw, RT)
    # angle = det.angle_raw + obj_paths[path_index].z_euler

    #长方体
    marker_cube = Marker()
    marker_cube.header.frame_id = "map"
    marker_cube.header.stamp = rospy.Time.now()
    marker_cube.ns = "my_cube"
    marker_cube.id = det.show_id
    marker_cube.type = Marker.CUBE  # Marker.MESH_RESOURCE
    marker_cube.action = Marker.ADD
    marker_cube.pose.position.x = det.cx
    marker_cube.pose.position.y = det.cy
    marker_cube.pose.position.z = det.cz
    r = Rota.from_euler('xyz', (0, 0, det.angle), degrees=False)
    quaternion = r.as_quat()
    marker_cube.pose.orientation.x = quaternion[0]
    marker_cube.pose.orientation.y = quaternion[1]
    marker_cube.pose.orientation.z = quaternion[2]
    marker_cube.pose.orientation.w = quaternion[3]
    marker_cube.scale.x = det.dx
    marker_cube.scale.y = det.dy
    marker_cube.scale.z = det.dz

    marker_cube.color.a = 0.5  # Don't forget to set the alpha!
    marker_cube.color.r = 138 / 255
    marker_cube.color.g = 226 / 255
    marker_cube.color.b = 52 / 255

    # marker.mesh_resource = "file:///home/luowei/ai/ORB_SLAM2_DENSE_ws/src/ORB-SLAM2_DENSE/Examples/ROS/orb_slam2_dense/params/cube_0.1.dae"
    #marker_pub.publish(marker)

    #3D框
    marker_box = Marker()
    marker_box.header.frame_id = "map"
    marker_box.header.stamp = rospy.Time.now()
    marker_box.ns = "my_box"
    marker_box.id = det.show_id
    marker_box.type = Marker.LINE_LIST
    marker_box.action = Marker.ADD
    #marker.lifetime = rospy.Duration(0)
    marker_box.pose.orientation.x = 0.0
    marker_box.pose.orientation.y = 0.0
    marker_box.pose.orientation.z = 0.0
    marker_box.pose.orientation.w = 1.0 #防止警告
    marker_box.color.a = 1  # Don't forget to set the alpha!
    marker_box.color.r = 138 / 255
    marker_box.color.g = 226 / 255
    marker_box.color.b = 52 / 255

    marker_box.scale.x = 0.03 #线宽
    marker_box.points = []

    corners = box_to_corners((det.cx, det.cy, det.cz, det.dx, det.dy, det.dz, det.angle, det.cls, det.conf, det.pnums))

    for line_index in line_indexs:
        sta = corners[line_index[0]]
        end = corners[line_index[1]]
        marker_box.points.append(Point(sta[0],sta[1],sta[2]))
        marker_box.points.append(Point(end[0],end[1],end[2]))
    #marker_pub.publish(marker)


    # TEXT_VIEW_FACING
    marker_text = Marker()
    marker_text.header.frame_id = "map"
    marker_text.header.stamp = rospy.Time.now()
    marker_text.ns = "my_text"
    marker_text.id = det.show_id
    marker_text.type = Marker.TEXT_VIEW_FACING
    marker_text.action = Marker.ADD
    marker_text.pose.position.x = det.cx
    marker_text.pose.position.y = det.cy
    marker_text.pose.position.z = det.cz + det.dz / 2 + 0.1
    r = Rota.from_euler('xyz', (0, 0, det.angle), degrees=False)

    quaternion = r.as_quat()
    marker_text.pose.orientation.x = quaternion[0]
    marker_text.pose.orientation.y = quaternion[1]
    marker_text.pose.orientation.z = quaternion[2]
    marker_text.pose.orientation.w = quaternion[3]

    marker_text.scale.z = 0.2

    marker_text.color.a = 1.0  # Don't forget to set the alpha!
    marker_text.color.r = 138 / 255
    marker_text.color.g = 226 / 255
    marker_text.color.b = 52 / 255

    marker_text.text = "%s%d_%.2f"%(cls_names[int(det.cls)],det.show_id,float(det.conf)) #,det.pnums)

    # marker.mesh_resource = "file:///home/luowei/ai/ORB_SLAM2_DENSE_ws/src/ORB-SLAM2_DENSE/Examples/ROS/orb_slam2_dense/params/cube_0.1.dae"
    #marker_pub.publish(marker)

    return marker_cube,marker_box,marker_text



def world_to_pixel(wx,wy,wz,RT,K):
    world_coordinate = np.mat([[wx], [wy], [wz], [1]])
    # 世界坐标系转换为相加坐标系 （Xw,Yw,Zw）--> (Xc,Yc,Zc)
    camera_coordinate = RT * world_coordinate
    # print(f'相机坐标为：\n{camera_coordinate}')
    Zc = float(camera_coordinate[2])
    if Zc==0:
        return 0,0
    # 相机坐标系转图像坐标系 (Xc,Yc,Zc) --> (x, y)  下边的f改为焦距
    # image_coordinate = (focal_length * camera_coordinate) / Zc
    image_coordinate = camera_coordinate[:-1] / Zc
    # 图像坐标系转换为像素坐标系
    pixel_coordinate = K * image_coordinate

    #print("pixel_coordinate=",wx,wy,wz,pixel_coordinate)

    if np.isnan(pixel_coordinate[0]) or np.isnan(pixel_coordinate[1]):
        return 0,0
    else:
        return int(pixel_coordinate[0]),int(pixel_coordinate[1])
        

def footprint_to_world(wx,wy,wz,RT):
    footprint_coordinate = np.mat([[wx], [wy], [wz], [1]])
    world_coordinate = RT * footprint_coordinate 

    return float(world_coordinate[0]),float(world_coordinate[1]),float(world_coordinate[2])
        
        

#假设Yw=0
def pixel_to_world(R,t,K,pixel_coordinates):
    R = np.mat(R)
    t = np.mat(t)

    world_list = []
    for x, y in pixel_coordinates:
        pixel_coordinate = np.mat([[x], [y], [1]])
        print("shape====", R.shape, t)
        mat1 = R.I * K.I * pixel_coordinate
        mat2 = R.I * t.T

        # 世界坐标系 = Zc(光心深度) * mat1  - mat2
        # Yworld  = 0 = Zcamera * mat1[1,0] - mat2[1,0]
        #Z_c = mat2[1, 0] / mat1[1, 0]
        Z_c = mat2[2, 0] / mat1[2, 0]
        world = Z_c * mat1 - mat2

        wx, wy, wz = world[0, 0], world[1, 0], world[2, 0]  # y轴向下为正

        print("world===",wx, wy, wz)
        world_list.append((wx, wy, wz))

    return world_list
    
    

if __name__ =='__main__':
    # print("当前python版本:",platform.python_version())

    net = demo.model_init()

    rospy.init_node("obj_detect_node")
    marker_pub = rospy.Publisher("box_marker", MarkerArray, queue_size=10)
    PointCloudSubscriber()

    image_pub = rospy.Publisher('det_img', Image, queue_size=10)
    bridge = CvBridge()

    #mot_tracker = Sort(max_age=1, min_hits=2, iou_threshold=0.2) #在这设置跟踪参数

    #注意在来第一帧的时候，没有发生关联，所以也就没有卡尔曼更新，第二帧的时候min_hits才等于1，表示关联命中一次
    mot_tracker = Sort(max_age=2, min_hits=0, iou_threshold=0.001)

    SHOW = False #True #False

    if SHOW:
        # 方法1（非阻塞显示）
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=900, height=900, left=0, top=0)

        # # 方法2（阻塞显示）：调用draw_geometries直接把需要显示点云数据
        # test_pcd.points = open3d.utility.Vector3dVector(points)  # 定义点云坐标位置
        # test_pcd.colors = open3d.Vector3dVector(colors)  # 定义点云的颜色
        # open3d.visualization.draw_geometries([test_pcd] + [axis_pcd], window_name="Open3D2")

        coordinate0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])



    start_flag=True
    POINTS = None


    frame_cnt = 0

    box_cnt = 0

    pcd = o3d.geometry.PointCloud()

    while  not rospy.is_shutdown():

        if pcd_update_flag==True:

            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

            # 新改的这个RT是为了和mm3d里面的sun rgbd数据集尽量一致
            r = Rota.from_euler('xyz', [-90, 0, 0], degrees=True)
            R = r.as_matrix()
            RT = np.identity(4)
            RT[0:3, 0:3] = R
            RT[2, 3] =  0 #0  # 0.326921
            # print("RT=",RT)
            pcd.transform(RT)


            if SHOW:
                vis.clear_geometries() #清除所有物体

                vis.add_geometry(pcd)
                
                #frame_cnt+=1
                #o3d.io.write_point_cloud("ply_output/%d.ply"%(frame_cnt), pcd)

                #print("!!Rota_c2f=",Rota_c2f)
                # Rota_world_2_camera = np.linalg.inv(Rota_c2f)
                #
                # Trans_world_2_camera = -Rota_world_2_camera@Trans_c2f
                # #print("Trans_world_2_camera=",Trans_world_2_camera)
                #
                # coordinate1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=Trans_world_2_camera)
                # coordinate1.rotate(Rota_world_2_camera, center=Trans_world_2_camera)
                #
                # vis.add_geometry(coordinate1)

                # pixel_coordinates = [(0, HEI), (WID, HEI),(0, HEI * 0.55), (WID, HEI * 0.55)]
                # img_outline_world_list = pixel_to_world(Rota_c2f, Trans_c2f, K, pixel_coordinates)
                #
                # for world_coord in img_outline_world_list:
                #     vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=world_coord))


            data_np = np.concatenate((np.array(pcd.points),rgb),axis=1)
            pred_bboxes,output_bboxes = demo.infer(net, pcd, data_np) #模型推理
            
            #output_bboxes = np.array([(2.,0.,0.05,1.,0.1,0.1,np.pi/6,1)]) #x, y, z, dx, dy, dz, angle, cls

            #output_bboxes = []

            if True:
                #color_img = np.zeros((HEI, WID, 3), np.uint8)

                RT = np.zeros((4, 4), np.float32)
                RT[:3, :3] = Rota_c2f
                RT[:3, 3] = Trans_c2f
                RT[3, 3] = 1


                keep_mask = np.zeros((len(output_bboxes)),np.bool)

                #过滤掉接近图像边缘的框
                for i,box in enumerate(output_bboxes):

                    corners = box_to_corners(box)
                    is_overflow = False

                    ps = []
                    for WX,WY,WZ in corners:
                        px, py = world_to_pixel(WX, WY, WZ, RT, K)
                        if px < 10:
                            #px = 10
                            is_overflow = True

                        if px > WID -10:
                            #px = WID -10
                            is_overflow = True




                        # if py < 20:
                        #     py = 20
                        #     is_overflow = True
                        #
                        # if py > HEI -20:
                        #     py = HEI - 20
                        #     is_overflow = True

                        #cv2.circle(color_img, (px, py), 5, (255, 255, 0), -1)

                        ps.append([px, py])

                    if is_overflow:
                        keep_mask[i] = False
                        color = (0, 0, 255)
                    else:
                        keep_mask[i] = True
                        color = (0, 255, 0)

                    for line_index in line_indexs:
                        start_index = line_index[0]
                        end_index = line_index[1]
                        cv2.line(color_img, (ps[start_index][0], ps[start_index][1]), (ps[end_index][0], ps[end_index][1]), color, 2)

                #过滤完成


                if len(output_bboxes)!=0:
                    #print("keep_mask=", keep_mask)
                    output_bboxes = output_bboxes[keep_mask]

                image_pub.publish(bridge.cv2_to_imgmsg(color_img, "bgr8"))




            path_index = len(obj_paths) - 1
            RT = np.zeros((4, 4), np.float32)
            RT[:3, :3] = obj_paths[path_index].r_matrix #Rota_c2w
            RT[:3, 3] =  obj_paths[path_index].trans #Trans_c2w
            RT[3, 3] = 1

            #转到世界坐标系
            dets = []
            for i,(x_raw, y_raw, z_raw, dx, dy, dz, angle_raw, cls, conf,pnums) in enumerate(output_bboxes):
                xw,yw,zw = footprint_to_world(x_raw, y_raw, z_raw,RT)
                dets.append(Det(path_index, x_raw, y_raw, z_raw,angle_raw, xw, yw, zw, dx, dy, dz, angle_raw + obj_paths[path_index].z_euler, cls,conf, pnums)) #欧拉角补偿

            #dets = [Det(x, y, z, dx, dy, dz, angle, cls) for x, y, z, dx, dy, dz, angle, cls in output_bboxes]

            #dets = [Det(1.6934431791305542,1.8482710123062134, 1, 0.5154457899289131, 0.4967212326376438, 0.4, 0)]

            # print("dets")
            # for trk in dets:
            #     print("det=", trk.cx, trk.cy, trk.dx, trk.dy, trk.id, trk.show_id, trk.cls, trk.conf)
            trks = mot_tracker.update(obj_paths,dets) #进函数之后首先会根据位姿更新所有卡尔曼对象里面的cx,cy,cz,angele
            # print("trks")
            # for trk in trks:
            #     print("trk=", trk.cx, trk.cy, trk.dx, trk.dy, trk.id, trk.show_id, trk.cls, trk.conf)

            marker_array = MarkerArray()
            for trk in trks:
                marker_cube,marker_box,marker_text = pub_marker(trk)

                marker_array.markers.append(marker_cube)
                marker_array.markers.append(marker_box)
                marker_array.markers.append(marker_text)

            marker_pub.publish(marker_array)




            if SHOW:
                for x,y,z,dx,dy,dz,angle in pred_bboxes:
                    z += dz / 2  # 移动到真的中心
                    R_box = pcd.get_rotation_matrix_from_axis_angle(np.array([0, 0, angle]).T)
                    boundingbox = o3d.geometry.OrientedBoundingBox(np.array([x,y,z]),
                                                                   R_box,
                                                                   np.array([dx,dy,dz])
                                                                   )
                    indices = boundingbox.get_point_indices_within_bounding_box(pcd.points)
                    boundingbox.color = (1, 0, 0)
                    vis.add_geometry(boundingbox)




            if SHOW:
                vis.add_geometry(coordinate0)

                para = vis.get_view_control().convert_to_pinhole_camera_parameters()
                R = pcd.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0, 0]).T)
                para.extrinsic = np.mat([
                    [R[0, 0], R[0, 1], R[0, 2], 0],
                    [R[1, 0], R[1, 1], R[1, 2], 0],
                    [R[2, 0], R[2, 1], R[2, 2], 10],
                    [0, 0, 0, 1]
                ])
                vis.get_view_control().convert_from_pinhole_camera_parameters(para)

                vis.poll_events()
                vis.update_renderer()



                cv2.imshow("color_img", color_img)
                if cv2.waitKey(1) == ord('q'):
                    print("按下q键值暂停",path_index)
                    vis.run() #会阻塞




            pcd_update_flag = False

        else:
            time.sleep(0.01)  #10ms






