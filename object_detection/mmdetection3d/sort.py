"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import copy

import iou

np.random.seed(0)

class Det():
  def __init__(self,path_index, x_raw, y_raw, z_raw,angle_raw,cx,cy,cz,dx,dy,dz,angle,cls,conf,pnums):

    self.path_index = path_index
    self.x_raw = x_raw
    self.y_raw = y_raw
    self.z_raw = z_raw
    self.angle_raw = angle_raw

    self.id = -1 #跟踪完成后，ID从1开始编号
    self.show_id = -1

    self.cx = cx #中心点x坐标(卡尔曼状态1)
    self.cy = cy #中心点y坐标(卡尔曼状态2)
    self.cz = cz  # 中心点z坐标

    self.dx = dx #x方向大小
    self.dy = dy #y方向大小
    self.dz = dz #z方向大小

    self.angle = angle #旋转角度

    self.cls = cls

    self.conf = conf

    self.pnums = pnums




def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

'''
def get_iou(a, b, epsilon=1e-5):
  # COORDINATES OF THE INTERSECTION BOX
  x1 = max(a[0], b[0])
  y1 = max(a[1], b[1])
  x2 = min(a[2], b[2])
  y2 = min(a[3], b[3])

  # AREA OF OVERLAP - Area where the boxes intersect
  width = (x2 - x1)
  height = (y2 - y1)
  # handle case where there is NO overlap
  if (width < 0) or (height < 0):
    return 0.0
  area_overlap = width * height

  # COMBINED AREA
  area_a = (a[2] - a[0]) * (a[3] - a[1])
  area_b = (b[2] - b[0]) * (b[3] - b[1])
  area_combined = area_a + area_b - area_overlap

  # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
  iou = area_overlap / (area_combined + epsilon)
  return iou

def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  # bb_gt = np.expand_dims(bb_gt, 0)
  # bb_test = np.expand_dims(bb_test, 1)
  #
  # xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  # yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  # xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  # yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  # w = np.maximum(0., xx2 - xx1)
  # h = np.maximum(0., yy2 - yy1)
  # wh = w * h
  # o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
  #   + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
  #return(o)

  iouss = []
  for det1 in bb_test:
    ious = []
    for det2 in bb_gt:
      det1_x1 = det1.cx - det1.dx / 2
      det1_x2 = det1.cx + det1.dx / 2
      det1_y1 = det1.cy - det1.dy / 2
      det1_y2 = det1.cy + det1.dy / 2

      det2_x1 = det2.cx - det2.dx / 2
      det2_x2 = det2.cx + det2.dx / 2
      det2_y1 = det2.cy - det2.dy / 2
      det2_y2 = det2.cy + det2.dy / 2

      iou = get_iou([det1_x1,det1_y1,det1_x2,det1_y2],
                    [det2_x1,det2_y1,det2_x2,det2_y2], epsilon=1e-5)
      ious.append(iou)
    iouss.append(ious)
  return np.array(iouss)
'''



def iou_batch(bb_test, bb_gt):

  iouss = []
  for det1 in bb_test:
    ious = []
    for det2 in bb_gt:
      res_iou = iou.get_iou([det1.cx, det1.cy, det1.dx, det1.dy, det1.angle],  # cx, cy, x_d, y_d, angle
                            [det2.cx, det2.cy, det2.dx, det2.dy, det2.angle])  ##cx, cy, x_d, y_d, angle

      ious.append(res_iou)
    iouss.append(ious)

  iouss = np.array(iouss)

  print("iou=",iouss)
  return iouss



def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  # w = bbox[2] - bbox[0]
  # h = bbox[3] - bbox[1]
  # x = bbox[0] + w/2.
  # y = bbox[1] + h/2.
  # s = w * h    #scale is just area
  # r = w / float(h)
  # return np.array([x, y, s, r]).reshape((4, 1))
  cx = bbox.cx
  cy = bbox.cy
  return np.array([cx,cy]).reshape((2, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  # w = np.sqrt(x[2] * x[3])
  # h = x[2] / w
  # if(score==None):
  #   return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  # else:
  #   return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
  cx = float(x[0])
  cy = float(x[1])
  return cx,cy


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  show_count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    #self.kf = KalmanFilter(dim_x=7, dim_z=4)

    self.kf = KalmanFilter(dim_x=4, dim_z=2)

    #                        x,y,s,r,x,y,s
    # self.kf.F = np.array([[1,0,0,0,1,0,0],
    #                       [0,1,0,0,0,1,0],
    #                       [0,0,1,0,0,0,1],
    #                       [0,0,0,1,0,0,0],
    #                       [0,0,0,0,1,0,0],
    #                       [0,0,0,0,0,1,0],
    #                       [0,0,0,0,0,0,1]])

    #                      x,y,x,y
    self.kf.F = np.array([[1,0,1,0],
                          [0,1,0,1],
                          [0,0,1,0],
                          [0,0,0,1]])

    #                        x,y,s,r,x,y,s
    # self.kf.H = np.array([[1,0,0,0,0,0,0],
    #                       [0,1,0,0,0,0,0],
    #                       [0,0,1,0,0,0,0],
    #                       [0,0,0,1,0,0,0]])

    #                      x,y,x,y
    self.kf.H = np.array([[1,0,0,0],
                          [0,1,0,0]])

    # self.kf.R[2:,2:] *= 10.
    # self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    # self.kf.P *= 10.
    # self.kf.Q[-1,-1] *= 0.01
    # self.kf.Q[4:,4:] *= 0.01




    self.kf.P *= 1000. #give high uncertainty to the unobservable initial velocities

    self.kf.R *= 5.

    self.kf.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.13)

    # print("状态转移矩阵F=\n",self.kf.F,"\n",
    #       "测量函数H=\n",self.kf.H,"\n",
    #       "协方差矩阵P=\n",self.kf.P,"\n",
    #       "测量噪声R=\n", self.kf.R, "\n",
    #       "过程噪声Q=\n",self.kf.Q,"\n")

    self.rect = bbox
    self.kf.x[:2] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    KalmanBoxTracker.count += 1
    self.id = KalmanBoxTracker.count
    self.rect.id = self.id

    self.show_id = -1


    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def set_show_id(self):
    if self.show_id==-1: #如果没有赋值过显示ID,则给一次赋值
      KalmanBoxTracker.show_count += 1
      self.show_id = KalmanBoxTracker.show_count
      self.rect.show_id = self.show_id

  def update(self,det):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    #self.kf.update(convert_bbox_to_z(bbox))

    #self.rect = det #除了id和show id(因为点迹里面没有)，中心点和大小都更新
    if det.pnums > self.rect.pnums: #如果新检测的大于之前的，才更新中心点和大小
      self.rect = det

    self.rect.id = self.id
    self.rect.show_id = self.show_id

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    # if((self.kf.x[6]+self.kf.x[2])<=0):
    #   self.kf.x[6] *= 0.0

    #self.kf.predict()
    self.age += 1
    #if(self.time_since_update>0):
    #  self.hit_streak = 0 #命中次数不要清0
    self.time_since_update += 1
    #self.rect.cx,self.rect.cy = convert_x_to_bbox(self.kf.x)
    self.history.append(copy.deepcopy(self.rect))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    #self.rect.cx, self.rect.cy = convert_x_to_bbox(self.kf.x)
    return self.rect

  def footprint_to_world(self, wx, wy, wz, RT):
    footprint_coordinate = np.mat([[wx], [wy], [wz], [1]])
    world_coordinate = RT * footprint_coordinate

    return float(world_coordinate[0]), float(world_coordinate[1]), float(world_coordinate[2])

  def update_trk_pose(self,obj_paths):
    path_index = self.rect.path_index  # 读取索引号
    RT = np.zeros((4, 4), np.float32)
    RT[:3, :3] = obj_paths[path_index].r_matrix  # Rota_c2w
    RT[:3, 3] = obj_paths[path_index].trans  # Trans_c2w
    RT[3, 3] = 1

    # 用新的RT做一次转换
    self.rect.cx, self.rect.cy, self.rect.cz = self.footprint_to_world(self.rect.x_raw, self.rect.y_raw, self.rect.z_raw, RT)
    self.rect.angle = self.rect.angle_raw + obj_paths[path_index].z_euler




def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, obj_paths, dets=[]):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    
    self.frame_count += 1
    # get predicted locations from existing trackers.
    # trks = np.zeros((len(self.trackers), 5))
    # to_del = []
    ret = []
    # for t, trk in enumerate(trks):
    #   pos = self.trackers[t].predict()[0]
    #   trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
    #   if np.any(np.isnan(pos)):
    #     to_del.append(t)


    for tracker in self.trackers:
      tracker.update_trk_pose(obj_paths)

    trks = []
    for tracker in self.trackers:
      pos = tracker.predict() #[0]
      trks.append(pos)


    #trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) #清除无效数值NaNs或infs
    # for t in reversed(to_del):
    #   self.trackers.pop(t)


    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0]])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i])
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        #print("d=",d)
        #print("time_since_update=",trk.time_since_update,"max_age=",self.max_age)
        #if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):#trk.time_since_update < 1表示不显示外推
        #if (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):#显示外推，hit_streak表示连续命中此数，这样还是显示不出来
        if (trk.hit_streak >= self.min_hits): #or self.frame_count <= self.min_hits):
            #ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            trk.set_show_id() #设置递增的show id
            ret.append(trk.get_state())
        i -= 1

        # remove dead tracklet
        if(trk.time_since_update >= self.max_age and trk.hit_streak<self.min_hits): #只删除不满足命中次数的，即没有起批的航迹
            self.trackers.pop(i)
            print("pop")
          
    # if(len(ret)>0):
    #   return np.concatenate(ret)
    #
    # return np.empty((0,5))

    return ret



# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(description='SORT demo')
#     parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
#     parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
#     parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
#     parser.add_argument("--max_age",
#                         help="Maximum number of frames to keep alive a track without associated detections.",
#                         type=int, default=1)
#     parser.add_argument("--min_hits",
#                         help="Minimum number of associated detections before track is initialised.",
#                         type=int, default=3)
#     parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
#     args = parser.parse_args()
#     return args



# if __name__ == '__main__':
#   # all train
#   args = parse_args()
#   display = args.display
#   phase = args.phase
#   total_time = 0.0
#   total_frames = 0
#   colours = np.random.rand(32, 3) #used only for display
#   if(display):
#     if not os.path.exists('mot_benchmark'):
#       print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
#       exit()
#     plt.ion()
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111, aspect='equal')
#
#   if not os.path.exists('output'):
#     os.makedirs('output')
#   pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
#   for seq_dets_fn in glob.glob(pattern):
#     mot_tracker = Sort(max_age=args.max_age,
#                        min_hits=args.min_hits,
#                        iou_threshold=args.iou_threshold) #create instance of the SORT tracker
#     seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
#     seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
#
#     with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
#       print("Processing %s."%(seq))
#       for frame in range(int(seq_dets[:,0].max())):
#         frame += 1 #detection and frame numbers begin at 1
#         dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
#         dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
#         total_frames += 1
#
#         if(display):
#           fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
#           im =io.imread(fn)
#           ax1.imshow(im)
#           plt.title(seq + ' Tracked Targets')
#
#         start_time = time.time()
#         trackers = mot_tracker.update(dets)
#         cycle_time = time.time() - start_time
#         total_time += cycle_time
#
#         for d in trackers:
#           print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
#           if(display):
#             d = d.astype(np.int32)
#             ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
#
#         if(display):
#           fig.canvas.flush_events()
#           plt.draw()
#           ax1.cla()
#
#   print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
#
#   if(display):
#     print("Note: to get real runtime results run without the option: --display")
