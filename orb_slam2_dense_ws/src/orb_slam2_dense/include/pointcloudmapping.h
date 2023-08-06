/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

// STL
#include <condition_variable>

// PCL
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/cloud_viewer.h>

// ORB_SLAM2
#include "System.h"
#include "PointCloude.h"

using namespace std;
using namespace ORB_SLAM2;

class PointCloudMapping
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    PointCloudMapping(const std::string &strSettingPath, bool bUseViewer=true);
    
    void save();
    
    // 插入一个keyframe，会更新一次地图
    void insertKeyFrame( KeyFrame* kf, cv::Mat& color, cv::Mat& depth,int idk,vector<KeyFrame*> vpKFs );
    
    void shutdown();
    
    void run();
    
    void updateCloseLoopCloud();
    
    int loopcount = 0;
    vector<KeyFrame*> currentvpKFs;
    
    // control flag
    bool mbCloudBusy;
    bool mbLoopBusy;
    bool mbStop;
    
protected:
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    
    shared_ptr<thread>  mThdRunning;
    
    // shutdown
    mutex   shutDownMutex;
    bool    mbShutDownFlag;
    
    // store keyframe
    condition_variable  keyFrameUpdated;
    mutex               keyFrameUpdateMutex;
    vector<PointCloude, Eigen::aligned_allocator<Eigen::Isometry3d> >     pointcloud;
    
    // data to generate point clouds
    vector<KeyFrame*>       keyframes;
    vector<cv::Mat>         colorImgs;
    vector<cv::Mat>         depthImgs;
    vector<cv::Mat>         colorImgks;
    vector<cv::Mat>         depthImgks;
    vector<int>             ids;
    mutex                   keyframeMutex;
    uint16_t                lastKeyframeSize =0;
    
    // statistical filter
    pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    double meank_ = 50;
    double thresh_ = 1;
    
    // voxel grid filter
    pcl::VoxelGrid<PointT>  voxel;
    double resolution_ = 0.04;

//! yonghui add
public:
    /**
     * @brief Set point cloud update flag
     */
    void setPointCloudMapUpdatedFlag(int flag);
    
    /**
     * @brief Get point cloud update flag
     */
    int getPointCloudMapUpdatedFlag();

    /**
     * @brief Thread-safe interface to get whole point cloud map
     */
    bool getGlobalCloud(PointCloud::Ptr &pCloud);

    bool getCurrentCloud(PointCloud::Ptr &pCloud);

    vector<cv::Mat> get_all_poses();

    
    /**
     * @brief Thread-safe interface to update footprint<--optical extrinsic matrix
     */
    void updateTbc(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &tbc);
    
    /**
     * @brief Thread-safe interface to update footprint<--optical extrinsic matrix
     */
    void updateTbc(const Eigen::Matrix4d &Tbc);
    
    /**
     * @brief Thread-safe interface to footprint<--optical extrinsic matrix
     */
    void getTbc(Eigen::Matrix4d &Tbc);
    

    


protected:


    
    // output point cloud
    mutex mMtxTbcUpdated;
    Eigen::Matrix4d mTbc;
    double mfCameraHeight;
    

    
    // point cloud updated complete flag
    mutex mMtxPointCloudUpdated;
    mutex mMtxPosesUpdated;
    PointCloud::Ptr mpPclGlobalMap;
    PointCloud::Ptr mpPclCurrentMap;
    vector<cv::Mat> all_poses;
    
    bool mbPointCloudMapUpdated;
    
    // pcl viewer
    bool mbUseViewer;
    //pcl::visualization::CloudViewer mViewer;
    //pcl::visualization::PCLVisualizer mViewer;
};

#endif // POINTCLOUDMAPPING_H
