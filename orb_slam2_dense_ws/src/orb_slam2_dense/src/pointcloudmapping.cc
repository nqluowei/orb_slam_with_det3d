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

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// PCL
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>

// ORB_SLAM2
#include "pointcloudmapping.h"
#include "KeyFrame.h"
#include "Converter.h"
#include "PointCloude.h"
#include "System.h"
#include "TicToc.h"

// STL
#include <chrono>

bool firstKF = true;
int currentloopcount = 0;


PointCloudMapping::PointCloudMapping(const std::string &strSettingPath, bool bUseViewer) :

mbCloudBusy(false), mbLoopBusy(false), mbStop(false), mbShutDownFlag(false), 
mpPclGlobalMap(new PointCloudMapping::PointCloud()),
mpPclCurrentMap(new PointCloudMapping::PointCloud()),
mbPointCloudMapUpdated(0),
mbUseViewer(bUseViewer)
{
    // parse parameters
    cv::FileStorage fsSetting = cv::FileStorage(strSettingPath, cv::FileStorage::READ);
    cv::FileNode fsPointCloudMapping = fsSetting["PointCloudMapping"];
    
    // set initial Tbc: footprint<--optical
    cv::FileNode fsTbc = fsPointCloudMapping["Tbc"];
    Eigen::Vector3d tbc(fsTbc["x"], fsTbc["y"], fsTbc["z"]);
    Eigen::Matrix3d Rbc;
    Rbc = Eigen::AngleAxisd(fsTbc["roll"],  Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(fsTbc["pitch"], Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(fsTbc["yaw"],   Eigen::Vector3d::UnitZ());
    updateTbc(Rbc, tbc);
    
    // voxel grid filter
    double resolution_x = fsPointCloudMapping["ResolutionX"];
    double resolution_y = fsPointCloudMapping["ResolutionY"];
    double resolution_z = fsPointCloudMapping["ResolutionZ"];
    voxel.setLeafSize( resolution_x, resolution_y, resolution_z);
    
    // statistical filter
    cv::FileNode fsStatisticFilter = fsPointCloudMapping["StatisticFilter"];
    meank_ = fsStatisticFilter["MeanK"];
    thresh_ = fsStatisticFilter["Thres"];
    statistical_filter.setMeanK(meank_);
    statistical_filter.setStddevMulThresh(thresh_);


    
    cout << "---" << endl;
    cout << "Point Cloud Thread Parameters:" << endl;
    cout << "- x,y,z: " << (float)fsTbc["x"] << " " <<(float)fsTbc["y"]<< " " << (float)fsTbc["z"]  << endl;
    cout << "- roll,pitch,yaw: " << (float)fsTbc["roll"] << " " <<(float)fsTbc["pitch"]<< " " << (float)fsTbc["yaw"]  << endl;
    cout << "- Tbc: " << endl << mTbc << endl;
    cout << "- CameraHeight: " << mfCameraHeight << endl;
    cout << "- Resolution: " << resolution_x << " " << resolution_y << " "<< resolution_z <<  endl;

    cout << "- StatisticFilter " << endl;
    cout << "\t- MeanK: " <<  meank_ << endl;
    cout << "\t- Thres: " << thresh_ << endl;


    // start point cloud mapping thread
    mThdRunning = make_shared<thread>( bind(&PointCloudMapping::run, this ) );
}


void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        mbShutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    mThdRunning->join();
}


void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth,int idk,vector<KeyFrame*> vpKFs)
{
    cout<<"receive a keyframe, Frame id = "<< idk << " , KF No." << kf->mnId << endl;
    //cout<<"vpKFs数量"<<vpKFs.size()<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    currentvpKFs = vpKFs;
    //colorImgs.push_back( color.clone() );
    //depthImgs.push_back( depth.clone() );
    PointCloude pointcloude;
    pointcloude.pcID = idk;
    pointcloude.T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );



    pointcloude.pcE = generatePointCloud(kf,color,depth);
    pointcloud.push_back(pointcloude);
    
    
    {//加锁
        unique_lock<mutex> lock_poses(mMtxPosesUpdated);
        all_poses.push_back(kf->GetPose());//记录poses
    }
    
    {//加锁
        unique_lock<mutex> lock(mMtxPointCloudUpdated);
        *mpPclCurrentMap = *(pointcloude.pcE);//给当前点云赋值
        //pcl::transformPointCloud( *(pointcloud[i].pcE), *mpPclCurrentMap, Tbc);//根据外参Tbc移动到地面上
    }
    
    keyFrameUpdated.notify_one();
}


PointCloudMapping::PointCloud::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)//,Eigen::Isometry3d T
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    int pt_cnt = 0;
    PointT p;
    float d;
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            d = depth.ptr<float>(m)[n];
            if (isnan(d) || d < 0.01 || d>5) //设置min range和max range
                continue;

            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;

            p.r = color.ptr<uchar>(m)[n*3];//注意这里颜色顺序
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.b = color.ptr<uchar>(m)[n*3+2];

            tmp->points.push_back(p);
            pt_cnt++;
        }
    }

    //tmp->is_dense = false;

    //cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return tmp;
}


void PointCloudMapping::run()
{


    while(true)
    {
        
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (mbShutDownFlag)
            {
                cout << "mbShutDownFlag!!!" <<endl;
                break;
            }
        }

        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }
        
        // keyframe is updated 
        size_t N=0;

        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }
        
        // loop busy, or thread request stop
        if(mbLoopBusy || mbStop)
        {
            cerr << "Point Cloud Mapping thread is Looping or has terminated!" << endl;
            continue;
        }
        


        mbCloudBusy = true;
    


        // get extrinsic matrix
        Eigen::Matrix4d Tbc;//base->camera
        getTbc(Tbc);
        Eigen::Matrix4d Tcb = Tbc.inverse();
        
        // create new PointCloud
        PointCloud::Ptr pNewCloud(new PointCloud());

        for ( size_t i=lastKeyframeSize; i<N ; i++ )//处理lastKeyframeSize～N之间新出现的关键帧
        {


            //PointCloud::Ptr p_footprint (new PointCloud);
            PointCloud::Ptr p (new PointCloud);
            
            
            pcl::transformPointCloud( *(pointcloud[i].pcE), *p, pointcloud[i].T.inverse().matrix());//本地点云根据位姿转换到世界坐标系
            pcl::transformPointCloud( *p, *p, Tbc);//根据外参Tbc移动到地面上
            
            *pNewCloud += *p;//新插入的点云
        }
        


        //加上以前的所有点云
        *pNewCloud += *mpPclGlobalMap;


        // voxel grid filter
        //降低点云分辨率
        PointCloud::Ptr pNewCloudVoxelFilter(new PointCloud());

        //printf("555 start pNewCloudOutliersFilter=%d\n",pNewCloud->points.size());
        voxel.setInputCloud( pNewCloud );
        voxel.filter( *pNewCloudVoxelFilter );
        //printf("555 end pNewCloudVoxelFilter=%d\n",pNewCloudVoxelFilter->points.size());


        //必须自己编译pcl1.7.2 否则这些函数会崩
        {
            unique_lock<mutex> lock(mMtxPointCloudUpdated);
            mpPclGlobalMap->swap(*pNewCloudVoxelFilter);//赋值给mpPclGlobalMap
        }


        cout << "show global map, N=" << N << " points size=" << mpPclGlobalMap->points.size() << endl;


        // update flag
        lastKeyframeSize = N;
        mbCloudBusy = false;
        setPointCloudMapUpdatedFlag(1);
    }
}


void PointCloudMapping::save()
{
	pcl::io::savePCDFile( "result.pcd", *mpPclGlobalMap );
	cout<<"globalMap save finished"<<endl;
}


void PointCloudMapping::updateCloseLoopCloud()
{
    while(mbCloudBusy)
    {
        std::cout << "CloseLooping thread has activate point cloud map reconstruct, "
                     "but PointCloudMapping thread is busy currently." << std::endl;
        usleep(1000);
    }

    mbLoopBusy = true;
    std::cout << "******************* Start Loop Mapping *******************" << std::endl;
    


    // transform the whole point cloud according to extrinsic matrix
    Eigen::Matrix4d Tbc;
    getTbc(Tbc);
    Eigen::Matrix4d Tcb = Tbc.inverse();
    
    // reset new point cloud map
    PointCloud::Ptr pNewCloud(new PointCloud());

    cout << "Current KeyFrame size: " << currentvpKFs.size() << endl;
    cout << "Curremt PointCloude size: " << pointcloud.size() << endl;

    {//加锁
        unique_lock<mutex> lock_poses(mMtxPosesUpdated);
        all_poses.clear();//清空所有pose


        for (int j=0;j<pointcloud.size();j++)//内外循环顺序换了
        {
            for (int i=0;i<currentvpKFs.size();i++)//这个KFs可能是乱序的，画出来的轨迹不对
            {
                if(pointcloud[j].pcID==currentvpKFs[i]->mnFrameId)
                {
                    cout << "Start dealing with KeyFrame [" << i << "]" << endl;
                    //Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(currentvpKFs[i]->GetPose() );
                    pointcloud[j].T = ORB_SLAM2::Converter::toSE3Quat(currentvpKFs[i]->GetPose() );//直接修改进pointcloud数组

                    all_poses.push_back(currentvpKFs[i]->GetPose());//记录poses

                    PointCloud::Ptr p (new PointCloud);
                    pcl::transformPointCloud( *(pointcloud[j].pcE), *p, pointcloud[j].T.inverse().matrix());
                    pcl::transformPointCloud( *p, *p, Tbc);
                    *pNewCloud += *p;

                }
            }
        }
    }

    cout << "Gather all KeyFrame complete." << endl;
    

    // voxel grid filter
    PointCloud::Ptr pNewCloudVoxelFilter(new PointCloud());
    voxel.setInputCloud(pNewCloud);
    voxel.filter( *pNewCloudVoxelFilter );

    {
        unique_lock<mutex> lock(mMtxPointCloudUpdated);
        mpPclGlobalMap->swap(*pNewCloudVoxelFilter);
    }

    
    // update flag
    mbLoopBusy = false;
    loopcount++;
    setPointCloudMapUpdatedFlag(2);
    
    std::cout << "******************* Finish Loop Mapping *******************" << std::endl;
}


void PointCloudMapping::setPointCloudMapUpdatedFlag(int flag)
{
    unique_lock<mutex> lock(mMtxPointCloudUpdated);
    mbPointCloudMapUpdated = flag;
}


int PointCloudMapping::getPointCloudMapUpdatedFlag()
{
    unique_lock<mutex> lock(mMtxPointCloudUpdated);
    return mbPointCloudMapUpdated;
}


bool PointCloudMapping::getGlobalCloud(PointCloud::Ptr &pCloud)//获取所有点云
{
    unique_lock<mutex> lock(mMtxPointCloudUpdated);
    if (mpPclGlobalMap->empty())
    {
        cout << "mpPclGlobalMap->empty()!!!" <<endl;
        return false;
    }

    pCloud = mpPclGlobalMap->makeShared();
    return true;
}

bool PointCloudMapping::getCurrentCloud(PointCloud::Ptr &pCloud)//获取当前点云
{
    unique_lock<mutex> lock(mMtxPointCloudUpdated);
    if (mpPclCurrentMap->empty())
    {
        cout << "mpPclCurrentMap->empty()!!!" <<endl;
        return false;
    }

    pCloud = mpPclCurrentMap->makeShared();
    
    
    return true;
}

vector<cv::Mat> PointCloudMapping::get_all_poses()
{
    unique_lock<mutex> lock_poses(mMtxPosesUpdated);
    return all_poses;
}

void PointCloudMapping::updateTbc(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &tbc)
{
    unique_lock<mutex> lock(mMtxTbcUpdated);
    mTbc = Eigen::Matrix4d::Identity();
    mTbc.block(0,0,3,3) = Rbc;
    mTbc.block(0,3,3,1) = tbc;
    mfCameraHeight = tbc[2];
}


void PointCloudMapping::updateTbc(const Eigen::Matrix4d &Tbc)
{
    unique_lock<mutex> lock(mMtxTbcUpdated);
    mTbc = Tbc;
    mfCameraHeight = Tbc(3,3);
}


void PointCloudMapping::getTbc(Eigen::Matrix4d &Tbc)
{
    unique_lock<mutex> lock(mMtxTbcUpdated);
    Tbc = mTbc;
}


