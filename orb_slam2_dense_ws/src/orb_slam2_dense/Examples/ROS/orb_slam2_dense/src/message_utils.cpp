//
// Created by yonghui on 19-10-30.
//

// ROS
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_conversions/pcl_conversions.h>



// PCL
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>

#include "message_utils.h"

namespace ORB_SLAM2_DENSE
{
    MessageUtils::MessageUtils(tf::TransformListener &listener, ORB_SLAM2::System *pSystem) :
    private_nh_("~"), listener_(listener), mpSystem_(pSystem), pcl_map_(new PointCloudMapping::PointCloud()),
    pcl_plane_(new PointCloudMapping::PointCloud())
    {
        // initial plane coefficients ( xy plane)
        plane_coeffs_ << 0.0, 0.0, 1.0, 0.0;
        last_plane_coeffs_ << 0.0, 0.0, 1.0, 0.0;
        
        kf_cnt = 0;
        
        // initalize publisher and subscriber
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("odom_slam", 100);
        odom_pub_1 = nh_.advertise<nav_msgs::Odometry>("odom_wheel", 100);
        frame_pub_ = nh_.advertise<sensor_msgs::Image>("slam_frame", 10);
        //pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud>("cloud", 10);
        pcl2_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("map_cloud", 10);
        cur_pcl2_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cur_cloud", 10);
        
        path_pub = nh_.advertise<nav_msgs::Path>("slam_path",10);
        

        // get parameters
        private_nh_.param("use_odom_pub", use_odom_pub_, true);
        private_nh_.param("use_tf", use_tf_, true);
        private_nh_.param("use_frame_pub", use_frame_pub_, true);
        private_nh_.param("use_pcl_pub", use_pcl_pub_, true);
//        private_nh_.param("use_plane_segment", use_plane_segment_, true);
//        private_nh_.param("segment_min_z", min_z_, -0.5);
//        private_nh_.param("segment_max_z", max_z_,  0.5);
//        private_nh_.param("plane_dist_thres", plane_dist_thres_, 0.2);
        //get frame id
        private_nh_.param("map_frame", map_frame_, std::string("map"));
        private_nh_.param("odom_frame", odom_frame_, std::string("odom"));
        private_nh_.param("footprint_frame", footprint_frame_, std::string("base_link_"));
        private_nh_.param("optical_frame", optical_frame_, std::string("camera_link_"));
        
        private_nh_.param("init_angle", init_angle, 0.0);
        
        

        // get slam thread pointer
        mpTracker_ = mpSystem_->GetTracker();
        mpFrameDrawer_ = mpSystem_->GetFrameDrawer();
        mpMapDrawer_ = mpSystem_->GetMapDrawer();
        mpPclMapper_ = mpSystem_->GetPointCloudMapper();
    }




    void MessageUtils::publishOdometry(cv::Mat matTcw)
    {
        ros::Time current_frame_time_ = ros::Time::now();
    
    
                //轮式里程计!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            tf::StampedTransform  Tob;
            
            tf::Quaternion qua = tf::createQuaternionFromRPY(0, 0, init_angle*M_PI/180.0);//只绕着z旋转 //(0.0,0.0,0.598181195731,0.801360878178);//只绕着z旋转
            tf::Vector3 tra(0.387218208555,6.4243182945,0.14192066);
            
      //x: 0.387218208555
      //y: 6.4243182945
     // z: 0.14192066
    //orientation: 
    //  x: 0.0
     // y: 0.0
    //  z: 0.598181195731
    //  w: 0.801360878178

            listener_.waitForTransform("/odom", "/base_link", ros::Time(0), ros::Duration(3.0));//tar<-src
            listener_.lookupTransform("/odom", "/base_link", ros::Time(0), Tob);//tar<-src
            
            
           // ROS_INFO_STREAM("Tob"<<Tob);
            
            
           // odom_msgs.header.stamp = current_frame_time_;
            odom_msgs.header.frame_id = odom_frame_;
            odom_msgs.child_frame_id = "/base_link";
            tf::poseTFToMsg(tf::Transform(qua,tra).inverse()*Tob, odom_msgs.pose.pose);//
            //odom_pub_1.publish(odom_msgs);//只在关键帧发送odom
            
            
            
            
            
        if (!use_odom_pub_)
        {
            ROS_WARN("!use_odom_pub_");
            return;
        }
        
        // odom_frame<--optical_frame, this is estimated by ORB-SLAM2
        //cv::Mat matTcw = mpTracker_->mCurrentFrame.mTcw.clone();//获取世界到相机的位姿变换
        
        if (mpTracker_->mState==ORB_SLAM2::Tracking::LOST)
        {
            ROS_WARN("ORB_SLAM2 has lost tracking.");
            return;
        }
        if (mpTracker_->mState!=ORB_SLAM2::Tracking::OK)
        {
            ROS_WARN("mpTracker_->mState!=ORB_SLAM2::Tracking::OK");
            return;
        }
            
        if (cv::determinant(matTcw) < 1e-4)
        {
            ROS_WARN("ORB_SLAM is tracking but the pose is ill state.");
            ROS_WARN_STREAM ("Abnormal pose\n" << matTcw);
            return;
        }


        


        
        

        tf::Transform tf_transform = TransformFromMat(matTcw,false);
        
        tf::Stamped<tf::Transform> Tcw(tf_transform, current_frame_time_, optical_frame_);//相机到世界
        

        //tf::Stamped<tf::Transform> Twb = Tcw;

        
        
        //原版代码
        // footprint_frame<--optical_frame, static transform
        

//        tf::Stamped<tf::Transform> Tbc(tf::Transform(), ros::Time::now(), optical_frame_);//底盘到相机
//        Tbc.setIdentity();
//        if ( !getTransformedPose(Tbc, footprint_frame_) )
//        {
//            ROS_WARN("!getTransformedPose(Tbc, footprint_frame_)");
//            return;
//        }

        tf::Quaternion quat = tf::createQuaternionFromRPY(-1.5708, 0, -1.5708);//只绕着z旋转
        tf::Vector3 tran(0,0,0.326921);
        tf::Stamped<tf::Transform> Tbc(tf::Transform(quat,tran),current_frame_time_, optical_frame_);//底盘到相机
        
        // visual odom transform to footprint odom



        tf::Stamped<tf::Transform> Twb(Tbc*Tcw.inverse()*Tbc.inverse(), current_frame_time_, odom_frame_);//世界到底盘，下面会发布出去
        Twb.setRotation(Twb.getRotation().normalized());  //! necessary, otherwise Rviz will complain unnormalized


        // publish odom message
//        odom_msgs.header.stamp = ros::Time::now();
//        odom_msgs.header.frame_id = odom_frame_;
//        odom_msgs.child_frame_id = footprint_frame_;
//        tf::poseTFToMsg(Twb, odom_msgs.pose.pose);



        // boardcast tf transform
        if (use_tf_)
        {
            // odom<--footprint ****发布footprint到odom的转换*****
            //发布TF
            tf::StampedTransform stamped_trans(Twb,current_frame_time_, odom_frame_, footprint_frame_);
            broadcaster_.sendTransform(stamped_trans);

            tf::Transform Tmo;
            Tmo.setIdentity();
            tf::StampedTransform trans(Tmo, current_frame_time_, map_frame_, odom_frame_); //map和odom重合
            broadcaster_.sendTransform(trans);
            

            //发布轨迹线条

//            path.header.stamp=odom_msgs.header.stamp;
//            path.header.frame_id=odom_frame_;
            
//            geometry_msgs::PoseStamped this_pose_stamped;
//            this_pose_stamped.pose = odom_msgs.pose.pose;
//            this_pose_stamped.pose.position.z += 0.326921;

//            this_pose_stamped.header.stamp=odom_msgs.header.stamp;
//            this_pose_stamped.header.frame_id=odom_frame_;
//            path.poses.push_back(this_pose_stamped);

//            path_pub.publish(path);


        }

//        nav_msgs::Path poses_path;
//        vector<cv::Mat> all_poses = mpPclMapper_->get_all_poses();
//        poses_path.header.stamp=odom_msgs.header.stamp;
//        poses_path.header.frame_id=odom_frame_;

//        for(int i=0;i<all_poses.size();i++)
//        {
//            tf::Transform tf_Tcw = TransformFromMat(all_poses[i],false);

//            geometry_msgs::PoseStamped this_pose_stamped;

//            tf::poseTFToMsg(Tbc*tf_Tcw.inverse(), this_pose_stamped.pose);//这里要少乘以Tbc.inverse()

//            this_pose_stamped.header.stamp=odom_msgs.header.stamp;
//            this_pose_stamped.header.frame_id=odom_frame_;


//            poses_path.poses.push_back(this_pose_stamped);
//        }

//        path_pub.publish(poses_path);



         //外参确定的情况下不用动态更新，程序一开始会读取yaml文件配置
        // update SLAM PointCloudMapping thread extrinsic matrix
//        Eigen::Matrix3d matRbc;
//        Eigen::Vector3d mattbc;
//        tf::matrixTFToEigen(Tbc.getBasis(), matRbc);
//        tf::vectorTFToEigen(Tbc.getOrigin(), mattbc);
//        mpPclMapper_->updateTbc(matRbc, mattbc);

        
        // debug log
//        Eigen::Isometry3d ematTwb;
//        tf::poseTFToEigen(Twb, ematTwb);
//        ROS_DEBUG_STREAM("SLAM output Twc: \n" << matTcw.inv());
//        ROS_DEBUG_STREAM("Footprint output Twb: \n" << ematTwb.matrix());
    }


    void MessageUtils::publishFrame()//发布带关键点的灰度图
    {
        if(!use_frame_pub_)
        {
            ROS_WARN("!use_frame_pub_");
            return;
        }

        // draw current frame
        cv::Mat frame_mat = mpFrameDrawer_->DrawFrame();

        // publish
        std_msgs::Header h;
        h.stamp = ros::Time::now();
        h.frame_id = optical_frame_;
        cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage(h, sensor_msgs::image_encodings::BGR8, frame_mat));
        frame_pub_.publish(cv_ptr->toImageMsg());
    }
    
    
    bool MessageUtils::publishPointCloud(const sensor_msgs::ImageConstPtr& msgRGB)
    {

        ros::Time current_frame_time_ = ros::Time::now();




        sensor_msgs::Image newRGB = *msgRGB;
        newRGB.header.stamp = current_frame_time_;

        //ROS_INFO("run publishPointCloud");

        //因为不一定每一帧都是关键帧，所以大量的会return,只有插入关键帧点云才会更新
        if (!use_pcl_pub_ || mpPclMapper_->getPointCloudMapUpdatedFlag()==0)
        {

            return false;
        }



        //ROS_INFO("run publishPointCloud flag ok");

        // get Tbc, Matrix4d
//        Eigen::Matrix4d Tbc = Eigen::Matrix4d::Identity();
//        if (!getTransformedPose(Tbc, footprint_frame_, optical_frame_))
//        {
//            ROS_WARN("!getTransformedPose(Tbc, footprint_frame_, optical_frame_)");
//            return;
//        }

        // update point cloud

        vector<cv::Mat> all_poses = mpPclMapper_->get_all_poses();//获取所有关键帧的位姿

        mpPclMapper_->getCurrentCloud(pcl_plane_);


        mpPclMapper_->getGlobalCloud(pcl_map_);

        

        // transform point cloud: footprint_base<--optical_base
        //pcl_map_->width = pcl_map_->size();  //! Occassionally it will conflict, force to equal
        //pcl_plane_->width = pcl_plane_->size();
//        ROS_WARN("Height: %d, Width: %d, Size: %d", pcl_map_->height, pcl_map_->width, pcl_map_->size());
//        ROS_WARN("Point cloud update!");

        // segment plane




        if (use_tf_)
        {
            tf::Transform tf_transform = TransformFromMat(all_poses[all_poses.size()-1],false);

            tf::Stamped<tf::Transform> Tcw(tf_transform, current_frame_time_, optical_frame_);//相机到世界

            tf::Quaternion quat = tf::createQuaternionFromRPY(-1.5708, 0, -1.5708);//只绕着z旋转
            tf::Vector3 tran(0,0,0.326921);
            tf::Stamped<tf::Transform> Tbc(tf::Transform(quat,tran),current_frame_time_, optical_frame_);//底盘到相机
//            if ( !getTransformedPose(Tbc, footprint_frame_) )
//            {
//                ROS_WARN("!getTransformedPose(Tbc, footprint_frame_)");
//                return false;
//            }


            //0 0 1
            //-1 0 0
            //0 -1 0


//            cout << "Tbc0=" << Tbc.getBasis()[0][0]  << endl;
//            cout << "Tbc1=" << Tbc.getBasis()[0][1]  << endl;
//            cout << "Tbc2=" << Tbc.getBasis()[0][2]  << endl << endl;



            tf::Stamped<tf::Transform> Twb(Tbc*Tcw.inverse()*Tbc.inverse(), current_frame_time_, odom_frame_);//世界到底盘，下面会发布出去
            Twb.setRotation(Twb.getRotation().normalized());  //! necessary, otherwise Rviz will complain unnormalized



            // publish odom message
            odom_msgs.header.stamp = current_frame_time_;
            odom_msgs.header.frame_id = odom_frame_;
            odom_msgs.child_frame_id = footprint_frame_;
            tf::poseTFToMsg(Twb, odom_msgs.pose.pose);
            odom_pub_.publish(odom_msgs);//只在关键帧发送odom
            
            
            
            

            
            



            frame_pub_.publish(newRGB); //发布图像


            // boardcast tf transform
            // odom<--footprint ****发布footprint到odom的转换*****
            //发布TF
//            tf::StampedTransform stamped_trans(Twb, current_frame_time_, odom_frame_, footprint_frame_);
//            broadcaster_.sendTransform(stamped_trans);



            nav_msgs::Path poses_path;
            poses_path.header.stamp=current_frame_time_;
            poses_path.header.frame_id=odom_frame_;

            for(int i=0;i<all_poses.size();i++)
            {
                tf::Transform tf_Tcw = TransformFromMat(all_poses[i],false);

                geometry_msgs::PoseStamped this_pose_stamped;

                //tf::poseTFToMsg(Tbc*tf_Tcw.inverse(), this_pose_stamped.pose);//这里要少乘以Tbc.inverse()
                tf::poseTFToMsg(Tbc*tf_Tcw.inverse()*Tbc.inverse(), this_pose_stamped.pose);

                this_pose_stamped.header.stamp=current_frame_time_;
                this_pose_stamped.header.frame_id=odom_frame_;

                poses_path.poses.push_back(this_pose_stamped);
            }

            path_pub.publish(poses_path);
        }

        // sensor_msgs::PointCloud2
        
        //if(kf_cnt%10==0)
        //{
            sensor_msgs::PointCloud2 pcl2_msgs;
            pcl::toROSMsg(*pcl_map_, pcl2_msgs);//pcl转ROS消息
            pcl2_msgs.header.stamp = current_frame_time_;
            pcl2_msgs.header.frame_id = map_frame_;
            pcl2_pub_.publish(pcl2_msgs);
        //}
        

        sensor_msgs::PointCloud2 cur_pcl2_msgs;
        pcl::toROSMsg(*pcl_plane_, cur_pcl2_msgs);//pcl转ROS消息
        cur_pcl2_msgs.header.stamp = current_frame_time_;
        cur_pcl2_msgs.header.frame_id = optical_frame_; //footprint_frame_; //footprint_frame_; //map_frame_; //改为相对于底盘的
        cur_pcl2_pub_.publish(cur_pcl2_msgs);
        

        kf_cnt++;

        mpPclMapper_->setPointCloudMapUpdatedFlag(0);//点云更新标志清0

        return true;
    }
    
    
    
    
    //以下是工具函数
    

    bool MessageUtils::getTransformedPose(tf::Stamped<tf::Transform> &output_pose, const string &target_frame, const double &timeout)
    {
        std::string source_frame = output_pose.frame_id_;
        try
        {
            if ( !listener_.waitForTransform(target_frame, source_frame, ros::Time::now(), ros::Duration(timeout)) )
            {
                ROS_ERROR("Wait transform timeout between: [%s]<--[%s], timeout %f s",
                        target_frame.c_str(), source_frame.c_str(), timeout);
                return false;
            }
            listener_.transformPose(target_frame, output_pose, output_pose);
        }
        catch (tf::TransformException &e)
        {
            ROS_ERROR("Fail to find the transform between: [%s]<--[%s]: %s",
                      footprint_frame_.c_str(), optical_frame_.c_str(), e.what());
            return false;
        }
        return true;
    }


//    bool MessageUtils::getTransformedPose(Eigen::Matrix4d &output_mat, const string &target_frame, const string &source_frame, const double &timeout)
//    {
//        Eigen::Isometry3d output_matT(output_mat);
//        tf::Transform initTfPose;
//        tf::poseEigenToTF(output_matT, initTfPose);
//        tf::Stamped<tf::Transform> To(initTfPose, ros::Time::now(), source_frame);
        
//        // call override function
//        if (!getTransformedPose(To, target_frame, timeout))
//        {
//            ROS_WARN("!getTransformedPose(To, target_frame, timeout)");
//            return false;
//        }
        
//        tf::poseTFToEigen(To, output_matT);
//        output_mat = output_matT.matrix();
//        return true;
//    }
    
    tf::Transform MessageUtils::TransformFromMat(cv::Mat position_mat,bool need_rotate)
    {
      cv::Mat rotation(3,3,CV_32F);
      cv::Mat translation(3,1,CV_32F);

      rotation = position_mat.rowRange(0,3).colRange(0,3);
      translation = position_mat.rowRange(0,3).col(3);


      tf::Matrix3x3 tf_camera_rotation (rotation.at<float> (0,0), rotation.at<float> (0,1), rotation.at<float> (0,2),
                                        rotation.at<float> (1,0), rotation.at<float> (1,1), rotation.at<float> (1,2),
                                        rotation.at<float> (2,0), rotation.at<float> (2,1), rotation.at<float> (2,2)
                                       );

      tf::Vector3 tf_camera_translation (translation.at<float> (0), translation.at<float> (1), translation.at<float> (2));
        
      

      //注意这里会把ORB的Z轴向前转化为X轴向前

      if(need_rotate)
      {
          //Coordinate transformation matrix from orb coordinate system to ros coordinate system
          const tf::Matrix3x3 tf_orb_to_ros (0, 0, 1,
                                            -1, 0, 0,
                                             0,-1, 0);

          //Transform from orb coordinate system to ros coordinate system on camera coordinates
          tf_camera_rotation = tf_orb_to_ros*tf_camera_rotation;
          tf_camera_translation = tf_orb_to_ros*tf_camera_translation;

          //Inverse matrix
          tf_camera_rotation = tf_camera_rotation.transpose();
          tf_camera_translation = -(tf_camera_rotation*tf_camera_translation);

          //Transform from orb coordinate system to ros coordinate system on map coordinates
          tf_camera_rotation = tf_orb_to_ros*tf_camera_rotation;
          tf_camera_translation = tf_orb_to_ros*tf_camera_translation;
      }


      return tf::Transform (tf_camera_rotation, tf_camera_translation);
    }
}
