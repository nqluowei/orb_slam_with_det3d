ros python3 tf 配置：

https://blog.csdn.net/liam_dapaitou/article/details/107656486

https://github.com/ros/geometry/issues/213

编译方法：

cd ~/ai/orb_slam2_dense_ws/src/orb_slam2_dense/build
make -j1

cd ~/ai/orb_slam2_dense_ws
catkin_make -j1

运行方法：
终端1：
roscore

终端2：
cd ~/ai/object_detection/mmdetection3d
conda activate py37
source ~/ai/test_env_ws/devel/setup.bash
python3 ros_pcd.py

终端三：cd ~/ai/orb_slam2_dense_ws
source devel/setup.bash
roslaunch orb_slam2_dense tum_pioneer.launch
