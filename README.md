代码完整压缩包（包括大文件）：

链接: https://pan.baidu.com/s/1Pl7-zV5j5yp9uMTBqoVduQ?pwd=3aws 提取码: 3aws 复制这段内容后打开百度网盘手机App，操作更方便哦

Complete code compression package (including large files):

Link: https://pan.baidu.com/s/1Pl7-zV5j5yp9uMTBqoVduQ?pwd=3aws Extract code: 3aws copies this content and opens the Baidu Wangpan mobile app, which is more convenient to operate

ros python3 tf 配置： / ros python3 tf config：

https://blog.csdn.net/liam_dapaitou/article/details/107656486

https://github.com/ros/geometry/issues/213

编译方法： / Compile method:

cd ~/ai/orb_slam2_dense_ws/src/orb_slam2_dense/build

make -j1

cd ~/ai/orb_slam2_dense_ws

catkin_make -j1

运行方法： / Run method：

终端1： / Terminal 1:

roscore

终端2： / Terminal 2：

cd ~/ai/object_detection/mmdetection3d

conda activate py37

source ~/ai/test_env_ws/devel/setup.bash

python3 ros_pcd.py

终端3：/ Terminal 3:

cd ~/ai/orb_slam2_dense_ws

source devel/setup.bash

roslaunch orb_slam2_dense tum_pioneer.launch
