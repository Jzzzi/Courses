## 机器狗信息

4-1 NC077 4-2 YN275

4-1:mi@10.0.0.189 4-2:mi@10.0.0.245

连接密码：123


## 编译方法

输入如下命令build

```
cd ~/VisualLocate
colcon build
source ~/VisualLocate/install/setup.bash
```

然后用

```
ros2 run VisualLocate ...
```

运行编写好的程序

已经编写在一个bash中，可以通过下面代码build

```
source ~/VisualLocate/bash/build.bash
```

## realsense相机

在一个命令行中执行

```
ros2 launch realsense2_camera on_dog.py
```

之后不要关闭该命令行，打开新的命令行一次运行

```
ros2 lifecycle set /camera/camera configure
ros2 lifecycle set /camera/camera activate
```

已经将之编写在一个bash文件内,可以直接执行下面命令开启相机（realsense）

```
source ~/VisualLocate/bash/realsense2_camera.bash
```

可以用指令查看节点topic_type

```
ros2 topic type /camera/infra1/image_rect_raw
```

强烈建议不要使用bash脚本！容易有莫名其妙的bug

消息的encoding格式为mono8

## 鱼眼和RGB相机

```
ros2 launch camera_test stereo_camera.py
ros2 lifecycle set /stereo_camera configure
ros2 lifecycle set /stereo_camera activate
# ros2 lifecycle set /stereo_camera deactivate

```

编写在

```
source ~/VisualLocate/bash/stereo_camera.bash
```