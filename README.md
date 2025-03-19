# Deep Learning 人脸和手势识别系统

这是一个基于OpenCV和深度学习的人脸和手势识别系统，支持实时视频流处理、人脸检测、人脸识别、手势识别等功能。

## 功能特点

- 实时人脸检测和识别
- 人脸关键点检测（眼睛、嘴巴等）
- 手势识别（支持OK手势、握拳手势等）
- GPU加速支持
- 实时性能监控
- 完整的日志系统

## 系统要求

- Python 3.8+
- CUDA支持（可选，用于GPU加速）
- 摄像头设备

## 依赖包

- OpenCV (opencv-contrib-python)
- dlib
- face_recognition
- mediapipe
- imutils
- numpy

## 安装步骤

1. 克隆项目
```bash
git clone https://github.com/yourusername/Deep_Learning.git
cd Deep_Learning
```

2. 创建并激活conda环境
```bash
conda create -n deep_learning python=3.8
conda activate deep_learning
```

3. 安装依赖包
```bash
conda install -c conda-forge opencv-contrib-python dlib face_recognition mediapipe imutils numpy
```

4. 下载必要的模型文件
- 下载人脸关键点检测模型：[shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- 解压并放置到 `demo/resource/data/` 目录

## 项目结构

```
demo/
├── config/          # 配置文件目录
├── detectors/       # 检测器模块
│   ├── face_detector.py    # 人脸检测器
│   ├── hand_detector.py    # 手势检测器
│   └── detector_factory.py # 检测器工厂类
├── models/         # 模型处理模块
│   ├── handle_eyes.py      # 眼睛状态检测
│   ├── handle_mouth.py     # 嘴巴状态检测
│   └── handle_gesture.py   # 手势识别处理
├── resource/       # 资源文件目录
│   ├── data/      # 模型数据
│   └── image/     # 图片资源
└── utils/         # 工具类目录
    ├── camera.py  # 摄像头控制
    └── logger.py  # 日志管理
```

## 使用方法

1. 运行主程序
```bash
python demo/main.py
```

2. 使用快捷键
- 按 'q' 键退出程序

## 配置说明

### 摄像头配置
- 分辨率：640x480
- 帧率：30fps

### 检测器配置
- 人脸检测置信度：0.5
- 手势检测置信度：0.5

### GPU加速
系统会自动检测是否支持CUDA，如果支持则启用GPU加速。

## 性能优化

1. 降低分辨率
- 默认使用640x480分辨率，可以根据需要调整

2. 跳帧处理
- 每2帧处理1帧，减少计算量

3. GPU加速
- 支持CUDA加速，提高处理速度

## 常见问题

1. 摄像头无法打开
- 检查摄像头是否被其他程序占用
- 检查摄像头驱动是否正确安装

2. 性能问题
- 降低分辨率
- 增加跳帧数
- 启用GPU加速

3. 模型文件缺失
- 确保已下载并正确放置模型文件
- 检查文件路径是否正确

## 日志系统

系统使用Python的logging模块记录日志，日志文件保存在 `app.log` 中，包含以下信息：
- 程序启动和退出
- 摄像头状态
- 检测器状态
- 错误信息

## 开发计划

- [ ] 添加更多手势识别
- [ ] 优化性能监控
- [ ] 添加Web界面
- [ ] 支持多摄像头
- [ ] 添加数据导出功能

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 作者：Your Name
- 邮箱：your.email@example.com
- 项目主页：https://github.com/yourusername/Deep_Learning 