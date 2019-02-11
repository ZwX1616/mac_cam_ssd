# Detect objects in webcam using deep learning models

_example of using mxnet c api for prediction_
  <br />
  <br />
![](https://github.com/ZwX1616/mac_cam_ssd/blob/master/misc/demo.jpg)

  (SSD is used in the demo, but other models like yolo can be used too)
  <br />
  <br />
  <br />
(tested on macOS High Sierra with Xcode)<br />
requirements:<br />
1. webcam <br />
2. opencv 4 <br />
3. mxnet 1.3 built from source <br />
4. (optional) gluoncv(python)
<br />
<br />
note:<br />
1. frameworks libmxnet, libopencv_imgcodecs.4.x.x, libopencv_imgproc.4.x.x, libopencv_videoio.4.x.x, libopencv_core.4.x.x, libopencv_highgui.4.x.x need to be added to project in Xcode<br />
2. you can either use your own model (for example, some model built and trained in python), or use pretrained models from gluoncv (gluoncv.model_zoo.get_model, gluoncv.utils.export_block)<br />
3. double check input data channel layout and whether it is normalized to make sure they agree with the model
<br />
<br />
