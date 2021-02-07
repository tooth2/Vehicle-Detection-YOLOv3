# Vehicle-Detection-YOLOv3

Implementation of Vehicle Detection using YOLOv3. The parameters used are mainly the ones suggested by the authors. The author of YOLOv3 have made available a set of pre-trained weights that enable the YOLOv3 to recognize 80 different objects in images and videos based on [COCO Data set](http://cocodataset.org/#home). 

1. [C++ Implementation](https://github.com/tooth2/YOLOv3-Object-Detection)
2. [PyTorch Implementation](https://github.com/tooth2/YOLOv3-Pytorch)
3. [Tensorflow/Kera Implementation]

## Implementation Approach 

### Data Set/Model
* Image: Kitti vehicel image data. 
* Label: [COCO Data set](http://cocodataset.org/#home). 
* YOLO Model Parameter: [YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)

### YOLO 
YOLO, a very fast detection framework that is shipped with the OpenCV library. By using this, a single neural network is applied to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.
* Input image is divided into a 13x13 grid of cells. Based on the size of the input image, the size of these cells in pixels varies. In the C++ implementation, a size of (`inpWidth`, `inpHeight`) = 416 x 416 pixels, leading to a cell size of 32 x 32 pixels.
* Each cell is then used for predicting a set of bounding boxes. For each bounding box, the neural network also predicts the confidence that the bounding box encloses a particular object as well as the probability of the object belonging to a particular class (taken from the COCO dataset).
* Non-maximum Suppression 
Every bounding box predicted by YOLOv3 is associated with a confidence score. The parameter 'confThreshold' is used to remove all bounding boxes with a lower score value. Then, a non-maximum suppression is applied to the remaining bounding boxes. A non-maximum suppression is used to eliminate bounding boxes with a low confidence level as well as redundant bounding boxes enclosing the same object. The NMS procedure is controlled by the parameter ‘nmsThreshold‘.

2. Prepare the Model
[yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) contains the pre-trained network’s weights and has been made available by the authors of YOLO.
The file 'yolov3.cfg' containing the network configuration is available here and the coco.names file which contains the 80 different class names used in the COCO dataset.
```code
    // load neural network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
```
After loading the network, the DNN backend is set to DNN_BACKEND_OPENCV. If OpenCV is built with Intel’s Inference Engine, DNN_BACKEND_INFERENCE_ENGINE should be used instead. The target is set to CPU in the code, as opposed to using DNN_TARGET_OPENCL, which would be the method of choice if a (Intel) GPU was available.

3. Generate 4D Blob from Input Image
"Blob" is the standard array, unified memory interface and a wrapper over the actual data being processed. "Blob" also provides synchronization capability between the CPU and the GPU. In OpenCV, blobs are stored as 4-dimensional cv::Mat array with NCHW dimensions order. 
* N: number of images in the batch
* H: height of the image
* W: width of the image
* C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)

```// generate 4D blob from input image
   cv::Mat blob;
   double scalefactor = 1/255.0;
   cv::Size size = cv::Size(416, 416);
   cv::Scalar mean = cv::Scalar(0,0,0);
   bool swapRB = false;
   bool crop = false;
   cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);
```
An image loaded from the file is passed through blobFromImage function to be converted into an input block for the neural network. The pixel values are scaled with a scaling factor of 1/255 to a target range of 0 to 1. It also adjusts the size of the image to the specified size of (416, 416, 416) without cropping. 

4. Run a Single Forward-Pass through the Network
The output blob is passed as input to the network. Then, a forward function of OpenCV is executed to perform a single forward-pass thru network and obtain a list of predicted bounding boxes as output from the network. These boxes go through a post-processing step to filter out those with low confidence values.
OpenCV function `getUnconnectedOutLayers`, which gives the names of all unconnected output layers, which are in fact the last layers of the network. 

The result of the forward pass and thus the output of the network is a vector of size C (the number of blob classes) with the first four elements in each class representing the center in x, the center in y as well as the width and height of the associated bounding box. The fifth element represents the confidence that the respective bounding box actually encloses an object. The remaining elements of the matrix are the confidence associated with each of the classes contained in the coco.cfg file. Each box is assigned to the class corresponding to the highest confidence.

To visualize the result, green rectagle encloses idenetied bounding boxes along with label and confidance score. 

### Result 
![input image](/images/img1.png)
![detected output](/images/Object_classification.png)

YOLOv3 is fast, under cpu it took 2sec and this works well even in 30frame/s live video as well. Even though this model is useful to identify pedestrian, car, track , motorcycle on the road, however the bouding boxes have their limits: drawing boxes vehicles on a curvy road, forest or trees' or vechicles shadow. It is not easy to convery the true shape of an object. So that bouding boxes can acheive *"partial"* scene understanding. 

### Reference
* [Blobs](http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html)
* [YOLO Weights](https://pjreddie.com/media/files/yolov3.weights)
* [COCO Data set](http://cocodataset.org/#home)
* [NCHW :Intel math kernel for deep neural network](https://oneapi-src.github.io/oneDNN/understanding_memory_formats.html)
* [Yolov3:An Incremental Improvement](https://arxiv.org/abs/1804.02767)

### Next Step
- [x] YOLO tesnforflow 
- [x] YOLOv3 in Pytorch
- [x] YOLOv3 C++ in openCV
- [ ] SSD(Single shot detection) 
- [ ] Semantic Segmentation 
