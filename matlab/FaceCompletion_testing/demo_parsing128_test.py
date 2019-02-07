import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

caffe_root='../../'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# define model
model_def=caffe_root+'matlab/FaceCompletion_testing/model/Model_parsing.prototxt'
model_weights=caffe_root+'matlab/FaceCompletion_testing/model/Model_parsing.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

# Run the test
image_file = "TestImages/182659.png"
image = np.array(Image.open(image_file))
#image = np.dstack((image,image,image)) 
print image.shape


# preprocessing the image to fit the net requirement
input_ = image / 255.0
input_ = input_ * 2 - 1
input_ = input_.transpose(2, 1, 0)
input_ = input_[np.newaxis, ...]

net.blobs['data'].reshape(*input_.shape)
net.blobs['data'].data[...] = input_
output = net.forward()
scores = output['conv_decode0'][0]
scores = scores.transpose(0, 2, 1)
segmentation = scores.argmax(0)

segmentation_rgb = np.zeros(image.shape, dtype=np.uint8)

colors = [
    [0, 0, 255],
    [255, 255, 0],
    [160, 32, 240],
    [218, 112, 214],
    [210, 105, 30],
    [94, 38, 18],
    [0, 255, 0],
    [156, 102, 31],
    [0, 0, 0],
    [255, 127, 80],
    [255, 0, 0]
]

for i in range(11):
    segmentation_rgb[np.where(segmentation == i)] = colors[i]


plt.figure(1)
plt.subplot(121)
plt.imshow(image)
plt.axis('off')
plt.subplot(122)
plt.imshow(segmentation_rgb)
plt.axis('off')
plt.show()

import scipy.misc
scipy.misc.imsave('outfile.jpg', segmentation_rgb)



