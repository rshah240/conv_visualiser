# conv_visualiser
Python Package for Visualising Convolution Neural Network Features and Intermediate Activation of Tensorflow 2.0 Models

# Paper Reference
https://arxiv.org/abs/1610.02391

# Requirements
Tensorflow 2.0 or newer <br/>
OpenCv <br/>
Matplotlib <br/>
numpy <br/>

## Installation
```bash
pip install conv_visualiser
```
## Usage
```python
from cnn_visualiser import GRAD_CAM
from cnn_visualiser import GradVisualiser
from cnn_visualiser import IntermediateActivations
from cnn_visualiser import VanillaGradients
from cnn_visualiser import IntegratedGradients


from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = "./Input_Images/cat_dog.png" #path to image
vgg_model = vgg16.VGG16(weights = 'imagenet',input_shape=(224,224,3),include_top=True)
img = image.load_img(path,target_size=(224,224)) #make sure the dimensions of the image matches with the input_shape
img = image.img_to_array(img)
img = preprocess_input(img) #make sure the image is preprocessed
img = np.expand_dims(img,axis = 0)# adding a batch dimension


grad_cam = GRAD_CAM(model=vgg_model,input_image=img,layer_name='block5_conv3')
superimposed_image = grad_cam.get_superimposed(path) #heatmap superimposed on the original image
heatmap = grad_cam.get_heat_map()
guided_grads_cam = grad_cam.get_guided_grad_cam()

grad_visualiser = GradVisualiser(model=vgg_model,layer_name='block4_conv3',input_image=img)
guided_grads = grad_visualiser.guided_gradients()


vg = VanillaGradients(model = vgg_model,input_image = img)
filter_pattern = vg.generate_filter_pattern(filter_index=10,layer_name = 'block2_conv1',input_height=224,input_width=224)
grads = vg.vanilla_gradients(class_label = 245)


ia = IntermediateActivations(model=vgg_model,layer_names = ['block2_conv1','block2_conv2'])
ia.display_grid(input_image=img) #Matplotlib Plot

ia.display_single_channel(input_image = img)

ig = IntegratedGradients(model = vgg_model,input_image=img)
attributions = ig.integrated_gradients()[0] #to save  attributions
ig.display_plot_img_attributions()# to display plot img
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## Outputs
HeatMap SuperImposed Image Output













![Image of SuperImposed Image Output](https://github.com/rshah240/conv_visualiser/blob/master/Output/superimposed_image.png)











Intermediate Filter Pattern Output 


















![Image of Filter Pattern Output](https://github.com/rshah240/conv_visualiser/blob/master/Output/filter_pattern.png)










Guided Gradients Output
















![Image of Guided Gradients output](https://github.com/rshah240/conv_visualiser/blob/master/Output/guided_grads.png)













Guided Gradients with heatmap overlay according to GRAD CAM Algorithm











![Image of Guided Gradients Heatmap output](https://github.com/rshah240/conv_visualiser/blob/master/Output/guided_grads_cam.png)














Vanilla Gradients output














![Image of Vanilla Gradients Output](https://github.com/rshah240/conv_visualiser/blob/master/Output/vanilla_grads.png)





## Tutorial
please refer tutorial.ipynb file for more information
