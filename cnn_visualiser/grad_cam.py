import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from cnn_visualiser.gradient_visualiser import GradVisualiser
import cv2

class GRAD_CAM(GradVisualiser):
    def __init__(self,model,input_image,layer_name):
        """Class having functionalities mentioned in Grad CAM paper
        
        resource: https://arxiv.org/abs/1610.02391

        :arg
        model = Input model
        input_image = Input_image
        layer_name = name of the last convolution layer"""
        super().__init__(model=model,input_image=input_image,layer_name = layer_name)
        self.gbModel = self.build_guided_model()

    def get_heat_map(self,class_label = None):
        """Generating heat map using GRAD CAM algorithm useful when the model contains relu activation function
        :arg
        class_label Perform heat map computation based on a particular class
        :return heatmap"""


        if len(self.input_image.shape) != 4:
            self.input_image = np.expand_dims(self.input_image,axis = 0) #adding the batch dimension
        predictions = self.model.predict(self.input_image)

        if class_label == None:
            class_label = np.argmax(predictions[0])
        inter_model = Model(self.gbModel.input,[self.gbModel.get_layer(self.layer_name).output,
                                              self.gbModel.output[:,class_label]])

        self.input_image = tf.convert_to_tensor(self.input_image)
        with tf.GradientTape() as g:
            g.watch(self.input_image)
            conv_output,class_output = inter_model(self.input_image)

        #Positive gradients would have positive impact on the inputs for classification
        grads = g.gradient(class_output,conv_output) #Guided Backpropagation
        #Global Average Pooling
        pooled_grads = K.mean(grads,axis = (0,1,2))
        conv_output = conv_output.numpy()[0] #removing the batch dimension
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_output[:,:,i]*= pooled_grads[i]

        heatmap = np.mean(conv_output,axis = -1)
        heatmap = np.maximum(heatmap,0)
        heatmap /= np.max(heatmap)
        return heatmap

    def get_guided_grad_cam(self):
        """Generating guided grad cam"""
        grads = super().guided_gradients(deprocess=False)
        heatmap = self.get_heat_map()
        heatmap = cv2.resize(heatmap,(self.input_image.shape[1],self.input_image.shape[2]))
        heatmap = np.expand_dims(heatmap,axis = 2)
        #Element wise multiplication as mentioned in the paper
        grad_cam = grads*heatmap
        return GradVisualiser.deprocess_image(grad_cam)

    def get_superimposed(self,image_path,class_label = None):
        img = cv2.imread(image_path)
        heatmap = self.get_heat_map(class_label)
        #heatmap = np.squeeze(heatmap,axis = 0)
        heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
        heatmap = np.uint8(heatmap * 255)
        heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
        superimposed_img = heatmap*0.4 + img
        return superimposed_img


















