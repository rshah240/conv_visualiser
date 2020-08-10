from cnn_visualiser.gradient_visualiser import GradVisualiser
from tensorflow.keras import losses
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class VanillaGradients(GradVisualiser):
    def __init__(self,model,input_image = None):
        layer_name = model.layers[-1].name
        super().__init__(model,layer_name,input_image)

    @staticmethod
    @tf.function
    def transform_to_normalized_grayscale(tensor):
        grayscale_tensor = tf.reduce_sum(tensor,axis = -1)

        normalized_tensor = tf.cast(255*tf.image.per_image_standardization(grayscale_tensor),
                                    tf.uint8)
        return normalized_tensor

    def vanilla_gradients(self,class_label):

        if len(self.input_image.shape) != 4:
            self.input_image = np.expand_dims(self.input_image,axis = 0)
        num_classes = self.model.output.shape[1]
        expected_output = tf.one_hot(class_label*self.input_image.shape[0],
                                    num_classes)
        expected_output = tf.expand_dims(expected_output,axis=0)
        self.input_image = tf.cast(self.input_image, tf.float32)

        with tf.GradientTape() as g:
            g.watch(self.input_image)
            output = self.model(self.input_image)
            loss = losses.categorical_crossentropy(expected_output,output)

        grads = g.gradient(loss,self.input_image)
        grads = VanillaGradients.transform_to_normalized_grayscale(tf.abs(grads)).numpy()
        grads = np.reshape(grads,(self.input_image.shape[1],self.input_image.shape[2],1))
        return grads

    def generate_filter_pattern(self,filter_index,input_height,input_width,
                         layer_name,steps = 40,input_channels = 3):
        """
        Function for pattern generation of intermediate feature maps of convolution layers using gradient ascent

        if custom layer is not passed function will automatically use layer_name attributes of this class
        return: activation map"""
    
        inter_model = Model(self.model.input,self.model.get_layer(layer_name).output)
        with tf.GradientTape() as g:
            g.watch(self.model.input)
            loss = K.mean((inter_model(self.model.input)[:,:,:,filter_index]))
        grads = g.gradient(loss,self.model.input)[0]
        #Normalizing Gradients
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        function = K.function([self.model.input],[loss,grads])
        input_img_data = np.random.random((1,input_height,input_width,input_channels))*20 + 128
        ascent_step = 1
        for i in range(steps):
            loss_value, grads_value = function([input_img_data])
            if loss_value <= 0:
                break
            input_img_data += grads_value*ascent_step

        return GradVisualiser.deprocess_image(input_img_data[0])








