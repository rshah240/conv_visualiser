from cnn_visualiser.gradient_visualiser import GradVisualiser
from tensorflow.keras import losses
import numpy as np
import tensorflow as tf 

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








