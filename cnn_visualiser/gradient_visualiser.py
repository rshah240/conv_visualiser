#Guided Backpropagation
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
import numpy as np

@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0.,tf.float32) * tf.cast(x>0.,tf.float32) * dy
    return tf.nn.relu(x),grad


class GradVisualiser:
    def __init__(self,model,layer_name,input_image = None):
        """Initialising Class Attributes
        :type input_image: numpy array
        """
        self.model = model
        self.input_image = input_image
        self.layer_name = layer_name

    def __str__(self):
        return "Model = {}, layer_name = {}".format(self.model,self.layer_name)

    def build_guided_model(self):
        """Building a model with the custom activation function"""
        gbModel = Model(self.model.input,self.model.output)
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        return gbModel

    def guided_gradients(self,deprocess = True):
        """Function for visualization of intermediate gradients"""
        self.gbModel = self.build_guided_model()
        if len(self.input_image.shape)!=4:
            self.input_image = np.expand_dims(self.input_image,axis=0) #adding a batch dimension
        self.input_image = tf.convert_to_tensor(self.input_image)
        self.input_image = tf.cast(self.input_image,tf.float32)
        inter_model = Model(self.gbModel.input,self.gbModel.get_layer(self.layer_name).output)
        with tf.GradientTape() as g:
            g.watch(self.input_image)
            output = inter_model(self.input_image)
        grads = g.gradient(output,self.input_image)[0]
        grads = grads.numpy()
        if deprocess:
            return GradVisualiser.deprocess_image(grads)
        else:
            return grads

    def generate_filter_pattern(self,filter_index,input_height,input_width,
                         custom_layer = None,steps = 40,input_channels = 3):
        """
        Function for pattern generation of intermediate feature maps of convolution layers using gradient ascent

        if custom layer is not passed function will automatically use layer_name attributes of this class
        return: activation map"""
        if custom_layer == None:
            layer_name = self.layer_name
        else:
            layer_name = custom_layer

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

    @staticmethod
    def deprocess_image(x):
        """utility function to convert a float array into a valid uint8 image
        #Arguments
            x: a numpy array representing the generated image
        #:returns
            A processed numpy array, which could be used in e.g. imshow
        """
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.25
        
        x += 0.5
        x = np.clip(x,0,1)
        
        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1,2,0))
        x = np.clip(x,0,255).astype('uint8')
        return x








    
        
                
        
                
                










