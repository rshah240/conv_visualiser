from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

class IntermediateActivations:
    def __init__(self,model,layer_names):
        """Initializing intermediate models"""
        layer_outputs = [layer.output for layer in model.layers if layer.name in layer_names]
        self.activation_model = Model([model.input],[layer_outputs])
        self.activation_model.compile(optimizer="sgd",loss="categorical_crossentropy")

    def __str__(self):
        return "Model {}".format(self.activation_model)

    def display_single_channel(self,input_image,activation_no = 0,filter_index = 0,cmap='viridis'):
        """Function for plotting single channel activations
        return: Plot of the image"""
        activations = self.activation_model.predict(input_image)[0]
        activations  = np.array(activations)[activation_no]
        plt.matshow(activations[0,:,:,filter_index],cmap=cmap)

    def display_grid(self,input_image,images_per_row = 16,cmap = 'viridis'):
        layer_names = []
        for layer in self.activation_model.layers:
            layer_names.append(layer)

        if len(input_image.shape) != 4:
            input_image = np.expand_dims(axis = 0)

        activations = self.activation_model.predict(input_image)[0]
        activations = np.array(activations)
        for layer_name,layer_activation in zip(layer_names,activations):
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]

            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols,images_per_row * size))
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0,:,:,col*images_per_row + row]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image,0,255).astype('uint8')
                    display_grid[col * size: (col +1 )*size, row*size:(row + 1)*size] = channel_image
                
            scale = 1./size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid,aspect='auto',cmap=cmap)
            
                    
                
                    









