import tensorflow as tf
import matplotlib.pyplot as plt


class IntegratedGradients:
    '''Class having attributes and functions to calculate Integrated Gradients'''
    def __init__(self,model,input_image):
        #Initialising the class with input_image and model
        self.model = model
        self.input_image = tf.convert_to_tensor(input_image)
        #Initialising the baseline with the zero tensor of the same shape as of input image
        self.baseline = tf.zeros_like(self.input_image)

    def get_class_index(self):
        """Calculating class index of the image"""
        predictions = self.model(self.input_image)
        #Getting the most probable class
        class_idx = tf.argmax(predictions[0])
        return class_idx

    def interpolate_image(self,alphas):
        '''Calculating interpolation of images'''
        alphas_x = alphas[:,tf.newaxis,tf.newaxis,tf.newaxis]
        if (len(self.input_image.shape) != 4):
            #Adding the batch dimension
            self.input_image = tf.expand_dims(self.input_image,axis = 0)
            self.baseline = tf.expand_dims(self.baseline,axis = 0)
        delta = self.input_image - self.baseline
        images = self.baseline + alphas_x*delta #Braocasting to (m_steps + 1)
        return images

    def compute_gradients(self,images):
        '''Compute Gradients on the batch of images'''
        class_idx = self.get_class_index()
        with tf.GradientTape() as g:
            g.watch(images)
            logits = self.model(images)
            probs = tf.nn.softmax(logits,axis=-1)[:,class_idx]
        gradient = g.gradient(probs,images)
        return gradient

    def integral_approximation(self,grads):
        #reimann_trapezoidal
        grads = (grads[1:] + grads[:-1]) / tf.constant(2.0)
        grads = tf.reduce_mean(grads,axis = 0)
        return grads

    @tf.function
    def integrated_gradients(self,m_steps = 50,batch_size = 32):
        """Calculating attributions"""
        #1. Generate alphas
        alphas = tf.linspace(start=0.0,stop=1.0,num = m_steps+1)

        #Initializing TensorArray outside the loop to collect gradients
        gradient_batches = tf.TensorArray(tf.float32,size = m_steps+1)
        # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
        for alpha in tf.range(0,len(alphas),batch_size):
            from_ = alpha
            to = tf.minimum(from_ + batch_size,len(alphas))
            alpha_batch = alphas[from_:to]

            #2. Generated inerpolated inputs between baseline and input
            interpolated_images = self.interpolate_image(alpha_batch)

            #3. Calculate Gradients between model outputs and interpolated images
            gradient_batch = self.compute_gradients(interpolated_images)

            #4. Write batch indices and gradients to extend array
            gradient_batches = gradient_batches.scatter(tf.range(from_,to),gradient_batch)

        #Stack path gradients together row wise into single tensor
        total_gradients = gradient_batches.stack()

        # 4. Integral approximation through averaging gradients.
        avg_gradients = self.integral_approximation(total_gradients)

        # 5. Scale integrated gradients with respect to input.
        integrated_gradients = (self.input_image - self.baseline) * avg_gradients
        integrated_gradients = tf.reduce_sum(tf.math.abs(integrated_gradients),axis = -1)[0]

        return integrated_gradients

    def display_plot_img_attributions(self,cmap='viridis',m_steps = 50,batch_size = 32,overlay_alpha = 0.4):
        '''Function to plot Image Attributions'''
        attribution_mask = self.integrated_gradients(m_steps,batch_size)
        # Sum of the attributions across color channels for visualization.
        # The attribution mask shape is a grayscale image with height and width
        # equal to the original image.
        

        fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

        axs[0, 0].set_title('Baseline image')
        axs[0, 0].imshow(self.baseline[0])
        axs[0, 0].axis('off')

        axs[0, 1].set_title('Original image')
        axs[0, 1].imshow(self.input_image[0])
        axs[0, 1].axis('off')

        axs[1, 0].set_title('Attribution mask')
        axs[1, 0].imshow(attribution_mask, cmap=cmap)
        axs[1, 0].axis('off')

        axs[1, 1].set_title('Overlay')
        axs[1, 1].imshow(attribution_mask, cmap=cmap)
        axs[1, 1].imshow(self.input_image[0], alpha=overlay_alpha)
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.show(fig)
















    

        
