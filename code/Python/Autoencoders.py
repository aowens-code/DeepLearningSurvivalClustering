import tensorflow as tf

#Class Description: Create autoencoder models 
class Autoencoders:
        
    #Description: Creates autoencoder with single output 
    #Parameters: originalDim = original dimension of data, bottleneckDim = size of bottleneck dimension, activation = activation function, hiddenLayerDim = dimension of hidden layers
    #Returns: model = autoencoder model
    def getAutoencoder(originalDim, bottleneckDim, activation, hiddenLayerDim):
        input_layer = tf.keras.Input(shape=(originalDim,), name="input")
        hiddenLayerOne = tf.keras.layers.Dense(hiddenLayerDim, activation=activation, name="hidden_one",kernel_regularizer=tf.keras.regularizers.l1(0.001))(input_layer)    
        encoded = tf.keras.layers.Dense(bottleneckDim, activation=activation, name='bottleneck',kernel_regularizer=tf.keras.regularizers.l1(0.001))(hiddenLayerOne)
        hiddenLayerTwo =  tf.keras.layers.Dense(hiddenLayerDim, activation=activation, name="hidden_two",kernel_regularizer=tf.keras.regularizers.l1(0.001))(encoded)
        decoded = tf.keras.layers.Dense(originalDim, activation=activation, name='reconstructed', kernel_regularizer=tf.keras.regularizers.l1(0.001))(hiddenLayerTwo)        
        model = tf.keras.Model(inputs=input_layer, outputs=decoded)
        model.summary()
        return model
    
    #Description: Creates autoencoder with decoded and bottleneck outputs
    #Parameters: originalDim = original dimension of data, bottleneckDim = size of bottleneck dimension, activation = activation function, hiddenLayerDim = dimension of hidden layers
    #Returns: model = autoencoder model  
    def getAutoencoderBottleneck(originalDim, bottleneckDim, activation, hiddenLayerDim):
        input_layer = tf.keras.Input(shape=(originalDim,), name="input")
        hiddenLayerOne = tf.keras.layers.Dense(hiddenLayerDim, activation=activation, name="hidden_one",kernel_regularizer=tf.keras.regularizers.l1(0.001))(input_layer)    
        encoded = tf.keras.layers.Dense(bottleneckDim, activation=activation, name='bottleneck',kernel_regularizer=tf.keras.regularizers.l1(0.001))(hiddenLayerOne)
        hiddenLayerTwo =  tf.keras.layers.Dense(hiddenLayerDim, activation=activation, name="hidden_two",kernel_regularizer=tf.keras.regularizers.l1(0.001))(encoded)
        decoded = tf.keras.layers.Dense(originalDim, activation=activation, name='reconstructed',kernel_regularizer=tf.keras.regularizers.l1(0.001))(hiddenLayerTwo)        
        model = tf.keras.Model(inputs=input_layer, outputs=[decoded, encoded])
        model.summary()
        return model
    
    #Description: Creates autoencoder with decoded, bottleneck, and survival branch outputs
    #Parameters: originalDim = original dimension of data, bottleneckDim = size of bottleneck dimension, activation = activation function, hiddenLayerDim = dimension of hidden layers
    #Returns: model = autoencoder model
    def getAutoencoderBottleneckSurvival(originalDim, bottleneckDim, activation, hiddenLayerDim):
        input_layer = tf.keras.Input(shape=(originalDim,), name="input")
        hiddenLayerOne = tf.keras.layers.Dense(hiddenLayerDim, activation=activation, name="hidden_one", kernel_regularizer=tf.keras.regularizers.l1(0.001))(input_layer)    
        encoded = tf.keras.layers.Dense(bottleneckDim, activation=activation, name='bottleneck', kernel_regularizer=tf.keras.regularizers.l1(0.001))(hiddenLayerOne)
        survivalBranch = tf.keras.layers.Dense(1, activation='linear', name='risk', kernel_regularizer=tf.keras.regularizers.l1(0.001))(encoded)
        hiddenLayerTwo =  tf.keras.layers.Dense(hiddenLayerDim, activation=activation, name="hidden_two", kernel_regularizer=tf.keras.regularizers.l1(0.001))(encoded)
        decoded = tf.keras.layers.Dense(originalDim, activation=activation, name='reconstructed', kernel_regularizer=tf.keras.regularizers.l1(0.001))(hiddenLayerTwo)        
        model = tf.keras.Model(inputs=input_layer, outputs=[decoded, encoded, survivalBranch])
        model.summary()
        return model
    
    #Description: Creates autoencoder with decoded and survival branch outputs
    #Parameters: originalDim = original dimension of data, bottleneckDim = size of bottleneck dimension, activation = activation function, hiddenLayerDim = dimension of hidden layers
    #Returns: model = autoencoder model
    def getAutoencoderSurvival(originalDim, bottleneckDim, activation, hiddenLayerDim):
        input_layer = tf.keras.Input(shape=(originalDim,), name="input")
        hiddenLayerOne = tf.keras.layers.Dense(hiddenLayerDim, activation=activation, name="hidden_one", kernel_regularizer=tf.keras.regularizers.l1(0.001))(input_layer)    
        encoded = tf.keras.layers.Dense(bottleneckDim, activation=activation, name='bottleneck', kernel_regularizer=tf.keras.regularizers.l1(0.001))(hiddenLayerOne)
        survivalBranch = tf.keras.layers.Dense(1, activation='linear', name='risk',kernel_regularizer=tf.keras.regularizers.l1(0.001))(encoded)
        hiddenLayerTwo =  tf.keras.layers.Dense(hiddenLayerDim, activation=activation, name="hidden_two")(encoded)
        decoded = tf.keras.layers.Dense(originalDim, activation=activation, name='reconstructed',kernel_regularizer=tf.keras.regularizers.l1(0.001))(hiddenLayerTwo)        
        model = tf.keras.Model(inputs=input_layer, outputs=[decoded, survivalBranch])
        model.summary()
        return model
