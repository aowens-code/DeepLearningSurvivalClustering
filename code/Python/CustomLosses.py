import tensorflow as tf

#Class Description: Custom keras losses 
class CustomLosses:

    groupAssignments = None
    groupCentroids = None
    
    #Description: Reconstruction loss
    #Parameters: y_true = true data, y_pred = predicted data
    #Returns: loss
    def reconstruction(y_true, y_pred):
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        loss = mse(y_true, y_pred)
        loss = tf.keras.backend.print_tensor(loss)
        return loss
    
    #Description: Cluster loss, designed for k = 2
    #Parameters: y_true = true data, bottleneck = bottleneck prediction
    #Returns: loss
    def clusterLossCloseFar(y_true, bottleneck):
        closestCentroidOne = tf.math.reduce_sum(tf.square(tf.subtract(bottleneck,CustomLosses.groupCentroids[0])), axis=1) * tf.dtypes.cast(CustomLosses.groupAssignments == 0, tf.float32)  
        closestCentroidTwo = tf.math.reduce_sum(tf.square(tf.subtract(bottleneck,CustomLosses.groupCentroids[1])), axis=1) *  tf.dtypes.cast(CustomLosses.groupAssignments == 1, tf.float32)
        furthestCentroidOne = tf.math.reduce_sum(tf.square(tf.subtract(bottleneck,CustomLosses.groupCentroids[0])), axis=1) * tf.dtypes.cast(CustomLosses.groupAssignments == 1, tf.float32)  
        furthestCentroidTwo = tf.math.reduce_sum(tf.square(tf.subtract(bottleneck,CustomLosses.groupCentroids[1])), axis=1) *  tf.dtypes.cast(CustomLosses.groupAssignments == 0, tf.float32)
        closeLoss =  (closestCentroidOne + closestCentroidTwo) 
        farLoss = (furthestCentroidOne + furthestCentroidTwo)
        loss = closeLoss - farLoss
        loss = tf.math.reduce_mean(loss)
        return loss   
                                 
    #Description: Cox partial likelihood loss function
    #Parameters: y_true = true data, bottleneck = network output
    #Returns: negativeLikelihood = the loss
    def survivalLoss(censor, bottleneck):
        hazardRatio = tf.keras.backend.exp(bottleneck)    
        logRisk = tf.keras.backend.log(tf.keras.backend.cumsum(hazardRatio))
        uncensoredLikelihood = bottleneck - logRisk
        censoredLikelihood = uncensoredLikelihood * censor
        negativeLikelihood = -tf.keras.backend.sum(censoredLikelihood)
        negativeLikelihood = tf.keras.backend.print_tensor(negativeLikelihood)
        return negativeLikelihood   

