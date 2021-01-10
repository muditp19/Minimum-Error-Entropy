# Minimum-Error-Entropy
Time delay neural nets implemented with MEE loss and compared to MSE loss
Analysis of MEE versus MSE


1.	MEE criterion converges faster than MSE with a difference of 1/5th the iterations used in case of MEE loss as gradient of entropy is greater than the mean squared error.

2.	By looking at the prediction plots of MSE and MEE, it is clear that MEE is ale to extract more information from the signal and reduced the error when trained with noise.

3.	MEE outperforms MSE in noise rejection consistently for all sizes of data sets. I did experiments with less number of samples and it outperforms 

4.	MEE is data efficient compared to MSE as it obtains same level of accuracy with less samples.

5.	MEE is more robust to noise in the desired signal in a finite-sample case 



Model used for TDNN was single layer Conv1D. Below show the model summary with kernel size of 5 –


Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_23 (Conv1D)           (None, 6, 32)             192       
_________________________________________________________________
dropout_23 (Dropout)         (None, 6, 32)             0         
_________________________________________________________________
max_pooling1d_23 (MaxPooling (None, 6, 32)             0         
_________________________________________________________________
flatten_23 (Flatten)         (None, 192)               0         
_________________________________________________________________
dense_23 (Dense)             (None, 1)                 193       
=================================================================
Total params: 385
Trainable params: 385
Non-trainable params: 0



