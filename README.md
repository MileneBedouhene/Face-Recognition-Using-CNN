# Face Recognition using CNN

This repository contains a deep learning model for face recognition using a Convolutional Neural Network (CNN). The model achieves **95% accuracy** in identifying faces.

## Dataset
The dataset used for training includes labeled images of different individuals, processed for face detection and recognition. Preprocessing steps include image resizing, normalization, and data augmentation.

## Model and Approach
The model is built using the following steps:
1. **Data Preprocessing**: Image normalization, augmentation, and resizing for consistent input dimensions.
2. **CNN Architecture**:
   - Multiple convolutional layers with ReLU activation.
   - Max pooling for dimensionality reduction.
   - Fully connected layers for classification.
   - Softmax activation for multi-class classification.
3. **Training the Model**:
   - Optimized with Adam optimizer.
   - Categorical cross-entropy as the loss function.
   - Early stopping and dropout layers to prevent overfitting.
4. **Evaluation**: The model achieves an accuracy of **95%**, evaluated using precision, recall, and confusion matrix.

## Results
- Achieved **95% accuracy** in face recognition tasks.
- Effective feature extraction using deep CNN layers.
- Robust performance in different lighting conditions and facial angles.

## Future Improvements
- Experimenting with deeper CNN architectures like ResNet.
- Integrating real-time face recognition using OpenCV.
- Deploying the model as a web application.


## Acknowledgments
- TensorFlow and Keras for deep learning frameworks.
- OpenCV for image processing utilities.


