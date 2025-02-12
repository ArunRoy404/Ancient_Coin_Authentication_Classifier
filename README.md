# Ancient_Coin_Authentication_Classifier
In this model/Classifier, Convolutional Neural Network (CNN) model based on VGG16 was used to classify the real or fake Ancient Bengal Coins.  Convolutional Neural Network (CNN) model based on VGG16 to classify coins as real or fake




## Features
-Utilizes the VGG16 model for feature extraction.
-Custom layers added for binary classification (real vs. fake coins).
-Data preprocessing and augmentation for improved model performance.
-Evaluation metrics including accuracy, confusion matrix, and loss visualization.




## Technologies Used
-Python
-TensorFlow Keras
-OpenCV (for image preprocessing)
-NumPy, Pandas
-Matplotlib, Seaborn (for visualization)




## Dataset
The dataset consists of images of real and fake coins. Preprocessing includes resizing, normalization, and data augmentation to improve generalization.
Used dataset was collected form a research team. 




## Model Architecture
-Pre-trained VGG16 (with ImageNet weights)
-Fully connected layers for classification
-Softmax/Sigmoid activation for binary classification




## Training and Evaluation
-Data augmentation applied to enhance generalization.
-Model trained on Google Colab.
-Performance evaluated using accuracy, loss curves, and confusion matrix.




## Results
-Achieved more then 95% accuracy on the test dataset.
-Visualization of model performance using loss and accuracy curves.
-Confusion matrix analysis for detailed evaluation.
all can be found in the Jupyter Notebook File



## Future Improvements
-Expand dataset for better generalization.
-Implement real-time coin classification using a webcam.smartphone application.
-Improve model performance using other CNN architectures like ResNet or EfficientNet.
