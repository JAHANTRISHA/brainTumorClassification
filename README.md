
# Brain Tumor Multiclass Classification using VGG-19

This is a classification task using transfer learning. Here, a convolutional neural network is used to categorize medical images into four types.


##  Environment Setup And Tools
* Visual Studio Code(VS Code)
* Anaconda Navigator (To setup environments)

### Packages with version:
* python ------------------3.9.16
* keras ---------------------2.6.0
* keras-applications -------1.0.8
* keras-preprocessing------1.1.2
* tensorflow----------------2.6.0
* tensorflow-gpu-----------2.6.0
* Pillow --------------------10.2.0
* opencv --------------------4.5.1
* scikit-learn-----------------1.2.2
* scikit-image--------------0.18.1
* matplotlib-----------------3.7.1

## Dataset
The dataset has been taken from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). There are four types of brain tumor datasets: glioma, meningioma, notumor, and pituitary. After extracting the Kaggle dataset, there are two folders:
* Training
* Testing
  
In Training folder with subfolders, there are 5712 images.
In Testing folder with subfolders, there are 1311 images.

## Data Processing
All image sizes are not the same,  especially in notumor datasets.
* Resize all images to 512*512.
* Split the training dataset into two parts: 
     80% for training and 20% for validation.
## Data Visualization
* Pillow
  
![example](https://github.com/JAHANTRISHA/brainTumorClassification/assets/40772173/619341c5-7d0c-4f86-8343-7a07ae858af3)

* matplotlib
##  Feature Extraction and Train the model:
Transfer learning model VGG-19 is used to extract the feature and train the model. Global average pooling is utilized in the top layer to generate the feature map. Moreover, the model is trained using dropout and dense layers.Overfitting is avoided by using a regularizer. We adjusted the model using hyperparameters as well.
* optimizer: Adam
* activation function: relu
* last layer activation function: softmax 
* learning rate: .0001
* batch size: 64 
* loss fuction: categorical crossentropy
* regularizer L1: 0.01
## Fine-tuning the model
Fine-tuning improves in performance optimization for the model. So, fit the model with epoch 50.
## Evaluation
Here, Accuracy is 98.25%, AUC is 99.55%, Recall is 98.17% and Precision is 98.39%.
* Confusion Matrix
* Classification Report
## Prediction
To count the images that are correctly and incorrectly predicted.
