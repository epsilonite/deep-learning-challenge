# Neural Network Model
## Overview
The nonprofit foundation Alphabet Soup wants a tool that can help them select applicants for funding with the best chance of success in their ventures. They provide us with their dataset to which we apply our knowledge of machine learning and neural networks to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, we have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
* EIN and NAME — Identification columns
* APPLICATION_TYPE — Alphabet Soup application type
* AFFILIATION — Affiliated sector of industry
* CLASSIFICATION — Government organization classification
* USE_CASE — Use case for funding
* ORGANIZATION — Organization type
* STATUS — Active status
* INCOME_AMT — Income classification
* SPECIAL_CONSIDERATIONS — Special considerations for application
* ASK_AMT — Funding amount requested
* IS_SUCCESSFUL — Was the money used effectively
## Data Preprocessing
The dataset was manipulated to identify the target, gather the features and remove irreleant data.
* Target [y]: IS_SUCCESSFUL
* Features [X]: APPLICATION_TYPE, AFFLIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT
* Removed: EIN, NAME (Identification columns are not included in Features)

The number of unique values for each feature was inspected and features with categorical data that exceeded ten values (APPLICATION_TYPE and CLASSIFICATION) were reduced. The built-in pandas function `get_dummies()` was used to convert categorical features into numeric data. The resulting dataset was then split using sklearn `train_test_split()` with random_state=888, then noramlized using sklearn `StandardScaler()`.
## Compiling, Training, and Evaluating the Model
### Initial Model
Initial Model Parameters:
* Input Dimension: 43
* Hidden Layers: [3] 43, 29, 19
* Activation Function: sigmoid
* Output Layer: 1 (sigmoid)
* Epochs: 29
* Accuracy Score: 73.34%

We chose three hidden layers and set the number of neurons in the first layer to number of features, then took two-thirds the number to the closest odd number to derive the following two layers, resulting in 29 neurons for the second layer and 19 for the third layer. We set the activation function to sigmoid, thinking that it may perform better considering our features have been scaled to be between 1, -1. We originally set the epochs to a hundred, but after looking at the training, decided to reduce it by about two-thirds, and decided to set the value to the same number as the number of neurons in our second layer: 29.

We used the article *[How do determine the number of layers and neurons in the hidden layer? by Sandhya Krishnan](https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3)* as a guideline in informing our parameters for our initial model, specifically the following snippets:
> ### Number of Neurons In Input and Output Layers
> The number of neurons in the input layer is equal to the number of features in the data and in very rare cases, there will be one input layer for bias. Whereas the number of neurons in the output depends on whether is the model is used as a regressor or classifier. If the model is a regressor then the output layer will have only a single neuron but in case if the model is a classifier it will have a single neuron or multiple neurons depending on the class label of the model.

> ### Table 5.1: Determining the Number of Hidden Layers
> | Number of Hidden Layers | Result |
> | ----------------------- | ------ |
> | none | Only capable of representing linear separable functions or decisions |
> | 1 | Can approximate any function that contains a continuous mapping from one finite space to another |
> | 2 | Can represent an arbitrary decision boundary to arbitrary accuracy with rational activation functions and can approximate any smooth mapping to any accuracy |
>
> credit: *[An Introduction to Neural Networks for Java, Second Edition by Jeff Heaton](https://web.archive.org/web/20140721050413/http://www.heatonresearch.com/node/707)*


> ### Some rule-of-thumb methods for determining number of neurons to use in the hidden layers:
> * The number of hidden neurons should be between the size of the input layer and the size of the output layer.
> * The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
> * The number of hidden neurons should be less than twice the size of the input layer.

### Keras-Tuner Optimization
The accuracy of the initial model was not able to reach 75%, so we decided to import the Keras-Tuner library to see if we could quickly iterate through different models to get better results. We were not able to get better results in our initial optimization, but after we made some tweaks, we were able to get some models that performed marginally better than the initial model.

Initial Keras-Tuner Parameters:
* Input Dimension: 43
* Hidden Layers: [<=5]
* First Layer: 43-83 neurons
* Subsequent Layers: 1-87 neurons
* Activation Function: sigmoid | relu | tanh
* Output Layer: 1 (sigmoid)
* Max Epochs: 20
* Accuracy: < Initial Model

Refined Keras-Tuners Parameters:
* Input Dimension: 43
* Hidden Layers: [<=3]
* First Layer: 43-87 neurons
* Subsequent Layers: 1-87 neurons
* Activation Function: sigmoid
* Output Layer: 1 (sigmoid)
* Accuracy: >= Initial Model

Refined Keras-Tuner Best Model 1:
* Hidden Layers: [3] 57, 21, 1
* Epochs/Max Epochs: 20 / 20
* Accuracy Score: 73.34%

Refined Keras-Tuner Best Model 2:
* Hidden Layers: [3] 67, 7, 57
* Epochs/Max Epochs: 2 / 10
* Accuracy Score: 73.35%

Refined Keras-Tuner Best Model 3:
* Hidden Layers: [3] 73, 27, 43
* Epochs/Max Epochs: 50 / 50
* Accuracy Score: 73.39%

### Manual Optimization
The models derived from the Keras-Tunner didn't show enough improvement, but all the models that did had higher initial neurons, so we decided to manually increment the first number, first we tried the initial neuron counts for the Keras-Tuner Models, of which the best was a strting layer of 57 neurons. We then tried the inverse two-thirds of 43, and got 65, which resulted in the highest accuracy of 73.44%.

Keras-Tuner Based Model Parameters:
* Input Dimension: 43
* Hidden Layers: [3] 57, 29, 19
* Activation Function: sigmoid
* Output Layer: 1 (sigmoid)
* Epochs: 50
* Accuracy Score: 73.29%

Inverse Two-Thirds Model Parameters:
* Input Dimension: 43
* Hidden Layers: [3] 65, 29, 19
* Activation Function: sigmoid
* Output Layer: 1 (sigmoid)
* Epochs: 7
* Accuracy Score: 73.44%

## Summary
The best model produced in the training process ended up being a model loosely based on the two-thirds approach. It is most likely coincidental that this approach yielded the best results, however, it is probably an effective starting strategy. The Keras-Tuner enabled us to iterate through multiple models. In our use case, it yielded models that performed similarly to our initial model, but was informative in identifying which activation function had a tendency to yield the best results, which was then applied to produce better models. In general, our models were prone to overfitting, so the next iteration of optimization would be to explore and incorporate strategies to prevent overfitting to see if that would lead to better models.
* **Preprocessing:** Any reduction of data (further binning) seems to reduce the model performance, further preprocessing resulting in data enrichment or augmentation may yield better results
* **Activation Function:** The sigmoid function seems to best fit our dataset according to the feedback from the Keras-Tuner, further exploration can be done on combinations of activation functions
* **Layers:** We explored between 2-5 dense layers, and we were able to produce the best models with 3 layers; exploring the implementation of strategies to prevent overfitting may yield better results, by adding some combination of weight regularization and dropout layers
* **Neurons:** Having an initial number of neurons greater than the input number resulted in the best results for dense layers, the values of the best models being between the input number and double the input number, however the numbers should be adjusted to accomodate overfitting prevention strategies
* **Epochs:** We started with 100 epochs, but this seemed to be overfitting the data, so we were able to get better performing models with 7, 20 and 50 epochs; these numbers could possible be further adjusted by smaller increments to improve accuracy, but it seems more likely that we can benefit from implementing strategies to prevent overfitting instead of reducing epochs.
* **Train/Test Split:** We used the default train/test split of 75/25 with random seed 888, further exploration can be done to see if there is a better split for our dataset between 60/40 to 90/10 by increments of 5.