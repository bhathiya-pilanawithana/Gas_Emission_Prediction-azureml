# Gas_Emission_Prediction-azureml
Prediction of turbine CO emission with sensor data

In this we train a simple neural network with one hidden layer together with hyper parameter optimization for scaling method used, number of hidden layer neurons, hidden layer activation function, learning rate, number of epochs and the momentum to predict the CO gas emission from turbine sensor data. The original work was carried out in https://journals.tubitak.gov.tr/cgi/viewcontent.cgi?article=1505&context=elektrik with DOI: 10.3906/elk-1807-87.

We make use of Azure Machine Learning (AzureML) cloud platform for training, testing and deployment of the model. We were able to achive a slightly higher validation and testing performance in terms of coeeffcient of determination, than what was mentioned in the original paper.

### Prerequisites:
1. An Azure account with an active subscription together with Owner or Contributer role.
2. An AzureML Workspace with all required dependancies.
3. Azure Machine Learning Python SDK v2 in Notebook execution environment.
