# CPEG586Assignment5
CPEG586 Assignment 5

The CNN layers have been added to the model.  In order to create and run a model you would use a format such as:

    ``model = nn.Model(number_epochs=30,batch_size=100,lr=0.1 )``
    ``model.AddCNN(filters=6,kernel=5,ActivationType.RELU,prevInput=(28,28,1))``
    ``model.addAvgPool()``
    ``model.AddCNN(filters=12,kernel=5,ActivationType.RELU)``
    ``model.addAvgPool()``
    ``model.addFlatten()``
    ``model.AddDense(size=50,ActivationType.RELU)``
    ``model.AddDense(size=10,ActivationType.SOFTMAX)``
