from NeuralNet import Optimizers
from NeuralNet import Layers
from NeuralNet import Utilities
from NeuralNet import model
import numpy as np



if __name__=="__main__":
    batch_size        = 20
    num_epochs        = 2
    samples_per_class = 100
    num_classes       = 3
    hidden_units      = 100
    data,target       = Utilities.genSpiralData(samples_per_class,num_classes)
    model             = model.Model()
    model.add(Layers.Linear(2,hidden_units))
    model.add(Layers.ReLU())
    model.add(Layers.Linear(hidden_units,num_classes))
    optim   = Optimizers.SGD(model.parameters,lr=1.0,weight_decay=0.001,momentum=.9)
    loss_fn = Layers.SoftmaxWithLoss()
    model.fit(data,target,batch_size,num_epochs,optim,loss_fn)
    predicted_labels = np.argmax(model.predict(data),axis=1)
    accuracy         = np.sum(predicted_labels==target)/len(target)
    print("Model Accuracy = {}".format(accuracy))
    Utilities.plot2DDataWithDecisionBoundary(data,target,model)
   
