import matplotlib.pyplot as plt
import numpy as np



def plot2DData(data,target):
    plt.scatter(x = data[:,0],y = data[:,1],c = target,cmap=plt.cm.rainbow)
    plt.show()


def plot2DDataWithDecisionBoundary(data,target,model):
    x_min,x_max = np.min(data[:,0])-.5,np.max(data[:,0])+.5
    y_min,y_max = np.min(data[:,1])-.5,np.max(data[:,1])+.5
    X,Y = np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02)
    XX,YY = np.meshgrid(X,Y)
    Z = np.argmax(model.predict(np.c_[XX.ravel(),YY.ravel()]),axis=1).reshape(XX.shape)
    plt.contourf(XX,YY,Z,cmap=plt.cm.seismic)
    plt.scatter(x=data[:,0],y=data[:,1],c=target,cmap=plt.cm.seismic)
    plt.show()



def genSpiralData(points_per_class,num_classes):
    data   = np.ndarray((points_per_class*num_classes,2),np.float32)
    target = np.ndarray((points_per_class*num_classes,),np.uint8)
    r = np.linspace(0,1,points_per_class)
    radians_per_class = 2*np.pi/num_classes
    for i in range(num_classes):
        t = np.linspace(i*radians_per_class,(i+1.5)*radians_per_class,points_per_class)+0.1*np.random.randn(points_per_class)
        data[i*points_per_class:(i+1)*points_per_class] = np.c_[r*np.sin(t),r*np.cos(t)]
        target[i*points_per_class:(i+1)*points_per_class] = i

    return  data,target