import numpy

class SOM():
    def __init__(self, x, y,delta0,t1,alpha0=0.1,t2=1000):        
        self.map = []
        self.n_neurons = x*y
        self.shape = [x,y]
        self.epoch = 0
        self.delta0 = delta0
        self.alpha0 = alpha0
        self.t1 = t1
        self.t2 = t2
        self.template = numpy.arange(x*y).reshape(self.n_neurons,1)
        #delta0=numpy.sqrt(x)
        #t1=1000/numpy.log(delta0)
        
    def modifyparameter(self,n):
        self.sigma = self.delta0 * numpy.exp(-1.0*n/self.t1)
        self.alpha = self.alpha0 * numpy.exp(-1.0*n/self.t2)
        
    def train(self, X, iterations, batch_size=1):
        if len(self.map) == 0:
            self.map = numpy.ones((self.n_neurons, len(X[0])))
        self.total = iterations    
        samples = [i for i in range(len(X))]
        numpy.random.shuffle(samples)
        for i in xrange(iterations):
            self.modifyparameter(i)
            idx = samples[i%len(samples)]
            self.iterate(X[idx])
            if(self.alpha < 0.01):
                print 'tatol iteration time %d' % i
                break
     
    def iterate(self, vector):  
        x, y = self.shape
        delta = self.map - vector
        # Euclidian distance of each neurons with the example
        dists = numpy.sum((delta)**2, axis=1).reshape(x,y)
        # Best maching unit
        idx = numpy.argmin(dists)
        #print "Epoch ", self.epoch, ": ", (idx/x, idx%y), "; Sigma: ", self.sigma, "; alpha: ", self.alpha
        # Linearly reducing the width of Gaussian Kernel
        dist_map = self.template.reshape(x,y)     
        # Distance of each neurons in the map from the best matching neuron
        dists = numpy.sqrt((dist_map/x - idx/x)**2 + (numpy.mod(dist_map,x) - idx%y)**2).reshape(self.n_neurons, 1)
        #dists = self.template - idx
        # Applying Gaussian smoothing to distances of neurons from best matching neuron
        h = numpy.exp(-(dists**2)/(2*(self.sigma**2)))      
        # Updating neurons in the map
        self.map -= self.alpha*h*delta
        # Decreasing alpha
        self.epoch = self.epoch + 1 


import  matplotlib
import matplotlib.pyplot as plt
import re
def load_data():
    datamat = []
    try:
        fr = open("data.txt")
    except Exception,e:
        print 'error'
    for line in fr.readlines():
        linearr = re.split(r'\s+',line.strip())
        length = len(linearr)
        tmp = []
        for i in range(length):
            tmp.extend([float(linearr[i])])
        datamat.append(tmp)
    return datamat
def DrowFigure(cl):
    datamat = load_data()
    length = len(datamat)
    # Plotting hidden units
    W = cl.map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plotmat = numpy.mat(W)
    plt.scatter(plotmat[:,0],plotmat[:,1])
    plt.show()
    
def demo():
    # Get data
    X = load_data()
    
    cl = SOM(5,5,2.23,450,0.1,1000)#set parameters here(including parameters for learning rate and gussian distance function
    cl.train(X, 5000)  
    DrowFigure(cl)
if __name__ == '__main__':
    demo()
