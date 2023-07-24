import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import gc

class MeanShift:
    ''' Implementation of the mean shift alogrithm in a TF friendly
    manner. See https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.217.3313&rep=rep1&type=pdf

    This algorithm aims to determine the modes of a given n dimensional
    distribution. This method will use a Gaussian Kernel for calculating the 
    kernel density estimate. For large datasets, it is best to use a small 
    bandwidth parameter as it will reduce calculation time.

    The method for determining the modes (or centroids) is as follows:
    
    1. Calculate the mean shift vector m(x^t_i)for each point x^t_i in the set
    2. Update each point so that x^(t+1)_i = x^t_i + m(x^t_i)
    3. Repeat 1 and 2 until convergence of the kernel density estimate gradient
       to zero.

    The result is a gradient ascent procedure for each point in the set. This 
    is due to the fact that the mean shift vector points in the direction of
    the maximal increase in the kernel density estimate.

    All points that reach the same centroid in this procedure will be associated
    with one another in a cluster. The total number of clusters will be the 
    total number of centroids determined through the mean shift algorithm

    '''


    # Input data set should be a rank two tensor. If you have n samples of 
    # dimension d, the shape should be [n,d]
    def __init__(self, data):

        self.data = data
        self.X = data
        self.labels = tf.reduce_sum(tf.zeros_like(data), axis=1)
        self.Old_X = 2 * data
        self.bw = 0.1
        self.max_iter = 1000
        self.i = 0
        self.centroids = tf.zeros(shape=(0, 2), dtype=tf.float64)
        self.n_centroids = 0

    def fit(self):
        self.X, self.Old_X, self.max_iter, self.i = tf.while_loop(lambda X, Old_X, max_iter, i: self.converge_check(X, Old_X, max_iter, i), lambda X, Old_X, max_iter, i: self.mean_shift_step(X, Old_X, max_iter, i), [self.X, self.Old_X, self.max_iter, self.i])
        self.labels, self.centroids = self.prune_centroids(self.X, self.labels)

        return self.X, self.data, self.labels, self.centroids

    # Function to calculate the input norms for the kernel
    # Inputs:   X = data to be used for the calculation (shape = [n, d])
    # Outputs:  normX = distance of each point with respect to all others (shape = [n,n])
    def norm(self, X):

        diffX = (tf.expand_dims(X, 1) - tf.expand_dims(X,0)) / self.bw
        normX = tf.reduce_sum(tf.pow(diffX, 2), axis=2)

        return normX

    # Negative derivate of the Gaussian Kernel
    # Inputs:   X = data to be used for the calculation (shape = [n, d])
    # Outputs:  gX = Gaussian Kernel on all norms (shape = [n, n]) 
    def g(self, X):
        
        gX = tf.exp(-self.norm(X))

        return gX

    # Returns the mean shift vector for all points in the input data set
    # Inputs:   X = data to be used for the calculation (shape = [n, d])
    # Outputs:  msX = mean shift vectors for all n inputs (shape = [n, d]) 
    def mean_shift_vec(self, X):

        gX = self.g(X)

        num = tf.reduce_sum(tf.expand_dims(gX, 2) * X, axis=1)
        denom = tf.expand_dims(tf.reduce_sum(gX, axis=1), 1) 

        return (num / denom)

    # Checking to see if any of the points have not converged at a mode after
    # iteration step. Convergence requirement of max_diff < 1e-5
    def converge_check(self, new_data, data, max_iter, i):
        max_diff = tf.reduce_max(tf.reduce_sum(tf.sqrt(tf.pow(new_data - data, 2))))
        return tf.logical_and(tf.greater(max_diff,1e-3), tf.greater(max_iter, i))

    # While loop body which will be executed until convergence of the mean shifted inputs
    # Inputs:   X = data to be used for the calculation (shape = [n, d])
    # Outputs:  new_X = mean shifted values of X after one iteration (shape = [n, d])
    def mean_shift_step(self, X, Old_X, max_iter, i):

        self.Old_X = X
        self.X = self.mean_shift_vec(X)        

        self.i = self.i + 1
    
        gc.collect()

        return self.X, self.Old_X, self.max_iter, self.i

    # To be run after the while loop has converged to determine centroids.
    # Compares the final values of the mean shifted points to each other to
    # find all centroids
    # Inputs: X = mean shifted data after while loop (shape = [n, d])
    # Outputs: centroids = list of centroids (shape = [k] for k centroids)
    def prune_centroids(self, X, labels, i_lab=0, first=0):
        if i_lab == 5:
            return labels, self.centroids
        temp_centroid = tf.gather(X, first)
        self.centroids = tf.concat((tf.cast(self.centroids, dtype=tf.float32), tf.cast(tf.reshape(temp_centroid, shape=[1,2]),dtype=tf.float32)), axis=0)
        test_cent = tf.abs(X - temp_centroid)
        where = tf.where(tf.logical_and(tf.greater(tf.reduce_sum(test_cent, axis=1), 1e-3), tf.equal(labels, i_lab)))
        first = tf.cond(tf.equal(tf.size(where), tf.constant(0)), lambda: tf.reshape(first, shape=[1,]), lambda: tf.cast(tf.gather(where, 0),dtype=tf.int32))
        labels = tf.where(tf.logical_and(tf.greater(tf.reduce_sum(test_cent, axis=1), 1e-3), tf.equal(tf.cast(labels, dtype=tf.int32), tf.cast(i_lab, dtype=tf.int32))), i_lab + 1, tf.cast(labels, dtype=tf.int32))

        new_X = tf.reshape(tf.gather(X, tf.where(tf.greater(labels, i_lab))), [-1,2])
  
        gc.collect()

        return tf.cond(tf.equal(tf.size(new_X), tf.constant(0)), lambda: (labels, self.centroids), lambda: self.prune_centroids(X, labels, i_lab+1, first))

    def per_cluster(self):
        self.n_per_cluster = tf.zeros(shape=[0, 1])

        def count_occur(self, label):
            tot = tf.reduce_sum(tf.one_hot(self.labels, label))
            label = label + 1
            tf.concat((self.n_per_cluster, tot), axis=0)
            return label

        def check_done(self, label):
            return tf.less(label, tf.reduce_max(self.labels))

        label = 0
        tf.while_loop(lambda label: check_done(self, label), lambda label: count_occur(self, label), [label])
        
        print(self.n_per_cluster)
        return self.n_per_cluster 

    def loss(self):
        
        return 

def generate_data():
    
    cov = np.array([[0.1,0],[0,0.1]])
    data = np.random.multivariate_normal((0.25,0.25), 0.1*cov, 1000)
    data = np.concatenate((data, np.random.multivariate_normal((0.25,0.75), 0.01*cov, 500)))
    data = np.concatenate((data, np.random.multivariate_normal((0.75,0.25), 0.01*cov, 500)))
    data = np.concatenate((data, np.random.multivariate_normal((0.75, 0.75), 0.02*cov, 100)))

    data = np.array(data)

    np.random.shuffle(data)

    return tf.reshape(tf.constant(data),[-1, 2])

def plot(X, startX, labels):
    
    colors = ['green', 'orange', 'blue', 'pink', 'yellow']
    
    plt.scatter(startX[:, 0], startX[:, 1], c=labels, cmap=mpl.colors.ListedColormap(colors))
    plt.scatter(X[:, 0], X[:, 1], color='red')

    plt.show()

if __name__ == '__main__':
    
    data = generate_data()

    model = MeanShift(data)
    X, startX, labels, centroids = model.fit()
    new_lab = []
    for i in labels:
        new_lab.append(int(i))
 
    #model.per_cluster()

    #plot(centroids, data, new_lab)
