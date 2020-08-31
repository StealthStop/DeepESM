import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.stats import pearsonr, spearmanr

class Correlation:
    @staticmethod
    def distance_corr(var_1, var_2, normedweight, power=1):
        """
        https://github.com/gkasieczka/DisCo
        var_1: First variable to decorrelate (eg mass)
        var_2: Second variable to decorrelate (eg classifier output)
        normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
        power: Exponent used in calculating the distance correlation
        
        va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries
        
        Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
        """
        
        xx = tf.reshape(var_1, [-1, 1])
        xx = tf.tile(xx, [1, tf.size(var_1)])
        xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])
        
        yy = tf.transpose(xx)
        amat = tf.abs(xx-yy)
        
        xx = tf.reshape(var_2, [-1, 1])
        xx = tf.tile(xx, [1, tf.size(var_2)])
        xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
        
        yy = tf.transpose(xx)
        bmat = tf.abs(xx-yy)
        
        amatavg = tf.reduce_mean(amat*normedweight, axis=1)
        bmatavg = tf.reduce_mean(bmat*normedweight, axis=1)
        
        minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
        minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
        minuend_2 = tf.transpose(minuend_1)
        Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg*normedweight)
        
        minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
        minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
        minuend_2 = tf.transpose(minuend_1)
        Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg*normedweight)
        
        ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
        AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
        BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)
        
        if power==1:
            dCorr = tf.reduce_mean(ABavg*normedweight)/tf.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
        elif power==2:
            dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
        else:
            dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)))**power
        return dCorr
            
    def dist_corr(X, Y):
        """ 
        https://gist.github.com/satra/aa3d19a12b74e9ab7941
        Compute the distance correlation function
        
        >>> a = [1,2,3,4,5]
        >>> b = np.array([1,2,9,4,4])
        >>> distcorr(a, b)
        0.762676242417
        """
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        if np.prod(X.shape) == len(X):
            X = X[:, None]
        if np.prod(Y.shape) == len(Y):
            Y = Y[:, None]
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        n = X.shape[0]
        if Y.shape[0] != X.shape[0]:
            raise ValueError('Number of samples must match')
        a = squareform(pdist(X))
        b = squareform(pdist(Y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcor

    def pearson_corr(X, Y):
        cor, pvalue = pearsonr(X, Y)
        return cor

    def spearman_corr(X, Y):
        cor, pvalue = spearmanr(X, Y)
        return cor

