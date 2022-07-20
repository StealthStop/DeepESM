import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata

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

    def pearson_corr_tf(v, u):

        mv, sv = tf.nn.moments(v, axes=[0])
        mu, su = tf.nn.moments(u, axes=[0])

        ev = v - mv
        eu = u - mu

        j = ev*eu
        mj = tf.reduce_mean(j)

        num = mj
        den = sv*su

        return tf.abs(num)

    def pearson_corr(X, Y):
        cor, pvalue = pearsonr(X, Y)
        return cor

    def spearman_corr(X, Y):
        cor, pvalue = spearmanr(X, Y)
        return cor
    
    @staticmethod
    def rdc(var_1, var_2, f=tf.sin, k=20, s=1/6., n=1):

        # Some functions to handle all of the conditions within the while loop for
        # ensuring real eigvals in tensor friendly syntax

        # What to do when eigs aren't real or not between zero and one
        def nonreal(ub, lb, k):
            ub -= 1 
            k = (ub + lb) // 2 
            return ub, lb, k
       
        # What to do when eigs are real and between zero and one
        def real(ub, lb, k):
            '''
            Should behave the same as:
            if lb == ub: break
            bound_manip(ub, lb, k)
            '''
            return tf.cond(tf.equal(lb, ub), lambda: (ub, lb, k), lambda: bound_manip(ub, lb, k))

        # Manipulate lower bound when not the same as upper bound
        def bound_manip(ub, lb, k):
            '''
            # Logic should be the same as the code below
            lb = k
            if ub == lb + 1:
                k = ub
            else:
                k = (ub + lb) // 2
            return ub, lb, k 
            '''
            lb = k
            k = tf.cond(tf.equal(ub, lb+1), lambda: ub, lambda: (ub + lb) // 2)
            return ub, lb, k

        # Case for determining if we have reached a value of k where the eigenvalues are in [0, 1] and real
        def while_case(C, ub, lb, k, k0, eigs, case):
            return tf.reduce_all([tf.not_equal(ub, lb), tf.logical_not(case)])       

        # While loop to determine real eigenvalues... but make it tensor friendly :) 
        def while_body(C, ub, lb, k, k0, eigs, case): 
            # Compute canonical correlations
            Cxx = C[:k, :k]
            Cyy = C[k0:k0+k, k0:k0+k]
            Cxy = C[:k, k0:k0+k]
            Cyx = C[k0:k0+k, :k]

            eigs = tf.linalg.eigvals(tf.matmul(tf.matmul(tf.linalg.pinv(Cxx), tf.transpose(Cxy)), tf.transpose(tf.matmul(tf.linalg.pinv(Cyy), tf.transpose(Cyx)))))
   
            mag = tf.abs(eigs)

            # Case to determine if all eigenvalues are real and within [0,1]
            case = tf.reduce_all(tf.stack([tf.reduce_all(tf.equal(0., tf.math.imag(eigs))), tf.reduce_all(tf.less_equal(0., tf.reduce_min(mag))), tf.reduce_all(tf.greater_equal(1., tf.reduce_max(mag)))]))

            ub, lb, k = tf.cond(case, lambda: real(ub, lb, k), lambda: nonreal(ub, lb, k))
            return[C, ub, lb, k, k0, eigs, case]

            '''
            # Original logic for computing proper eigenvalues by manipulating k
            if not case: 
                ub -= 1 
                k = (ub + lb) // 2 
                continue
            # Binary search if k is too large
            if lb == ub: break
            lb = k
            if ub == lb + 1:
                k = ub
            else:
                k = (ub + lb) // 2
            '''

        # Ensure correct shape of input vectors
        var_1_temp = tf.reshape(var_1, [-1, 1])
        var_2_temp = tf.reshape(var_2, [-1, 1])
       
        #Copula Transformation
        # NO ITERATING OVER TENSORS
        #x = tf.divide(tf.constant([rankdata(xc, method='ordinal') for xc in tf.transpose(var_1)]), var_1.shape[0])
        #y = tf.divide(tf.constant([rankdata(yc, method='ordinal') for yc in tf.transpose(var_2)]), var_2.shape[0])

        x = 1+tf.argsort(tf.argsort(tf.transpose(var_1))) / tf.shape(var_1)[0]
        y = 1+tf.argsort(tf.argsort(tf.transpose(var_2))) / tf.shape(var_2)[0]

        # Adding in column of ones to each tensor to make the random linear projection a dot product
        x = tf.reshape(x, (-1, 1))
        y = tf.reshape(y, (-1, 1))
        ones = tf.ones_like(x)

        X = tf.concat([x, ones], 1)
        Y = tf.concat([y, ones], 1)

        # Random linear projection
        normX = tf.random.normal([X.shape[1], k], 0, s)
        normY = tf.random.normal([Y.shape[1], k], 0, s)

        X = tf.matmul(tf.cast(X, tf.float32), normX)
        Y = tf.matmul(tf.cast(Y, tf.float32), normY)
        
        # Apply nonlinear function to random projection
        fX = tf.map_fn(fn=f, elems=X)
        fY = tf.map_fn(fn=f, elems=Y)

        # Compute covariance matrix and eigen values
        mean_x = tf.reduce_mean(fX, axis=0)
        mean_y = tf.reduce_mean(fY, axis=0)

        Cxx = tf.cast(1 / (tf.shape(fX)[0]), tf.float32) * tf.matmul(tf.transpose(fX - mean_x), fX - mean_x)
        Cyx = tf.cast(1 / (tf.shape(fX)[0]), tf.float32) * tf.matmul(tf.transpose(fY - mean_y), fX - mean_x)
        Cxy = tf.cast(1 / (tf.shape(fX)[0]), tf.float32) * tf.matmul(tf.transpose(fX - mean_x), fY - mean_y)
        Cyy = tf.cast(1 / (tf.shape(fX)[0]), tf.float32) * tf.matmul(tf.transpose(fY - mean_y), fY - mean_y)
       
        Cx = tf.concat((Cxx, Cyx), axis=0)
        Cy = tf.concat((Cxy, Cyy), axis=0)
        C = tf.concat((Cx, Cy), axis=1)

        k = tf.constant(k)
        k0 = k
        lb = tf.constant(1)
        ub = k
        eigs = tf.reshape(tf.convert_to_tensor((), dtype=tf.complex64), shape=(-1,))
        case = tf.constant(False)
    
        C, ub, lb, k, k0, eigs, case = tf.while_loop(while_case, while_body, [C, ub, lb, k, k0, eigs, case], [C.get_shape(), ub.get_shape(), lb.get_shape(), k.get_shape(), k0.get_shape(), tf.TensorShape([None,]), case.get_shape()])

        '''
        # Original while loop logic (not tensor friendly)
        while True:

            # Compute canonical correlations
            Cxx = C[:k, :k]
            Cyy = C[k0:k0+k, k0:k0+k]
            Cxy = C[:k, k0:k0+k]
            Cyx = C[k0:k0+k, :k]

            eigs = tf.linalg.eigvals(tf.matmul(tf.matmul(tf.linalg.pinv(Cxx), tf.transpose(Cxy)),
                                            tf.transpose(tf.matmul(tf.linalg.pinv(Cyy), tf.transpose(Cyx)))))
   
            mag = tf.abs(eigs)

            case = tf.reduce_all(tf.stack([tf.reduce_all(tf.equal(0., tf.math.imag(eigs))), tf.reduce_all(tf.less_equal(0., tf.reduce_min(mag))), tf.reduce_all(tf.greater_equal(1., tf.reduce_max(mag)))]))

            if case is None: continue

            # Binary search if k is too large
            ub, k = tf.cond(case, true_fn=lambda: (ub, k), false_fn=lambda: bounds(ub,lb,k) )
            if lb == ub: break
            lb = k
            if ub == lb + 1:
                k = ub
            else:
                k = (ub + lb) // 2
        '''

        return tf.sqrt(tf.reduce_max(tf.math.real(eigs)))

        C = np.cov(np.hstack([fX, fY]).T)

        # Due to numerical issues, if k is too large,
        # then rank(fX) < k or rank(fY) < k, so we need
        # to find the largest k such that the eigenvalues
        # (canonical correlations) are real-valued
        k = 20

        k0 = k
        lb = 1
        ub = k
        while True:

            # Compute canonical correlations
            Cxx = C[:k, :k]
            Cyy = C[k0:k0+k, k0:k0+k]
            Cxy = C[:k, k0:k0+k]
            Cyx = C[k0:k0+k, :k]

            eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                            np.dot(np.linalg.pinv(Cyy), Cyx)))

            # Binary search if k is too large
            if not (np.all(np.isreal(eigs)) and
                    0 <= np.min(eigs) and
                    np.max(eigs) <= 1):
                ub -= 1
                k = (ub + lb) // 2
                continue
            if lb == ub: break
            lb = k
            if ub == lb + 1:
                k = ub
            else:
                k = (ub + lb) // 2

        print(eigs)
        print(k, ub, lb)
        print(np.sqrt(np.max(eigs)))

#c = Correlation()

#a = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float64)
#b = tf.constant([4.0, 3.0, 2.0, 1.0], dtype=tf.float64)

#print(c.rdc(a, b))
