import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel

# You are allowed to import any submodules of sklearn e.g. metrics.pairwise to construct kernel Gram matrices
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_kernel, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

################################
# Non Editable Region Starting #
################################
def my_kernel(X1, Z1, X2, Z2):
    
    K_z = polynomial_kernel(
        Z1, Z2, degree = 2, coef0 = 20, gamma = 1.0,
    )

    K_x = X1 @ X2.T

    return K_x * K_z + 1.0 # Gram Matrice


################################
# Non Editable Region Starting #
################################

def factorW(w):
	# reshaping w
	w = w.reshape(33, 33)
    
	#extracting u_hat, v_hat via rank-1 SVD
	u, s, vt = np.linalg.svd(w, full_matrices=False) 
	uHat, vHat = np.sqrt(s[0]) * u[:, 0], np.sqrt(s[0]) * vt[0, :]
    
	return uHat, vHat

def utoAB(u):
    # alpha = A, beta = B
    A = u[ :32].copy()
    B = np.zeros_like(A)
    B[-1] = u[32]
    return A, B

def ABtoDelay(A, B):
    # alpha = A, beta = B
	A, B = np.asarray(A), np.asarray(B)

	# keeping all delays non negative
	k = np.maximum(0.0, np.maximum(-(A + B), -(A - B)))

	a, b, c, d = k + (A + B), k.copy(), k + (A - B), k.copy()

	return a, b, c, d


def my_decode(w):
	# alpha = A, beta = B
    
	uHat, vHat = factorW(w)

    # 1st PUF
	A1, B1 = utoAB(uHat)
	a, b, c, d = ABtoDelay(A1, B1)

    # 2nd PUF
	A2, B2 = utoAB(vHat)
	p, q, r, s = ABtoDelay(A2, B2)

    # Clamp to keep everything non-negative
	a = np.maximum(a, 0.0)
	b = np.maximum(b, 0.0)
	c = np.maximum(c, 0.0)
	d = np.maximum(d, 0.0)
	p = np.maximum(p, 0.0)
	q = np.maximum(q, 0.0)
	r = np.maximum(r, 0.0)
	s = np.maximum(s, 0.0)
      
	
	return a, b, c, d, p, q, r, s

