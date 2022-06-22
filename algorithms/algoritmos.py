import numpy as np
import __future__ as division
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import copy
from collections import Sequence
from itertools import chain, count
from scipy.linalg import block_diag
from typing import Any


# -*- Some useful preliminary functions -*-
def trinv(matrix):
    tri = np.trace(la.inv(matrix))
    return tri

def permutation_matrix(n):
    rr = range(n)
    np.random.shuffle(rr)
    P = np.take(np.eye(n), rr, axis=0)
    return  P

def select_mat(matrix, index_row, index_column):
    # sort the row/cols lists
    index_row.sort()
    index_column.sort()
    index_row = list(set(index_row))
    index_column = list(set(index_column))
    S = np.transpose(np.transpose(matrix[index_row])[index_column])
    return S

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def ismember(a, B):
    response = False
    index = 0
    while response == False and index < len(B):
        b = B[index]
        if b == a:
            return True
        else:
            index = index+1

    return response

def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return
    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset
        yield [ [ first ] ] + smaller

def select_from_list(list, indices):
    list_select = [list[i] for i in indices]
    return list_select


# Finding a pendent-pair of a supermodular system(V,f):
# For a given sets W and Q, find the most far element
# in Q to W
# W and Q should be list of lists to account for fused elements

def Find_Far_Element(SS, F, WW, QQ):

    # Find the most far element to WW in QQ

    u = QQ[0]  # a list not an index
    W_cp = list(copy.copy(WW))
    W_cp.append(u)
    print(W_cp)
    A = F(SS, SS)
    print(A,"A")

    
    elt_far = u
    Q_ = copy.copy(QQ)
    Q_.remove(u)
    for elt in Q_:
        W_cp = copy.copy(WW)
        W_cp.append(elt)
        dist_elt = F(SS, W_cp) - F(SS, elt)
        if dist_elt > dist_max:
            dist_max = dist_elt
            elt_far = elt

    return elt_far

# ----- Finding a pendent pair is a fundamental step in Queyranne's algorithm ----- #
def PENDENT_PAIR(SS, VV, F):

    # V is the set of all points including fused pairs
    # The size of V goes from n to 2
    # Start with a random element in V

    V_ = list(copy.copy(VV))
    print(VV,"a------------------",V_)
    rnd_pattern = np.random.permutation(len(V_))
    x = V_[rnd_pattern[0]]
    #x = V_[0]
    if type(x) == list:
        W = x
    else:
        W = [x]

    Q = list(copy.copy(V_))
    print(Q)
    Q.remove(x)
    V_.remove(x)
    for i in range(len(V_)):
        elt_far = Find_Far_Element(SS, F, W, Q)
        W.append(elt_far)
        Q.remove(elt_far)

    return W[-2], W[-1]

def tr_inv(SS, set):
    """
    :rtype: float
    """
    if type(set) == int:
        LIST = [set]
    else:
        LIST = []
        for i in range(len(set)):
            if type(set[i]) == list:
                LIST.extend(set[i])
            else:
                LIST.append(set[i])

    return trinv(select_mat(SS, LIST, LIST))

def log_det(SS, set):
    """
    :rtype: float
    """
    if type(set) == int:
        LIST = [set]
    else:
        LIST = []
        for i in range(len(set)):
            if type(set[i]) == list:
                LIST.extend(set[i])
            else:
                LIST.append(set[i])

    return -np.log(la.det(select_mat(SS, LIST, LIST)))

def fuse(A, B):
    if type(A) == int and type(B) == int:
        f = [A, B]
    elif type(A) == int and type(B) == list:
        f = [A] + B
    elif type(A) == list and type(B) == int:
        f = A + [B]
    elif type(A) == list and type(B) == list:
        f = A + B

    return f

#-*- Full implementation of Queyranne's algorithm -*-

def QUEYRANNE(SS, F):
   # # type: (matrix, function) -> (list, float, list)

    dim, _ = SS.shape
    V = range(dim)  # is the space of points which is updated at each step we find a pendent pair
    C = []  # set of candidates updated at each step
    while len(V) >= 3:
        W = copy.deepcopy(V)
        a, b = PENDENT_PAIR(SS, W, F)  # find a pendent pair in (V,F)
        if type(b) == int:
            C.append([b])
        else:
            C.append(b)

        fus = fuse(a, b)  # fuse this pair as a list
        V.append(fus)
        if ismember(a, V) is True and ismember(b, V) is True:
            V.remove(a)
            V.remove(b)

    for subset in V:
        if type(subset) == int:
            C.append([subset])
        else:
            C.append(subset)


    #  Once we have the list of candidates, we return the best one
    max_value = -np.Inf
    subset_opt = []
    cluster_max = 0
    partition_value = 0
    for subset in C:
        cluster_value = F(SS, subset)
        subset_value = cluster_value + F(SS, diff(range(dim), subset))
        if subset_value > max_value:
            subset_opt = subset
            partition_value = subset_value
            cluster_max = cluster_value
            max_value = subset_value

    return subset_opt, partition_value, cluster_max

############################################# MI ########################################################

def mutualInformation (x, y):
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) 
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) 
    for i in range(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list])
        t = pxy[Py>0.]/Py[Py>0.] /Px[i]
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) 
    return sum_mi   


#################################### RMCMC ##############################################################
def log_prob(x):
  return -0.5 * np.sum(x ** 2)


def proposal(x, stepsize):
  return np.random.uniform(low=x - 0.5 * stepsize,high=x + 0.5 * stepsize,size=x.shape)


def p_acc_MH(x_new, x_old, log_prob):
  return min(1, np.exp(log_prob(x_new) - log_prob(x_old)))


def sample_MH(x_old, log_prob, stepsize):
  x_new = proposal(x_old, stepsize)
  # here we determine whether we accept the new state or not:
  # we draw a random number uniformly from [0,1] and compare
  # it with the acceptance probability
  accept = np.random.random() < p_acc_MH(x_new, x_old, log_prob)
  if accept:
      return accept, x_new
  else:
      return accept, x_old


def build_MH_chain(init, stepsize, n_total, log_prob):

  n_accepted = 0
  chain = [init]

  for _ in range(n_total):
    accept, state = sample_MH(chain[-1], log_prob, stepsize)
    chain.append(state)
    n_accepted += accept

  acceptance_rate = n_accepted / float(n_total)

  return chain, acceptance_rate

if __name__ == "__main__":
    x=  np.array([1, 0, 1, 0,0,1])
 
    chain, acceptance_rate = build_MH_chain(x, 3 , len(x), log_prob)

    print("Acceptance rate: {:.4f}".format(acceptance_rate))
    print(chain)

"""  
from re import M

def pendent_pair(Vprime, V, S, f, params=None):
   
    x = 0
    vnew = Vprime[x]
    n = len(Vprime)
    Wi = []
    used = np.zeros((n,1))
    used[x] = 1

    for i in range(n-1):
        vold = vnew
        Wi = Wi + S[vold]

        ## update keys
        keys = np.ones((n,1))*np.inf
        for j in range(n):
            if used[j]:
                continue
            keys[j] = f(Wi + S[Vprime[j]], V, params) - f(S[Vprime[j]], V, params)

        ## extract min
        argmin = np.argmin(keys)
        vnew = Vprime[argmin]
        used[argmin] = 1
        fval = np.min(keys)

    s = vold
    t = vnew

    return s, t, fval


def diff(A, B):
    
    m = np.amax(np.array([np.amax(A), np.amax(B)]))
    vals = np.zeros((m+1,1))
    vals[A] = 1
    vals[B] = 0
    idx = np.nonzero(vals)
    return idx[0]


def optimal_set(V, f, params=None):
   
    n = len(V)
    S = [[] for _ in range(n)]
    for i in range(n):
        S[i] = [V[i]]

    p = np.zeros((n-1,1))
    A = []
    idxs = range(n)
    for i in range(n-1):
        ## find a pendant pair
        t, u, fval = pendent_pair(idxs, V, S, f, params)

        ## candidate solution
        A.append(S[u])
        p[i] = f(S[u],V,params)
        S[t] = [*S[t], *S[u]]
        idxs = diff(idxs, u)
        S[u] = []
     

    ## return minimum solution
    i = np.argmin(p)
    R = A[i]
    fval = p[i]

    ## make R look pretty
    notR = diff(V,R)
   ## R = sorted(R)
   ## notR = sorted(notR)

    if R[0] < notR[0]:
        R = (tuple(R),tuple(notR))
    else:
        R = (tuple(notR),tuple(R))

    return R, fval


def k_subset(s, k):
    if k == len(s):
        return (tuple([(x,) for x in s]),)
    k_subs = []
    for i in range(len(s)):
        partials = k_subset(s[0:i] + s[i + 1:len(s)], k)
        for partial in partials:
            for p in range(len(partial)):
                k_subs.append(partial[:p] + (partial[p] + (s[i],),) + partial[p + 1:])
    return k_subs


def uniq_subsets(s):
    u = set()
    for x in s:
        t = []
        for y in x:
            y = list(y)
            y.sort()
            t.append(tuple(y))
        t.sort()
        u.add(tuple(t))
    return u


def find_partitions(V,k):
    
    k_subs = k_subset(V,k)
    k_subs = uniq_subsets(k_subs)

    return k_subs


def intersection(a, b):
   
    return list(set(a) & set(b))


def union(a, b):
   
    return list(set(a) | set(b))

    V = ([1,2,3,1,4,6,7])
    v1= np.array([0,0,1,1,0])
    v2= np.array([0,0,1,1,0])
    f = lambda S, V, params: mutualInformation(v1,v2)
    R, fval = optimal_set(V, f)
    print(R, fval)

"""
    



