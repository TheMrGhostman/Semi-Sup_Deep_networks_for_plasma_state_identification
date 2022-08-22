import numpy as np

"""
Time series precision metric implemented according to paper :

\"Precision and Recall for Time Series\", 
Nesime Tatbul, Tae Jun Lee, Stan Zdonik, Mejbah Alam, Justin Gottschlich, 
32nd Annual Conference on Neural Information Processing Systems (NeurIPS'18), 
Montreal, Canada, December 2018. (https://arxiv.org/abs/1803.03639/)
"""


delta_flat = lambda i, anomaly_length: 1
delta_front_end = lambda i, anomaly_length: anomaly_length - i + 1
delta_back_end = lambda i, anomaly_length: i
delta_middle = lambda i, anomaly_length: i if i <= anomaly_length/2 else anomaly_length -i + 1 

gamma_one = lambda x: 1
gamma_reciprocal = lambda x: 1/x if x > 0 else 1


def get_delta(task_type):
    if task_type == "flat":
        return delta_flat
    elif task_type == "front-end":
        return delta_front_end
    elif task_type == "back-end":
        return delta_back_end
    elif task_type == "middle":
        return delta_middle
    else:
        raise ValueError("Unknown type of task! Use: \"flat\", \"front-end\", \"back-end\" or \"middle\" ")


def omega(Range, OverlapSet, delta):
    my_val = 0
    max_val = 0
    anomaly_length = Range[1] - Range[0] + 1
    for i in range(1, anomaly_length + 1):
        bias = delta(i, anomaly_length)
        max_val += bias
        j = Range[0] + i -1  
        if j >= OverlapSet[0] and j <= OverlapSet[1]:
            my_val += bias
    if max_val > 0:
        return my_val / max_val
    else:
        return 0
    

def omega_reward(Range1, Range2, overlap_count, delta):
    if Range1[1] < Range2[0] or Range1[0] > Range2[1]:
        return 0
    else:
        overlap_count[0] += 1
        overlap = np.array([max(Range1[0], Range2[0]), min(Range1[1], Range2[1])])
        #print(overlap)
        return omega(Range1, overlap, delta)


def Precision_T(R, P, delta=delta_flat, gamma=gamma_reciprocal):
    precision = 0
    if len(P) == 0:
        return 0
    for Pi in P:
        omega_reward_pi = 0
        overlap_count = [0] # made as list insted of global variable
        for Ri in R:
            omega_reward_pi += omega_reward(Pi, Ri, overlap_count, delta)
        PrecisionRPi = omega_reward_pi * gamma(overlap_count[0])
        #print(omega_reward_, PrecisionRPi, gamma(overlap_count[0]), overlap_count[0])
        precision += PrecisionRPi
    return precision / len(P)


def get_ranges(y, class_idx):
    """
    Function takes 1D label array \"y\", find indexes with value equal to \"class_idx\"
        and turn them into ranges.
    Example:
        >>> x = np.hstack((np.zeros(3), np.ones(3), 2*np.ones(3), np.zeros(3)))
        >>> x
        array([0., 0., 0., 1., 1., 1., 2., 2., 2., 0., 0., 0.])
        >>> get_ranges(x, 0)
        array([[0,  2],
               [9, 11]], dtype=int64)
        >>> get_ranges(x, 1)
        array([[3, 5]], dtype=int64)
        >>> get_ranges(x, 1)
        array([[6, 8]], dtype=int64)
    """
    def shift(arr, num, fill_value=np.nan):
        arr = np.roll(arr,num)
        if num < 0:
            arr[num:] = fill_value
        elif num > 0:
            arr[:num] = fill_value
        return arr
    
    y = np.array(y)
    y_ = np.argwhere(y == class_idx).ravel()
    y_shift_forward = shift(y_, 1, fill_value=y_[0])
    y_shift_backward = shift(y_, -1, fill_value=y_[-1])
    y_start = np.argwhere((y_shift_forward - y_) != -1).ravel()
    y_finish = np.argwhere((y_ - y_shift_backward) != -1).ravel()
    return np.hstack([y_[y_start].reshape(-1, 1), y_[y_finish].reshape(-1, 1)])


""" Following metrics are simple "transition" Precision and Recall"""
def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def transition_preprocess(y_pred, y_true, mask=1, verbose=False):
    """
    Function preprocess and locate all transitions in input arrays/tensors.
    If "mask" is set to 1, it detect every (even one-point) transitions. 
    Value of "mask" therefore states how many consecutive same class predictions 
    are needed to accept transition.

    Args:
        y_pred (numpy.ndarray):     Binary states predicted by model.
        y_true (numpy.ndarray):     Real binary states. Ground Truth.
        mask (int, optional):       Threshold for acceptance of transitions. Defaults to 1 (the simplest case).
        verbose (bool, optional):   Allow function to print logs or not. Defaults to False.

    Returns:
        (y_p, y_t) (numpy.ndarray, numpy.ndarray) : Indexes of input array where transition was detected. 
    """
    
    # extract transitions from ground truth (there is no threshold for GT)
    y_t = np.convolve(y_true, [1,-1], "same")
    y_t[0] = 0
    if verbose:
        print(y_t)
    y_t = np.where(y_t == 1)[0]
    
    # extracting transitions from model prediciton
    # running mask via convolution through array. 
    # example: mask = 3
    # y_pred = [0,0,0,0,1,1,1,1,0,0,0,1,1,0]  ... at least 3 (mask) transition
    # will demonstrate itself after convolution as (0,1,2,3)
    # [0, 0, 0, 1, 2, 3, 3, 2, 1, 0, 1, 2, 2, 1] 
    # "transition-characteristic" sequence (0,1,2,3) appears only onece in array
    # that means there is only one transition with lenght mask (3)  
    y_p = np.convolve(y_pred, np.ones(mask), "same")
    #y_p[0]=0
    if verbose:
        print(y_p)
    y_p = rolling_window(y_p, mask+1) # spliting into sequences of length mask
    if verbose:
        print(y_p)
    # computing similiarity between splitted sequences and "transition-characteristic" 
    # accepting only full match
    y_p = np.sum(y_p==np.arange(mask+1), axis=1)
    # computing indexes of that match (if any) and correcting them by mask//2
    y_p = np.where(y_p==mask+1)[0] + int((mask+1)/2)
    if verbose:
        print(y_p, y_t)
    return y_p, y_t


def transition_metrics(mtype, y_pred, y_true, tau=3, mask=1, aggregation="mean", verbose=False):
    """
    Wrapper function around "transition_preprocess" that allows to compute "transition_precision" and "transition_recall"

    Args:
        mtype (str):                    Type of metrics to compute. Transition precision ("precision") or transition recall ("recall")
        y_pred (numpy.ndarray):         Binary states predicted by model.
        y_true (numpy.ndarray):         Real binary states. Ground Truth.
        tau (int, tuple, optional):     Window length (range) in which transition is considered correctly detected. Defaults to 3.
        mask (int, optional):           Threshold for acceptance of transitions. Defaults to 1 (the simplest case).
        aggregation (str, optional):    Type of aggregation. "mean", "sum" and "pre-mean". 
                                        Pre-mean returns tuple of integers (sum of correctly detected, total number of transitions). 
                                        Defaults to "mean".
        verbose (bool, optional):       Allow function to print logs or not. Defaults to False.

    Raises:
        ValueError: Unknown type of "tau".
        ValueError: Unknown/Non-imlemented type of metrics ("mtype").

    Returns:
        (float, (int, int)):    Value of "mtype" metrics in form of "aggregation". 
    """
    
    if isinstance(tau, int):
        tau1 = tau2 = tau 
    elif isinstance(tau, tuple) and len(tau)==2:
        tau1, tau2 = tau
    else:
        raise ValueError("Unknown tau")
        
    if  mtype=="precision": 
        # TODO fix -- but NO CHANGE IN PRECISION
        y_p, _ = transition_preprocess(y_pred, y_true, mask=mask, verbose=verbose)
        y_t, _ = transition_preprocess(y_true, y_pred, mask=mask, verbose=verbose) # preprocess y_true too

        
    elif mtype=="recall":
        # TODO fix -- MINIMAL CHANGE IN RECALL (+/- 1%)
        y_t, _ = transition_preprocess(y_pred, y_true, mask=mask, verbose=verbose)
        y_p, _ = transition_preprocess(y_true, y_pred, mask=mask, verbose=verbose) # preprocess y_true too
    else:
        raise ValueError("unknown metrics")
    
    detected = 0
    for i in y_p:
        for j in np.arange(i-tau1, i+tau2+1):
            if j in y_t:
                detected += 1
                break 
    try:
        if aggregation=="mean":
            return detected/len(y_p)
        elif aggregation=="sum":
            return detected
        elif aggregation=="pre-mean":
            return detected, len(y_p)
        else:
            return 0
    except (TypeError, ZeroDivisionError):
        return 0


def LH_precision(y_pred, y_true, tau=3, mask=1, aggregation="mean"):
    y_t = np.copy(y_true)
    y_p = np.copy(y_pred)
    # ELM start & ELM end are still "parts" of H-mod
    y_p[(y_p==0) | (y_p==2) | (y_p==3)] = -1 
    y_t[(y_t==0) | (y_t==2) | (y_t==3)] = -1
    # recreate array to be L-mode ~ 0 and H-mode ~ 1
    # L-mode ~ 0 
    y_p[y_p==1] = 0
    y_t[y_t==1] = 0
    #H-mode ~ 1
    y_p[y_p==-1] = 1
    y_t[y_t==-1] = 1
    return transition_metrics("precision", y_pred=y_p, y_true=y_t, tau=tau, mask=mask, aggregation=aggregation)


def LH_recall(y_pred, y_true, tau=3, mask=1, aggregation="mean"):
    y_t = np.copy(y_true)
    y_p = np.copy(y_pred)
    # ELM start & ELM end are still "parts" of H-mod
    y_p[(y_p==0) | (y_p==2) | (y_p==3)] = -1 
    y_t[(y_t==0) | (y_t==2) | (y_t==3)] = -1
    # recreate array to be L-mode ~ 0 and H-mode ~ 1
    # L-mode ~ 0 
    y_p[y_p==1] = 0
    y_t[y_t==1] = 0
    #H-mode ~ 1
    y_p[y_p==-1] = 1
    y_t[y_t==-1] = 1
    return transition_metrics("recall", y_pred=y_p, y_true=y_t, tau=tau, mask=mask, aggregation=aggregation)


def HElm_precision(y_pred, y_true, tau=3, mask=1, aggregation="mean"):
    y_t = np.copy(y_true)
    y_p = np.copy(y_pred)
    # discarted ELM_end and mark it as L-mod for example, because we will ignore L-mods
    # setting -1 to L-mode and ELM-trailing, those two states will be ignored
    y_p[(y_p==3) | (y_p==1)] = -1 
    y_t[(y_t==3) | (y_t==1)] = -1
    # ELM_leading to 1, H-mode is already 0
    y_p[y_p==2] = 1
    y_t[y_t==2] = 1
    return transition_metrics("precision",y_pred=y_p, y_true=y_t, tau=tau, mask=mask, aggregation=aggregation)


def HElm_recall(y_pred, y_true, tau=3, mask=1, aggregation="mean"):
    y_t = np.copy(y_true)
    y_p = np.copy(y_pred)
    # discarted ELM_end and mark it as L-mod for example, because we will ignore L-mods
    # setting -1 to L-mode and ELM-trailing, those two states will be ignored
    y_p[(y_p==3) | (y_p==1)] = -1 
    y_t[(y_t==3) | (y_t==1)] = -1
    # ELM_leading to 1, H-mode is already 0
    y_p[y_p==2] = 1
    y_t[y_t==2] = 1
    return transition_metrics("recall",y_pred=y_p, y_true=y_t, tau=tau, mask=mask, aggregation=aggregation)