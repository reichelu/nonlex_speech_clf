import math
import numpy as np
import os

def time2idx(t, sr):
    return np.array([int(np.floor(t[0] * sr)),
                     int(np.ceil(t[1] * sr))])

def stm(x):
    ''' returns file name stem '''
    return os.path.splitext(os.path.basename(x))[0]

def opt_default(c, d):
    r"""recursively adds default fields of dict d to dict c
      if not yet specified in c
    Arguments:
      c: (dict) someDict
      d: (dict) defaultDict
    Returns:
      c: (dict) mergedDict; defaults added to c
    """
    for x in d:
        if x not in c:
            c[x] = d[x]
        elif type(c[x]) is dict:
            c[x] = opt_default(c[x], d[x])
    return c

def idx2sec(i, fs, ons=0):

    '''
    transforms numpy array indices (=samplesIdx-1) to seconds
    
    Args:
    i: (int) index
    fs: (int) sampling rate
    ons: (int) onset index

    Returns:
    s: (float) time in seconds

    '''
    
    return (i + 1 + ons) / fs

def rmsd(x, y=None):

    '''
    returns RMSD of two vectors x and y
    
    Args:
     x: (np.array)
     y: (np.array) <zeros(len(x))>
    
    Returns:
     (float) root mean squared dev between x and y
    '''
    
    if y is None:
        y = np.zeros(len(x))

    x = np.array(x)
    y = np.array(y)
        
    return np.sqrt(np.mean((x - y) ** 2))

def windowing_idx(i, s):

    '''
    as windowing(), but returning all indices from onset to offset

    Args:
      i: (int) current index
      s: (dict)
       .win window length
       .rng [on, off] range of indices to be windowed
    
    Returns:
      (np.array) [on:1:off] in window around i

    '''

    on, off = windowing(i, s)
    return np.arange(on, off, 1)

def windowing(i, o):

    '''
    window of length o["win"] on and offset around single index
    limited by range o["rng"]. vectorized version: seq_windowing()
    
    Args:
      i: (array) indices
      o: (dict)
        win: (int) window length in samples
        rng: (list) [on, off] range of indices to be windowed
    
    Returns:
      on, off (int-s) indices of window around i
    '''

    win = o["win"]
    rng = o["rng"]
    
    # half window
    wl = max([1, math.floor(win / 2)])
    on = max([rng[0], i - wl])
    off = min([i + wl, rng[1]])
    
    # extend window
    d = (2 * wl - 1) - (off - on)
    if d > 0:
        if on > rng[0]:
            on = max([rng[0], on - d])
        elif off < rng[1]:
            off = min([off + d, rng[1]])

    return on, off


def idx_a(l, sts=1):

    '''
    returns index array for vector of length len() l
    thus highest idx is l-1

    Args:
    l: (int) length
    sts: (int) stepsite

    Returns:
    (np.arange)
    '''
    
    return np.arange(0, l, sts)

def sec2smp(i, fs, ons=0.0):

    '''
    transforms seconds to sample indices (arrayIdx+1)

    Args:
    i: (float) time in sec
    fs: (int) sampling rate
    ons: (float) time onset to be added in sec

    Returns:
    (int) sample index
    '''
    
    return np.round(i * fs + ons).astype(int)

def push(x, y, a=0):

    '''
    pushes 1 additional element y to array x (default: row-wise)
      if x is not empty, i.e. not []: yDim must be xDim-1, e.g.
          if x 1-dim: y must be scalar
          if x 2-dim: y must 1-dim
    if x is empty, i.e. [], the dimension of the output is yDim+1
    Differences to np.append:
      append flattens arrays if dimension of x,y differ, push does not
    REMARK: cmat() might be more appropriate if 2-dim is to be returned
    
    Args:
      x: (np.array) (can be empty)
      y: (np.array) (if x not empty, then one dimension less than x)
      a: (int) axis (0: push row, 1: push column)
    
    Returns:
      (np.array) [x y] concatenation
    '''

    if (type(y) in [list, np.array] and len(y) == 0):
        return x
    if len(x) == 0:
        return np.array([y])
    return np.concatenate((x, [y]), axis=a)

