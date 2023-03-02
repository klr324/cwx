import numpy as np
import scipy.stats as sstats

def hss(o, p, wett=0):
    '''Calculate Heidke Skill Score for wet day occurrence
    '''
    mask_fin = np.nonzero(np.logical_and(np.isfinite(o),
                                         np.isfinite(p)))[0]
    
    obs = np.take(o, mask_fin)
    predicted = np.take(p, mask_fin)
    obs = obs > wett
    predicted = predicted > wett

    a = np.sum(np.logical_and(obs, predicted))
    b = np.sum(np.logical_and(~obs, predicted))
    c = np.sum(np.logical_and(obs, ~predicted))
    d = np.sum(np.logical_and(~obs, ~predicted))
    
    num = 2 * ((a * d) - (b * c))
    denum = ((a + c) * (c + d)) + ((a + b) * (b + d))
    
    hss = np.float(num) / denum

    return hss

def pearsonr(o, p):
       
    mask_fin = np.nonzero(np.logical_and(np.isfinite(o),
                                         np.isfinite(p)))[0]
    obs = np.take(o, mask_fin)
    predicted = np.take(p, mask_fin)
    return sstats.pearsonr(obs, predicted)[0]