'''
Python implementation of the Livneh/Maurer SYMAP inverse distance weighting
interpolation algorithm.

Livneh, B., T. J. Bohn, D. W. Pierce, F. Munoz-Arriola, B. Nijssen, R. Vose,
D. R. Cayan, and L. Brekke, 2015: A spatially comprehensive, hydrometeorological
data set for Mexico, the U.S., and Southern Canada 1950-2013. Scientific Data,
2, 150042.

Original C code:

ftp://192.12.137.7/pub/dcp/archive/OBS/livneh2014.1_16deg/gridding.codes.livneh.2015/

'''

import numpy as np

_DTOR = np.pi/180.0
_RADIUS = 6378.0
_CIRCUM = 2.0*np.pi*_RADIUS

def _distance(lat1, lon1, lat2, lon2):
    
    theta1 = _DTOR * lon1
    phi1 = _DTOR * lat1
    theta2 = _DTOR * lon2
    phi2 = _DTOR * lat2

    term1 = np.cos(phi1) * np.cos(theta1) * np.cos(phi2) * np.cos(theta2)
    term2 = np.cos(phi1) * np.sin(theta1) * np.cos(phi2) * np.sin(theta2)
    term3 = np.sin(phi1) * np.sin(phi2)
    sumterm = term1+term2+term3

    try:
        sumterm[sumterm > 1.0] = 1.0
    except TypeError:
        sumterm = sumterm if sumterm <= 1.0 else 1.0

    dist = _RADIUS*np.arccos(sumterm)

    return dist

def _calc_theta(lat_pt, lon_pt, lat_tstn, lon_tstn, lat_ostn, lon_ostn):

    a = _distance(lat_tstn, lon_tstn, lat_ostn, lon_ostn)
    b = _distance(lat_pt, lon_pt, lat_ostn, lon_ostn)
    c = _distance(lat_pt, lon_pt, lat_tstn, lon_tstn)

    a = a/_CIRCUM*360.0*_DTOR
    b = b/_CIRCUM*360.0*_DTOR
    c = c/_CIRCUM*360.0*_DTOR

    b_c = np.sin(b)*np.sin(c)

    try:
        b_c[b_c < 1.0e-4] = 1.0e-4
    except TypeError:
        b_c = b_c if b_c >= 1.0e-4 else 1.0e-4

    iw=(np.cos(a)-np.cos(b)*np.cos(c))/b_c

    try:
        b_c[b_c < 1.0e-4] = 1.0e-4
    except TypeError:
        b_c = b_c if b_c >= 1.0e-4 else 1.0e-4

    try:
        amask = np.abs(iw) > 1.0
        iw[amask] = np.round(iw[amask])
    except TypeError:

        if np.abs(iw) > 1.0:
            iw = np.round(iw)

    iw = 1.0-iw

    try:
        iw[iw < 1.0e-6] = 1.0e-6
    except TypeError:
        iw = iw if iw >= 1.0e-6 else 1.0e-6

    return iw


def _calc_t(lat_pt, lon_pt, lat_tstn, lon_tstn, lat_ostn, lon_ostn):

    t = np.sum((1.0/_distance(lat_pt,lon_pt,lat_ostn,lon_ostn)) *
               _calc_theta(lat_pt,lon_pt,lat_tstn,lon_tstn,lat_ostn,lon_ostn))

    return t


def _calc_h(lat_pt, lon_pt, lat_tstn, lon_tstn, lat_ostn, lon_ostn):
    h = (np.sum(1.0 / _distance(lat_pt, lon_pt, lat_ostn, lon_ostn)) +
         (1.0/_distance(lat_pt, lon_pt, lat_tstn, lon_tstn)))
    return h

def _calc_w(lat_pt, lon_pt, lat_tstn, lon_tstn, lat_ostn, lon_ostn):

    t = _calc_t(lat_pt,lon_pt,lat_tstn,lon_tstn,lat_ostn,lon_ostn)
    h = _calc_h(lat_pt,lon_pt,lat_tstn,lon_tstn,lat_ostn,lon_ostn)

    return 1.0/(_distance(lat_pt, lon_pt, lat_tstn, lon_tstn)**2)*(t+h)

def _calc_regrid_val(lat_pt, lon_pt, elev_pt, lat_stn, lon_stn, elev_stn, obs_stn, apply_tair_lapse=False):

    idx = np.arange(obs_stn.size)

    w = np.empty(idx.size)

    for i in idx:

        idx_ostns = np.nonzero(idx!=i)[0]

        w[i] = _calc_w(lat_pt,lon_pt,lat_stn[i],lon_stn[i],np.take(lat_stn,idx_ostns),np.take(lon_stn,idx_ostns))

    if apply_tair_lapse:
        return np.average(obs_stn+0.0065*(elev_stn - elev_pt),weights=w)
    else:
        return np.average(obs_stn,weights=w)
    
def symap_prcp(lat_pt, lon_pt, lat_stn, lon_stn, obs_stn, ninterp=4):
    '''
    Interpolate a single precipitation value to a point using the Livneh et al. SYMAP method

    Parameters
    ----------
    lat_pt : float
        Latitude of of interpolation point
    lon_pt: float
        Longitude of of interpolation point
    lat_stn: numpy.array
        Latitudes of stations with precipitation observations
    lon_stn: numpy.array
        Longitudes of stations with precipitation observations
    obs_stn : numpy array
        Precipitation observations of the stations
    ninterp : int, optional
        The number of closest stations to use for interpolation. Method will find
        the ninterp closest stations of those provided and use their observations
        for the interpolation. Default is 4 as used by Livneh et al.
    
    Returns
    ----------
    prcp : float
    '''
    
    
    d = _distance(lat_pt, lon_pt, lat_stn, lon_stn)
    idx = np.argsort(d)[0:ninterp]
    
    lat_stn = np.take(lat_stn, idx)
    lon_stn = np.take(lon_stn, idx)
    obs_stn = np.take(obs_stn, idx)
    
    elev_stn = np.empty_like(lon_stn)
    elev_stn.fill(np.nan)
    elev_pt = np.nan
    
    return _calc_regrid_val(lat_pt, lon_pt, elev_pt, lat_stn, lon_stn, elev_stn,
                            obs_stn, apply_tair_lapse=False)
    