'''
Functions for quality assurance of station location metadata
(longitude, latitude, elevation). In most cases, elevation is correct, but
longitude and/or latitude are either imprecise or incorrect. The functions can
be used to compare the provided elevation of a station to that of a high
resolution DEM. Those stations that have an elevation that significantly differs
from the corresponding DEM elevation likely have issues with their metadata. 
'''


__all__ = ['get_elevation', 'LocQA']

from io import BytesIO
from time import sleep
import json
import numpy as np
import pandas as pd
import pycurl
import urllib

def _get_elev_usgs(lon, lat, maxtries):
    """Get elev value from USGS NED 1/3 arc-sec DEM.

    http://ned.usgs.gov/epqs/
    """

    USGS_NED_NODATA = -1000000

    # url GET args
    values = {'x': lon,
              'y': lat,
              'units': 'Meters',
              'output': 'json'}
    
    data = urllib.parse.urlencode(values)
    
    req_url = "http://ned.usgs.gov/epqs/pqs.php?%s" % (data,)
                
    ntries = 0
    
    while 1:
        
        try:
            
            # Use pycurl instead of urllib2 due to sporadic performance issues
            # with urllib2
            
            buf = BytesIO()

            c = pycurl.Curl()
            c.setopt(pycurl.WRITEDATA, buf)
            c.setopt(pycurl.URL, req_url)
            c.setopt(pycurl.FAILONERROR, True)
            c.perform()
            c.close()
            
            break

        except pycurl.error:
            
            ntries += 1
        
            if ntries >= maxtries:
        
                raise
        
            sleep(1)

    json_response = json.loads(buf.getvalue().decode('UTF-8'))
    elev = np.float(json_response['USGS_Elevation_Point_Query_Service']
                    ['Elevation_Query']['Elevation'])

    if elev == USGS_NED_NODATA:

        elev = np.nan

    return elev

def _get_elev_geonames(lon, lat, usrname_geonames, maxtries):
    """Get elev value from geonames web sevice (SRTM or ASTER)
    """

    URL_GEONAMES_SRTM = 'http://api.geonames.org/srtm3'
    URL_GEONAMES_ASTER = 'http://api.geonames.org/astergdem'

    url = URL_GEONAMES_SRTM

    while 1:
        # ?lat=50.01&lng=10.2&username=demo
        # url GET args
        values = {'lat': lat, 'lng': lon, 'username': usrname_geonames}

        # encode the GET arguments
        
        data = urllib.parse.urlencode(values)

        # make the URL into a qualified GET statement
        get_url = "".join([url, "?", data])
        
        req = urllib.request.Request(url=get_url)
        
        ntries = 0
        
        while 1:
            
            try:
                
                response = urllib.request.urlopen(req)
                break

            except urllib.error.HTTPError:
                
                ntries += 1
            
                if ntries >= maxtries:
            
                    raise
            
                sleep(1)
        
        elev = float(response.read().strip())

        if elev == -32768.0 and url == URL_GEONAMES_SRTM:
            # Try ASTER instead
            url = URL_GEONAMES_ASTER
        else:
            break

    if elev == -32768.0 or elev == -9999.0:
        elev = np.nan

    return elev

def get_elevation(lon, lat, usrname_geonames=None, maxtries=3):

    elev = _get_elev_usgs(lon, lat, maxtries)

    if np.isnan(elev) and usrname_geonames is not None:

        elev = _get_elev_geonames(lon, lat, usrname_geonames, maxtries)

    return elev

class LocQA(object):
    '''Class for managing location quality assurance HDF database
    '''
    
    _cols_locqa = ['station_id','station_name', 'longitude', 'latitude',
                   'elevation','elevation_dem','longitude_qa','latitude_qa',
                   'elevation_qa']

    def __init__(self, fpath_locqa_hdf, mode='a', usrname_geonames=None):
        
        self.fpath_locqa_hdf = fpath_locqa_hdf
        self.usrname_geonames = usrname_geonames
        
        self._store = pd.HDFStore(fpath_locqa_hdf, mode)
        self.reload_stns_locqa()
                    
    def reload_stns_locqa(self):
        
        try:
            
            self._stns_locqa = self._store.select('stns')
            
        except KeyError:

            self._stns_locqa = pd.DataFrame(columns=self._cols_locqa)
            
            # Make sure numeric columns are float
            cols_flt = ['longitude','latitude','elevation','elevation_dem',
                        'longitude_qa','latitude_qa','elevation_qa']
            
            self._stns_locqa[cols_flt] = self._stns_locqa[cols_flt].astype(np.float)
        
        self._stns_locqa = (self._stns_locqa[~self._stns_locqa.index.
                                             duplicated(keep='last')].copy())
        
    
    def get_elevation_dem(self, lon, lat):
        
        return get_elevation(lon, lat, self.usrname_geonames)
        
    def update_locqa_hdf(self, stns, reload_locqa=True):
        
        self._store.append('stns', stns[self._cols_locqa], min_itemsize={'station_id':50,
                                                                         'station_name':50,
                                                                         'index':50})
        self._store.flush()
        
        if reload_locqa:
        
            self.reload_stns_locqa()
    
    def add_locqa_cols(self, stns):
        
        stns = stns.copy()
        
        locqa_cols = ['elevation_dem', 'longitude_qa', 'latitude_qa', 'elevation_qa']
        isclose_cols = ['longitude', 'latitude', 'elevation']
        
        a_stns = stns[isclose_cols].join(self._stns_locqa, how='left', rsuffix='_')
                
        for c in locqa_cols:
            stns[c] = a_stns[c]
                                        
        return stns
        
    def get_locqa_fail_stns(self, stns, elev_dif_thres=200):
        
        stns = stns.copy()
        stns['elevation_dif'] = stns.elevation - stns.elevation_dem 
        
        mask_fail = ((stns.elevation_dif.abs() > elev_dif_thres) |
                     (stns.elevation_dif.isnull())).values
        mask_noqa = ((stns.longitude_qa.isnull()) | (stns.latitude_qa.isnull())
                     | (stns.elevation_qa.isnull())).values
        mask_fnl = np.logical_and(mask_fail, mask_noqa)
                     
        return stns[mask_fnl].copy()
            
    def close(self):
        
        self._store.close()
        self._store = None
    