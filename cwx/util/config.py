from copy import copy
try:
    import configparser as cp
except ImportError:
    import ConfigParser as cp
import numpy as np
import pandas as pd

class ChesWxConfig():
    
    def __init__(self, fpath_ini):
        
        try:
        
            cfg = cp.ConfigParser()
            cfg.read(fpath_ini)
            has_config = True
        
        except TypeError:
        
            has_config = False

        if has_config:
            
            self.data_root = cfg.get('CHESWX', 'data_root')
            self.wet_thres = float(cfg.get('CHESWX', 'wet_thres'))
            
            bbox_str = cfg.get('CHESWX', 'stn_bbox')
            self.stn_bbox = tuple([np.float(i) for i in bbox_str.split(',')])
            bbox_str = cfg.get('CHESWX', 'interp_bbox')
            self.interp_bbox = tuple([np.float(i) for i in bbox_str.split(',')])
            
            self.obs_start_date = pd.Timestamp(cfg.get('CHESWX', 'obs_start_date'))
            self.obs_end_date = pd.Timestamp(cfg.get('CHESWX', 'obs_end_date'))
            self.interp_start_date = pd.Timestamp(cfg.get('CHESWX', 'interp_start_date'))
            self.interp_end_date = pd.Timestamp(cfg.get('CHESWX', 'interp_end_date'))
            
            self.stn_write_chunk_nc = int(cfg.get('CHESWX', 'stn_write_chunk_nc'))
            self.stn_agg_chunk = int(cfg.get('CHESWX', 'stn_agg_chunk'))
            
            self.username_geonames = cfg.get('CHESWX', 'username_geonames')
        
        else:

            print("WARNING: No configuration file found. All config options will be None")
            self.data_root = None
            self.wet_thres = None
            self.stn_bbox = None
            self.interp_bbox = None
            self.obs_start_date = None
            self.obs_end_date = None
            self.interp_start_date = None
            self.interp_end_date = None
            self.stn_write_chunk_nc = None
            self.stn_agg_chunk = None
            self.username_geonames = None
        
    def to_str_dict(self):
        
        a_dict = copy(self.__dict__)
        for a_key in a_dict.keys():
            a_dict[a_key] = str(a_dict[a_key])
        return a_dict