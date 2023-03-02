from __future__ import print_function
from datetime import datetime
import errno
import numpy as np
import os
import sys
import time
import xarray as xr

def runs_of_ones_array(bits):
    '''Find length of identical value sequences in a binary numpy array
    
    http://stackoverflow.com/questions/1066758/
    find-length-of-sequences-of-identical-values-in-a-numpy-array
    '''
    #make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts

def mkdir_p(path):
    '''
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise 
        
def read_xarray_netcdfs(fpaths, dim, transform_func=None, engine=None, verbose=False):
    '''
    Based off example at: http://xarray.pydata.org/en/stable/io.html#combining-multiple-files
    '''
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
	# klr 12/5/2022: open_dataset() got an unexpected keyword argument 'filter_by_keys'. Used above link example to fix this error.
        #with xr.open_dataset(path, engine=engine, filter_by_keys={'stepType': 'accum', 'typeOfLevel': 'surface'}) as ds:
        with xr.open_dataset(path) as ds:
    
            if verbose:
                print("Processing " + path)
            
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    datasets = [process_one_path(p) for p in fpaths]
    combined = xr.concat(datasets, dim)

    return combined

def ymdL(date):
    try:
        return int(datetime.strftime(date,"%Y%m%d"))
    except ValueError:
        return int("%d%02d%02d"%(date.year,date.month,date.day))

class StatusCheck(object):
    '''
    Class for printing out progress messages
    '''


    def __init__(self, total_cnt, check_cnt):
        '''
        total_cnt: the total number of items being processed
        check_cnt: the number of items completed at which a progress message
        should be printed
        '''
        self.total_cnt = total_cnt
        self.check_cnt = check_cnt
        self.num = 0 
        self.num_last_check = 0
        self.status_time = time.time()
        self.start_time = self.status_time
    
    def increment(self, n=1):
        
        self.num += n
        
        if self.num - self.num_last_check >= self.check_cnt:
            
            currentTime = time.time()
            
            if self.total_cnt != -1:
                
                print ("Total items processed is %d.  Last %d items took "
                       "%f minutes. %d items to go." % 
                       (self.num, self.num - self.num_last_check,
                        (currentTime - self.status_time) / 60.0,
                        self.total_cnt - self.num))
                print ("Current total process time: %f minutes" %
                       ((currentTime - self.start_time) / 60.0))
                print ("Estimated Time Remaining: %f" %
                       (((self.total_cnt - self.num) / float(self.num)) * 
                        ((currentTime - self.start_time) / 60.0)))
            
            else:
            
                print ("Total items processed is %d.  "
                       "Last %d items took %f minutes" %
                       (self.num, self.num - self.num_last_check,
                        (currentTime - self.status_time) / 60.0))
                print ("Current total process time: %f minutes" %
                       ((currentTime - self.start_time) / 60.0))
            
            sys.stdout.flush()
            self.status_time = time.time()
            self.num_last_check = self.num

def nn_xr_data_array(da, lon, lat, check_dist=True):
    
    if check_dist:
        
        tol = np.mean([float(da.lon.diff(dim='lon',n=1).mean()),
                       float(da.lat.diff(dim='lat',n=1).mean())])*1.1
                       
        try:
            
            a_nn = da.sel(lon=lon, lat=lat, method='nearest', tolerance=tol)
    
        except KeyError:
            
            raise KeyError('Could not determine nearest neighbor for %.2f,%.2f. '
                           'Point too far from nearest grid cell'%(lon,lat))
            
    else:
        
        a_nn = da.sel(lon=lon, lat=lat, method='nearest')
        
    return a_nn
