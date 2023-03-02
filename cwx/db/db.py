from ..util import StatusCheck, ymdL
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

def obs_cnts(ds_path, elem, start_date, end_date, stn_chk=500, status=True):
    '''Calculate the number of daily observations in each month
    
    For each station in a netCDF station database, calculates the number of
    daily observations in each month over a specified time period.
    
    Parameters
    ----------
    ds_path : str
        File path to a netCDF station database
    elem : str
        Element name for which to calculation observation count (e.g.-prcp)
    start_date : date-like
        The start date for time period over which to calculation obs counts
    end_date : date-like
        The end date for time period over which to calculation obs counts
    stn_chk : int, optional
        The number of stations to process in memory at a time. Default: 500.
        
    Returns
    ----------
    cnts : xarray DataArray
    '''
    
    ds = xr.open_dataset(ds_path)
    
    cnts = []
    
    if status:
        schk = StatusCheck(ds.station_id.size, stn_chk)
    
    for i in np.arange(ds.station_id.size, step=stn_chk):
    
        da = ds[elem][:, i:(i + stn_chk)].load().loc[start_date:end_date, :]
        
        cnt = da.groupby('time.month').count(dim='time')
        
        cnts.append(cnt)
        
        if status:
            schk.increment(stn_chk)
    
    cnts = xr.concat(cnts, dim='station_id')
    ds.close()
    del ds
    
    return cnts

def add_ncvar_obs_cnts(ds_path, elem, cnts, start_date, end_date):
    '''Add observation counts as a netCDF variable 
    
    Adds observation counts from twxp.db.obs_cnts to a netCDF station database
    as a variable of name: obs_cnt_[elem]_[ymd-start]_[ymd-end]
    
    Parameters
    ----------
    ds_path : str
        File path to a netCDF station database
    elem : str
        Element name observation counts (e.g.-prcp)
    cnts : xarray.DataArray
        Observation counts from twxp.db.obs_cnts
    start_date : date-like
        The start date for time period over obs counts were calculated
    end_date : date-like
        The end date for time period over which obs counts were calculated    
    '''
    
    ds = nc.Dataset(ds_path, 'r+')
    
    vname = "obs_cnt_%s_%d_%d" % (elem, ymdL(start_date), ymdL(end_date))
    
    if "mth" not in ds.dimensions.keys():
    
        ds.createDimension('mth', 12)
        vmth = ds.createVariable('mth', np.int, ('mth',),
                                 fill_value=False)
        vmth[:] = np.arange(1, 13)
        
    if vname not in ds.variables.keys():
        
        vcnt = ds.createVariable(vname, np.int, ('mth', 'station_id'),
                                 fill_value=False)
        vcnt.comments = "Number of observations per calendar month"
        
    else:
        
        vcnt = ds.variables[vname]
        
    vcnt[:] = cnts.values
    
    ds.sync()
    ds.close()
    
def _build_a_por_mask(obs_cnts, min_por_yrs):
    
    days_in_mth = np.array([d.days_in_month for d in
                            pd.date_range('2015-01', '2015-12', freq='MS')])
    
    nmin = days_in_mth * min_por_yrs
    nmin.shape = (nmin.size, 1)
    
    mask_por = obs_cnts >= nmin
    mask_por = np.sum(mask_por, axis=0) == 12

    return mask_por

def build_por_mask(ds, elems, start_date, end_date, min_por_yrs):
    '''Build a boolean mask for stations that have long enough period-of-record
    
    Requires obs_cnt_[elem]_[ymd-start]_[ymd-end] variable to be previously set
    by twxp.db.add_ncvar_obs_cnts
    
    Parameters
    ----------
    ds : netCDF4.Dataset
        Dataset pointing to station observation netCDF file
    elems : list
        List of element names for which to build the mask. A station will be
        included if has a long enough period-of-record for one or more of the
        specified elements.
    start_date : date-like
        The start date for period-of-record time period
    end_date : date-like
        The end date for period-of-record time period
    min_por_yrs : int
        The minimum period of record in years. The function tests
        whether a station has at least min_por_yrs years of data in each month.
    '''
    
    por_masks = []
    
    start_ymd = ymdL(start_date)
    end_ymd = ymdL(end_date)

    for elem in elems:

        vname = "obs_cnt_%s_%d_%d" % (elem, start_ymd, end_ymd)
        
        obs_cnts = ds[vname][:]
        
        mask_por = _build_a_por_mask(obs_cnts, min_por_yrs)
        
        por_masks.append(mask_por)

    por_masks = np.array(por_masks)

    por_mask_fnl = np.sum(por_masks, axis=0) >= 1
    
    return por_mask_fnl

def add_obs_variable(ds, varname, long_name, units, dtype,
                     fill_value=None, zlib=True, chunksizes=None,
                     reset=True):
    '''Add and initialize a 2D observation variable in a netCDF dataset
    
    Parameters
    ----------
    varname : str
        The name of the variable.
    long_name : str
        The long name of the variable.
    units : str
        The units of the variable.
    dtype : str
        The data type of the variable as a string.
    fill_value : int or float
        The fill or no data value for the variable.
        If None, the default netCDF4 fill value will be used
    zlib : boolean, optional
        Use zlib compression for the variable. Default: True
    chunksize: tuple of ints, optional
        Chunksize of the variable
    reset: boolean, optional
        Reset variable values if already exists. Default: True
        
    Returns
    -------
    newvar : netCDF4.Variable
        The new netCDF4 variable
    '''
    
    fill_value = nc.default_fillvals[dtype] if fill_value is None else fill_value
    
    if varname not in ds.variables.keys():
        
        newvar = ds.createVariable(varname, dtype, ('time', 'station_id'),
                                   fill_value=fill_value, zlib=zlib,
                                   chunksizes=chunksizes)
        newvar.long_name = long_name
        newvar.missing_value = fill_value
        newvar.units = units
                    
    else:
        
        newvar = ds.variables[varname]
        
        if reset:
        
            newvar[:] = fill_value
    
    ds.sync()
    
    return newvar

def create_stn_obs_db(fpath, stns, times, main_vars):
    '''Create netcdf dataset file for station observations.
    
    Parameters
    ----------
    fpath : str
        File path for the output netcdf file
    stns : pandas.DataFrame
        DataFrame of station metadata
    times : pandas.DatetimeIndex
        Index of times for time dimension
    main_vars : list of tuples
        list of 2 value tuples containing the name and datatype (as string) of
        the main 2D variables (time x station) to be created in the netcdf file.
        
    Returns
    -------
    ds_out : netCDF4.Dataset
        The created netCDF4 Dataset
    '''

    # Create output netcdf file
    ds_out = nc.Dataset(fpath, 'w')

    # Create station id dimension and add station metadata
    ds_out.createDimension('station_id', stns.shape[0])
         
    for acol in stns.columns:
    
        adtype = np.str if stns[acol].dtype == np.dtype('O') else stns[acol].dtype
        avar = ds_out.createVariable(acol, adtype, ('station_id',))
         
        if adtype == np.str:
            avar[:] = stns[acol].astype(np.str).values
        else:
            avar[:] = stns[acol].values
    
    ds_out.sync()   
                
    # Add station index number column    
    stns['station_num'] = np.arange(len(stns))
     
    ds_out.createDimension('time', times.size)
     
    # Create time dimension and variable
    times_var = ds_out.createVariable('time', 'f8', ('time',), fill_value=False)
    times_var.long_name = "time"
    times_var.units = "days since %d-%02d-%02d 0:0:0"%(times[0].year, times[0].month,
                                                       times[0].day)
    times_var.standard_name = "time"
    times_var.calendar = "standard"
    times_var[:] = nc.date2num(times.to_pydatetime(), times_var.units)
         
    # Create main element variables. Optimize chunkshape for single time series slices
    for vname, a_dtype in main_vars:
         
        a_var = ds_out.createVariable(vname, a_dtype, ('time', 'station_id'),
                                      chunksizes=(times.size,1),
                                      fill_value=nc.default_fillvals[a_dtype],
                                      zlib=True)
        
        a_var.missing_value = nc.default_fillvals[a_dtype]
        
    ds_out.sync()
        
    return ds_out
    