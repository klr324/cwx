'''
Utilities for making time-of-observation adjustments to daily precipitation
observations.
'''

from cwx.util import nn_xr_data_array
from obsio import ObsIO, NcObsIO
#klr 2/8/2023: scipy.stats.stats is now scipy.stats
#from scipy.stats.stats import pearsonr
from scipy.stats import pearsonr
from tzwhere.tzwhere import tzwhere
import numpy as np
import pandas as pd

class TobsIdentifier():
    """Class for estimating a station's time of observation
    """

    _a_tzw = None

    def __init__(self, hrly_prcp_da):
        '''

        Parameters
        ----------
        hrly_prcp_da : list of xarray.DataArray
            List of xarray.DataArrays pointing to gridded supplemental data
            sources of hourly precipitation (e.g. NLDAS2, CPC Hourly). For
            overlapping hours between the data sources, priority is given by
            list order (e.g.--first DataArray in list is given highest priority).
            DataArray time dimension values should be in UTC.
        '''
        self.hrly_prcp_das = hrly_prcp_da

    @property
    def _tzw(self):

        if TobsIdentifier._a_tzw is None:

            print ("Initializing tzwhere for time zone retrieval...",)
            TobsIdentifier._a_tzw = tzwhere(forceTZ=True)
            print ('done.')

        return TobsIdentifier._a_tzw

    def _get_prcp_hrly(self, lat, lon, start_date, end_date):


        # Get time zone of lat/lon
        tz = self._tzw.tzNameAt(lat, lon, forceTZ=True)

        if tz is None:

            raise ValueError("Could not determine time zone for location: "
                             "%.4f, %.4f"%(lat, lon))


        utc_hrs_por = pd.date_range(start_date - pd.Timedelta(days=1),
                                    end_date + pd.Timedelta(days=2),
                                    freq='H', tz='UTC')

        prcp_hrly = pd.Series(np.nan, index=utc_hrs_por)

        for da in self.hrly_prcp_das:

            a_prcp_hrly = nn_xr_data_array(da, lon, lat).to_series()
            #klr 2/24/2023: Problem fixed. Problem with dataset and not code.
            #klr 2/8/2023: Comment out tz_localize ("Already tz-aware, use tz_convert to convert.")
            #a_prcp_hrly.index = a_prcp_hrly.index.tz_localize('UTC')
            a_prcp_hrly.index = a_prcp_hrly.index.tz_localize('UTC')
            a_prcp_hrly = a_prcp_hrly.reindex(utc_hrs_por)

            prcp_hrly = prcp_hrly.fillna(a_prcp_hrly)

        # Convert to local time zone and subset to dates
        prcp_hrly.index = prcp_hrly.index.tz_convert(tz).tz_localize(None)
        # Subset to period-of-record dates
        prcp_hrly = prcp_hrly.loc[start_date:(end_date + pd.Timedelta(days=1) -
                                              pd.Timedelta(seconds=1))]

        return prcp_hrly

    def identify_tobs(self, lat, lon, prcp):
        '''Identify time of observation using auxiliary hourly data.

        Parameters
        ----------
        lat : float
            Station latitude. Used in combination with lon to get station
            time zone
        lon : float
            Station longitude. Used in combination with lat to get station
            time zone
        prcp : pandas.Series
            Daily precipitation observations for a station as a pandas Series with
            a time index.

        Returns
        ----------
        prcp_adj : pandas.Series
            Series with precipitation values adjusted for time-of-observation.
        '''

        # Get hours for period-of-record
        start_date = prcp.index.min()
        end_date = prcp.index.max()

        prcp_hrly = self._get_prcp_hrly(lat, lon, start_date, end_date)

        tobs_ls = np.arange(100,2500,100)
        #klr 2/8/2023: DeprecationWarning: `np.str` is a deprecated alias for the builtin `str`.
        #tobs_str =  tobs_ls.astype(np.str)
        tobs_str =  tobs_ls.astype(str)

        df_dly = pd.DataFrame({str(a_tobs):prcp_hrly_to_dly(prcp_hrly, a_tobs)
                               for a_tobs in tobs_ls}).loc[:,tobs_str].copy()
        df_dly['stn_prcp'] = prcp
        df_dly = df_dly.dropna()

        cor_dly = pd.Series([pearsonr(df_dly.loc[:,a_tobs],df_dly.stn_prcp)[0]
                             for a_tobs in tobs_str], index=tobs_str)
        #klr 2/8/23: `np.int` is a deprecated alias for the builtin `int`.
        #tobs = np.int(cor_dly.idxmax())
        tobs = int(cor_dly.idxmax())

        return tobs

def tobs_mode(df_tobs, stns, tobs_est=None):

    # Replace any vals > 2400 or < 0 with na
    df_tobs.values[df_tobs.values > 2400] = np.nan
    df_tobs.values[df_tobs.values < 0] = np.nan

    # Replace 0 with 2400 to represent midnight-to-midnight
    df_tobs = df_tobs.replace(0, 2400)

    # Set CoCoRaHS stations to a 700 time-of-observation
    ids_cocorahs = stns.index[(stns.sub_provider == 'CoCoRaHS').values]
    df_tobs.loc[:, ids_cocorahs.values] = 700

    # Get the most frequent time-of-observation for each station
    tobs_mode = df_tobs.mode().iloc[0, :]

    if tobs_est is not None:

        tobs_mode = tobs_mode.fillna(tobs_est)

    return tobs_mode

def prcp_hrly_to_dly(prcp_hrly, target_tobs):
    '''
    Aggregate hourly prcp data to daily based on a time-of-observation.

    Parameters
    ----------
    prcp_hrly : pandas.Series
        Hourly precipitation observations
    target_tobs: int, optional
        The target time-of-observation for the daily aggregation.

    Returns
    ----------
    prcp_dly : pandas.Series
    '''

    prcp_hrly_adj = prcp_hrly.copy()

    # Adjust times for time-of-observation
    # Shift in hours from midnight-to-midnight
    hr_shift = 2400 - np.float(target_tobs)
    # Convert to decimal hrs
    hr_shift = hr_shift / 100
    hr_shift = (np.int(hr_shift) + (((hr_shift - np.int(hr_shift)) * 100) / 60.0))
    # Convert to ns timedelta
    hr_shift = np.array([hr_shift*3.6e+12]).astype('timedelta64[ns]')[0]
    # Shift index times based on hr shift
    new_time = prcp_hrly.index + hr_shift
    prcp_hrly_adj.index = new_time

    prcp_dly = prcp_hrly_adj.resample('D').sum()
    prcp_dly = prcp_dly.loc[prcp_hrly.index.min().date():prcp_hrly.index.max().date()]

    return prcp_dly

def process_tobs_default(df_tobs, stns):
    
    df_tobs = df_tobs.copy()
    
    # Replace any vals > 2400 or < 0 with na
    df_tobs[df_tobs > 2400] = np.nan
    df_tobs[df_tobs < 0] = np.nan
    
    # Replace 0 with 2400 to represent midnight-to-midnight
    df_tobs = df_tobs.replace(0, 2400)
    
    # Set CoCoRaHS stations to a 700 time-of-observation
    ids_cocorahs = stns.station_id[(stns.sub_provider == 'CoCoRaHS')]
    df_tobs.loc[:, ids_cocorahs.values] = 700
    
    # Propagate nearest value to any missing time-of-observation
    df_tobs = df_tobs.interpolate(axis=0, method='nearest')

    # Fill missing values at the start (end) with the first (last) valid
    # time-obs-observation
    df_tobs = df_tobs.fillna(method='bfill', axis=0)
    df_tobs = df_tobs.fillna(method='ffill', axis=0)
    
    # For stations with no time-of-observation data, assume value of 2400
    df_tobs.loc[:, df_tobs.columns[df_tobs.isnull().all(axis=0).values]] = 2400
    
    # Make sure no tobs are null
    if np.isnan(df_tobs.values).any():
        
        raise ValueError("Null time-of-observation values not permitted.")
    
    return df_tobs

class TobsAdj():
    """Class for performing time-of-observation adjustments on daily prcp
    """

    def __init__(self, func_adj):
        '''

        Parameters
        ----------
        func_adj : function
            The function to use for performing the time-of-observation adjustments
        '''

        self.func_adj = func_adj

    def adjust_for_tobs(self, lat, lon, prcp, tobs, target_tobs=2400):
        '''Adjust daily precipitation values for time-of-observation.

        Parameters
        ----------
        lat : float
            Station latitude
        lon : float
            Station longitude
        prcp : pandas.Series
            Daily precipitation observations for a station as a pandas Series with
            a time index.
        tobs : pandas.Series
            Daily time-of-observation values for a station as pandas Series with
            a time index. Values should be in military time format (e.g. 7:30am
            observation time is 730, 11:00pm observation time is 2300). Midnight
            observation times are required be set to 2400. Any null values will
            be assumed to be a midnight-to-midnight observation time.
        target_obs: int, optional
            The target time-of-observation. Default: 2400 (midnight-to-midnight).

        Returns
        ----------
        prcp_adj : pandas.Series
            Series with precipitation values adjusted for time-of-observation.
        '''

        prcp_adj = self.func_adj(prcp, tobs, target_tobs)

        return prcp_adj

def adjust_for_tobs_shift(prcp, tobs, target_tobs=2400):
    '''
    Adjust daily precipitation values for time-of-observation.

    Applies a single one day backward or forward shift to daily prcp observations
    based on the original and target time observation. If the original observation
    time is in the morning (tobs >= 0 and tobs < 1200) and the target is pm or
    midnight (target_tobs >= 1200), the prcp total is shifted back a day.
    The logic is that most of the precipitation total for a morning observation
    time occurred in the previous day. If the original observation time is pm or
    midnight and the target is am, the prcp total is shift forward a day

    Parameters
    ----------
    prcp : pandas.Series
        Daily precipitation observations for a station as a pandas Series with
        a time index
    tobs : pandas.Series
        Daily time-of-observation values for a station as pandas Series with
        a time index. Values should be in military time format (e.g. 7:30am
        observation time is 730, 11:00pm observation time is 2300). Midnight
        observation times are required be set to 2400.
    target_tobs: int, optional
        The target time-of-observation. Default: 2400 (midnight-to-midnight).

    Returns
    ----------
    prcp_adj : pandas.Series
        Series with precipitation values adjusted for time-of-observation.
    '''

    prcp_adj = prcp.copy()

    mask_am = ((tobs >= 0) & (tobs < 1200)).values
    mask_pm_midnight = tobs >= 1200
    #mask_pm = ((tobs >= 1200) & (tobs < 2100)).values
    #mask_midnight = (tobs >= 2100).values

    is_target_am = target_tobs >= 0 and target_tobs < 1200
    is_target_pm_midnight = target_tobs >= 1200

    mask_shift_back = np.logical_and(mask_am, is_target_pm_midnight)
    mask_shift_forward = np.logical_and(mask_pm_midnight, is_target_am)

    if mask_shift_back.any() or mask_shift_forward.any():

        df_prcp = pd.DataFrame({'prcp':prcp, 'prcp_shift_back':prcp.shift(-1),
                                'prcp_shift_forward':prcp.shift(1)})

        dates_shifted_back = prcp.index[mask_shift_back]
        dates_shifted_forward = prcp.index[mask_shift_forward]
        dates_shifted_all = dates_shifted_back.append(dates_shifted_forward)

        dates_shifted_back1 = prcp.index[mask_shift_back] - pd.Timedelta(days=1)
        dates_shifted_forward1 = prcp.index[mask_shift_forward] + pd.Timedelta(days=1)

        dates_nonshifted = prcp.index[~np.logical_or(mask_shift_back, mask_shift_forward)]

        # Final set of dates that should receive a prcp value from the next day
        fnl_dates_shifted_back = dates_shifted_back1[(~dates_shifted_back1.isin(dates_nonshifted)) &
                                                     (dates_shifted_back1.isin(prcp.index))]
        # Final set of dates that should receive a prcp value from the previous day
        fnl_dates_shifted_forward = dates_shifted_forward1[(~dates_shifted_forward1.isin(dates_nonshifted)) &
                                                           (dates_shifted_forward1.isin(prcp.index))]
        fnl_dates_shifted_all = fnl_dates_shifted_back.append(fnl_dates_shifted_forward)

        # Dates that were shifted but did not receive a replacement prcp value.
        # Set these dates to NA
        fnl_dates_na = dates_shifted_all[(~dates_shifted_all.isin(fnl_dates_shifted_all)) &
                                         (~dates_shifted_all.isin(dates_nonshifted))]

        # Make adjustments
        prcp_adj.loc[fnl_dates_shifted_back] = df_prcp.prcp_shift_back.loc[fnl_dates_shifted_back]
        prcp_adj.loc[dates_shifted_forward] = df_prcp.prcp_shift_forward.loc[dates_shifted_forward]
        prcp_adj.loc[fnl_dates_na] = np.nan

    return prcp_adj


class PrcpAdjIO(ObsIO):
    """Subclass of ObsIO to calculate and output time-of-observation adjusted prcp
    """

    _avail_elems = ['prcp']
    _requires_local = True
    name = 'PRCP_TOBS_ADJ'

    def __init__(self, fpath_nc, tobs_adj, start_date, end_date, target_tobs=2400, process_tobs_func=process_tobs_default):
        '''

        Parameters
        ----------
        fpath_nc : str
            File path to main station observation netCDF file
        tobs_adj : twxp.tobs.TobsAdj
            An instance of TobsAdj for adjusting daily precipitation observations
        start_date : pandas.Timestamp
            Start date of desired date range.
        end_date : pandas.Timestamp
            End date of desired date range.
        target_tobs: int, optional
            The target time-of-observation. Default: 2400 (midnight-to-midnight).
        '''

        super(PrcpAdjIO, self).__init__(elems=PrcpAdjIO._avail_elems,
                                        start_date=start_date,
                                        end_date=end_date)

        self.ncio = NcObsIO(fpath_nc, ['prcp', 'tobs_prcp'])
        self.tobs_adj = tobs_adj
        self.target_tobs = target_tobs
        self.process_tobs_func = process_tobs_func


    def _read_stns(self):

        stns = self.ncio.stns.drop(['station_index'], axis=1)
        return stns

    def _read_obs(self, stns_ids=None):

        if stns_ids is None:
            stns_obs = self.stns
        else:
            stns_obs = self.stns.loc[stns_ids]

        obs = self.ncio.read_obs(stns_ids, data_structure='array')
        prcp = obs.prcp.to_pandas().loc[self.start_date:self.end_date]
        tobs = obs.tobs_prcp.to_pandas().loc[self.start_date:self.end_date]
        tobs = self.process_tobs_func(tobs, stns_obs)

        prcp_adj = pd.DataFrame(np.nan, index=prcp.index, columns=prcp.columns)

        for a_id in stns_ids:

            stn_prcp = prcp.loc[:, a_id]
            stn_tobs = tobs.loc[:, a_id]
            stn_lat = stns_obs.loc[a_id].latitude
            stn_lon = stns_obs.loc[a_id].longitude

            if (stn_tobs != self.target_tobs).any():

                print (a_id + ": adjusting precipitation values...")

                try:
                    stn_prcp_adj = self.tobs_adj.adjust_for_tobs(stn_lat, stn_lon,
                                                                 stn_prcp, stn_tobs,
                                                                 target_tobs=self.target_tobs)
                    prcp_adj.loc[:, a_id] = stn_prcp_adj

                except ValueError as e:

                    if e.args[0].startswith("Could not determine time zone for location"):

                        print ("Warning: Could not determine timezone for station %s."
                               " Precipitation values will not be adjusted."%a_id)
                        prcp_adj.loc[:, a_id] = stn_prcp

                    else:

                        raise

            else:

                print (a_id + ": tobs same as target, not adjusting precipitation values...")
                # Station time-of-observations is the same as the target on
                # every day. No adjustments necessary
                prcp_adj.loc[:, a_id] = stn_prcp

        prcp_adj = pd.DataFrame(prcp_adj.stack(dropna=False))
        prcp_adj['elem'] = 'prcp'
        prcp_adj = prcp_adj.rename(columns={0:'obs_value'})
        prcp_adj = prcp_adj.set_index('elem', append=True)

        obs = prcp_adj
        # klr 2/28/2023: sortlevel depreciated. Use sort_index
        #obs = obs.reorder_levels(['station_id', 'elem',
        #                          'time']).sortlevel(0, sort_remaining=True)
        obs = obs.reorder_levels(['station_id', 'elem', 'time']).sort_index(0, sort_remaining=True)

        return obs
