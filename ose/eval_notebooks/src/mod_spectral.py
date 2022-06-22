
import numpy as np
import scipy.signal
import logging

from src.mod_write import *


def compute_segment_alongtrack(time_alongtrack, 
                               lat_alongtrack, 
                               lon_alongtrack, 
                               ssh_alongtrack, 
                               ssh_map_interp, 
                               lenght_scale,
                               delta_x,
                               delta_t):

    segment_overlapping = 0.25
    max_delta_t_gap = 4 * np.timedelta64(1, 's')  # max delta t of 4 seconds to cut tracks

    list_lat_segment = []
    list_lon_segment = []
    list_ssh_alongtrack_segment = []
    list_ssh_map_interp_segment = []

    # Get number of point to consider for resolution = lenghtscale in km
    delta_t_jd = delta_t / (3600 * 24)
    npt = int(lenght_scale / delta_x)

    # cut track when diff time longer than 4*delta_t
    indi = np.where((np.diff(time_alongtrack) > max_delta_t_gap))[0]
    track_segment_lenght = np.insert(np.diff(indi), [0], indi[0])

    # Long track >= npt
    selected_track_segment = np.where(track_segment_lenght >= npt)[0]

    if selected_track_segment.size > 0:

        for track in selected_track_segment:

            if track-1 >= 0:
                index_start_selected_track = indi[track-1]
                index_end_selected_track = indi[track]
            else:
                index_start_selected_track = 0
                index_end_selected_track = indi[track]

            start_point = index_start_selected_track
            end_point = index_end_selected_track

            for sub_segment_point in range(start_point, end_point - npt, int(npt*segment_overlapping)):

                # Near Greenwhich case
                if ((lon_alongtrack[sub_segment_point + npt - 1] < 50.)
                    and (lon_alongtrack[sub_segment_point] > 320.)) \
                        or ((lon_alongtrack[sub_segment_point + npt - 1] > 320.)
                            and (lon_alongtrack[sub_segment_point] < 50.)):

                    tmp_lon = np.where(lon_alongtrack[sub_segment_point:sub_segment_point + npt] > 180,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt] - 360,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt])
                    mean_lon_sub_segment = np.median(tmp_lon)

                    if mean_lon_sub_segment < 0:
                        mean_lon_sub_segment = mean_lon_sub_segment + 360.
                else:

                    mean_lon_sub_segment = np.median(lon_alongtrack[sub_segment_point:sub_segment_point + npt])

                mean_lat_sub_segment = np.median(lat_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_alongtrack_segment = np.ma.masked_invalid(ssh_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_map_interp_segment = []
                ssh_map_interp_segment = np.ma.masked_invalid(ssh_map_interp[sub_segment_point:sub_segment_point + npt])
                if np.ma.is_masked(ssh_map_interp_segment):
                    ssh_alongtrack_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(ssh_map_interp_segment), ssh_alongtrack_segment))
                    ssh_map_interp_segment = np.ma.compressed(ssh_map_interp_segment)

                if ssh_alongtrack_segment.size > 0:
                    list_ssh_alongtrack_segment.append(ssh_alongtrack_segment)
                    list_lon_segment.append(mean_lon_sub_segment)
                    list_lat_segment.append(mean_lat_sub_segment)
                    list_ssh_map_interp_segment.append(ssh_map_interp_segment)


    return list_lon_segment, list_lat_segment, list_ssh_alongtrack_segment, list_ssh_map_interp_segment, npt 




def compute_spectral_scores(time_alongtrack, 
                            lat_alongtrack, 
                            lon_alongtrack, 
                            ssh_alongtrack, 
                            ssh_map_interp, 
                            lenght_scale,
                            delta_x,
                            delta_t,
                            output_filename):
    
    # make time vector as days since 1950-01-01
    #time_alongtrack = (time_alongtrack - np.datetime64('1950-01-01T00:00:00Z')) / np.timedelta64(1, 'D')
    
    # compute segments
    lon_segment, lat_segment, ref_segment, study_segment, npt  = compute_segment_alongtrack(time_alongtrack, 
                                                                                            lat_alongtrack, 
                                                                                            lon_alongtrack, 
                                                                                            ssh_alongtrack, 
                                                                                            ssh_map_interp, 
                                                                                            lenght_scale,
                                                                                            delta_x,
                                                                                            delta_t)
    
    # Power spectrum density reference field
    global_wavenumber, global_psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                           fs=1.0 / delta_x,
                                                           nperseg=npt,
                                                           scaling='density',
                                                           noverlap=0)

    # Power spectrum density study field
    _, global_psd_study = scipy.signal.welch(np.asarray(study_segment).flatten(),
                                             fs=1.0 / delta_x,
                                             nperseg=npt,
                                             scaling='density',
                                             noverlap=0)

    # Power spectrum density study field
    _, global_psd_diff = scipy.signal.welch(np.asarray(study_segment).flatten()-np.asarray(ref_segment).flatten(),
                                            fs=1.0 / delta_x,
                                            nperseg=npt,
                                            scaling='density',
                                            noverlap=0)
    
    # Save psd in netcdf file
    ds = xr.Dataset({"psd_ref": (["wavenumber"], global_psd_ref),
                     "psd_study": (["wavenumber"], global_psd_study),
                     "psd_diff": (["wavenumber"], global_psd_diff),
                    },
                    coords={"wavenumber": (["wavenumber"], global_wavenumber)},
                   )
    
    ds.to_netcdf(output_filename)
    logging.info(f'  Results saved in: {output_filename}')

    