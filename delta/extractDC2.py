import numpy as np
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import os
import pandas as pd
from tqdm import tqdm
from reproject import reproject_interp
from skimage.feature import peak_local_max
import pickle
import glob
import json

def get_radec_bounds(wcs):
    max1=wcs.pixel_to_world(wcs._naxis[0], wcs._naxis[1])
    min1=wcs.pixel_to_world(0, 0)

    if max1.ra.deg > min1.ra.deg:
        ra_max, ra_min = max1.ra.deg, min1.ra.deg
    else:
        ra_max, ra_min = min1.ra.deg, max1.ra.deg
    if max1.dec.deg > min1.dec.deg:
        dec_max, dec_min = max1.dec.deg, min1.dec.deg
    else:
        dec_max, dec_min = min1.dec.deg, max1.dec.deg
    return ra_min, ra_max, dec_min, dec_max

def _get_rubin_wcs(key_name, wcs_json):
    return wcs.WCS(wcs_json[key_name])

def _get_roman_wcs(fname, wcs_json):
    return wcs.WCS(wcs_json[fname]['wcs'])

def get_roman_coadd(fname, wcs_json):
    # open the file
    coadd_roman = np.load(fname)
    fs = fname.split('/')
    wcs_entry = 'roman_data/truth/'+fs[-2]+'/'+fs[-1]
    try:
        wcs_roman = _get_roman_wcs(wcs_entry, wcs_json)
    except:
        return 1, 1
    return coadd_roman, wcs_roman

def get_rubin_coadd(fname, wcs_json):
    coadd_rubin = np.load(fname)
    fs = fname.split('_')[-3]
    try:
        wcs_rubin = _get_rubin_wcs(fs, wcs_json)
    except:
        return 1, 1
    return coadd_rubin, wcs_rubin

def reproject_rubin_to_roman(rubin_ims, wcs_rubin, wcs_roman, coadd_roman):
    return reproject_interp((rubin_ims,wcs_rubin),wcs_roman,shape_out=coadd_roman['data'].shape)

def make_cutout(img, wcs, pos_xy=None, pos_radec=None, cutout_size=64):
    if img.ndim == 3:
        multiband = True
        #     image = img[band_idx] # use specified band for finding peaks to center cutouts on
        # else:
        #     raise ValueError("For 3D image cubes, please specify the band index to use for finding local peaks to center cutouts on.")
    elif img.ndim == 2:
        multiband = False
        image = img
    else:
        raise ValueError("Input image must be either a 2D array or a 3D cube with shape (bands, height, width)")
    
    if pos_xy is not None:
        sc1=wcs.pixel_to_world(pos_xy[0], pos_xy[1])
    elif pos_radec is not None:
        sc1=SkyCoord(ra=pos_radec[0], dec=pos_radec[1], unit='deg')
    else:
        raise ValueError("Please specify either pixel coordinates (pos_xy) or world coordinates (pos_radec) for the cutout center.")
    try:
        cutout = Cutout2D(image, sc1, (cutout_size, cutout_size), wcs=wcs)
        cutout_slices = cutout.slices_original
        cutout_wcs = cutout.wcs
    except:
        print(f"Could not make cutout for source at ra={sc1.ra.deg}, dec={sc1.dec.deg}. Skipping this source.")
        return None, None
    if multiband:
        cutout_data = img[:, cutout_slices[0], cutout_slices[1]]
    else:
        cutout_data = img[cutout_slices[0], cutout_slices[1]]

    return cutout_data, cutout_wcs



if __name__ == "__main__":
    # annots = {'path':[], 'img':[]}
    with open('dir_list.pkl', 'rb') as f:
        dir_list = pickle.load(f)
        f.close()
    with open('/projects/bfhm/yse2/annotations_roman/all_wcs.json','r') as f:
        roman_wcs_json = json.load(f)
        f.close()

    for dir in dir_list:
        rubin_glob = glob.glob(f'/work/hdd/bdsp/yse2/lsst_data/truth/{dir}/*.npy')
        with open(glob.glob(f'/work/hdd/bdsp/yse2/lsst_data/truth/{dir}/*wcs*.json')[0], 'r') as f:
            rubin_wcs_json = json.load(f)
            f.close()
        # roman_glob = []
        for rubin_fname in rubin_glob:
            # roman_glob.append(os.path.join(f'/work/hdd/bdsp/yse2/truth-roman/{dir}', rubin_fname.split('/')[-1]))
            roman_fname = os.path.join(f'/work/hdd/bdsp/yse2/truth-roman/{dir}', rubin_fname.split('/')[-1])
            if os.path.exists(roman_fname):
                pass
            else:
                print(f"Roman file {roman_fname} does not exist. Please check the path and try again.")
                continue
            coadd_rubin, wcs_rubin = get_rubin_coadd(rubin_fname, rubin_wcs_json)
            coadd_roman, wcs_roman = get_roman_coadd(roman_fname, roman_wcs_json)
            if coadd_rubin == 1 or coadd_roman == 1:
                print(f"Could not get WCS for Rubin or Roman coadd for file {rubin_fname}. Skipping this file.")
                continue
            # TODO: add in cutout making and saving here

    # fpath = ''

