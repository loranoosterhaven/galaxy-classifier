from astropy.coordinates import SkyCoord
from astropy import units as u
import csv
import argparse
import numpy as np
from mtolib.io_mto import get_fits_header, get_sdss_fits_header_coordinates

def make_parser():
    parser = argparse.ArgumentParser(description='Given a .fits file and a GalaxyZoo .csv file, find all the objects in range')
    parser.add_argument('csv_filename', type=str, help='Location of csv file containing object coordinates to be match')
    parser.add_argument('gz_filename', type=str, help='Location of input GalaxyZoo .csv file')
    return parser

p = make_parser().parse_args()

# read in the entire GalaxyZoo table into an array
gz_coords = np.genfromtxt(p.gz_filename, delimiter=',', skip_header=1, usecols=(4,5,9),
    dtype=[('ra','float'),('dec','float'), ('class','S7')])

# create catalogue
catalogue = SkyCoord(ra=gz_coords['ra']*u.deg, dec=gz_coords['dec']*u.deg)

# read in the coordinates into its own array
object_coords = np.genfromtxt(p.csv_filename, delimiter=',', skip_header=2, usecols=(0,4,3),
    dtype=[('id', 'int'), ('ra','float'),('dec','float')])

# create object catalogue
obj_catalogue = SkyCoord(ra=object_coords['ra']*u.deg, dec=object_coords['dec']*u.deg)

idx, d2d, d3d = obj_catalogue.match_to_catalog_sky(catalogue)

mindistidx = np.argmin(d2d)

for i in range(0, len(object_coords)):
    oc = object_coords[i]['id']
    print(f'{oc} {idx[i]} {d2d[i]}')

print(f'{mindistidx}, {d2d[mindistidx]}')