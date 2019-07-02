from astropy.coordinates import SkyCoord
from astropy import units as u
import csv
import argparse
import numpy as np
from mtolib.io_mto import get_fits_header, get_sdss_fits_header_coordinates

def make_parser():
    parser = argparse.ArgumentParser(description='Given a .fits file and a GalaxyZoo .csv file, find all the objects in range')
    parser.add_argument('fits_filename', type=str, help='Location of input .fits file')
    parser.add_argument('csv_filename', type=str, help='Location of input .csv file')

    return parser



p = make_parser().parse_args()

# get fits coordinates
header = get_fits_header(p.fits_filename)
fits_coords = get_sdss_fits_header_coordinates(header)
fits_skycoord = SkyCoord(fits_coords.ref_coords[0], fits_coords.ref_coords[1], unit=u.degree)

bottom_right = 0.5 * np.array([fits_coords.dimensions[0], fits_coords.dimensions[1]])
corner_coords = fits_coords.ref_coords + np.dot(fits_coords.ra_dec_per_pixel_matrix, bottom_right)
corner_skycoord = SkyCoord(corner_coords[0], corner_coords[1], unit=u.degree)

corner_to_ref_sep = corner_skycoord.separation(fits_skycoord)

#max_lines = 25000

with open(p.csv_filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    # read the first line
    next(csv_reader)

    min_sep_deg = 360 * u.deg
    max_sep_deg = 0 * u.deg

    for row in csv_reader:
        ra_sx, dec_sx = row[1], row[2]
        object_skycoord = SkyCoord(ra_sx, dec_sx, unit=u.degree)
        sep = object_skycoord.separation(fits_skycoord)
        min_sep_deg = min(min_sep_deg, sep)
        max_sep_deg = max(max_sep_deg, sep)
        #print(f'object {row[0]}, seperation = {object_skycoord.separation(fits_skycoord)}')
        line_count += 1
        if line_count % 1000 == 0:
            print(f'processed {line_count} lines. min = {min_sep_deg}. max = {max_sep_deg}')
        if not object_skycoord.is_equivalent_frame(fits_skycoord):
            print(f'{line_count}: frames are not equivalent')
        # if line_count > max_lines:
        #     break
        # delta = abs(object_skycoord.ra.degree - fits_skycoord.ra.degree)
        # if delta > 50.0:
        #     #print(f'{delta}')
        #     for i in range(0, 50):
        #         next(csv_reader)
        #         line_count += 1

print(f'read {line_count} lines. min_sep={min_sep_deg}. max_sep={max_sep_deg}')

