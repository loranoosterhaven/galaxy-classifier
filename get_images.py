from astropy.coordinates import SkyCoord
from astropy import units as u
import csv
import pandas
from SciServer import Authentication, SkyServer
from mtolib import io_mto
from PIL import Image
from astropy.io import fits
import numpy as np

def process_val(val):
    if val > 0.0:
        return np.exp(val / 512.0)
    else:
        return val


num_lines = 25

LOGIN_NAME = 'loranoosterhaven'
LOGIN_PASSWORD = 'dXn3v5CjhvzEARfl'
IMG_WIDTH = 512
IMG_HEIGHT = 512

with open("GalaxyZoo1_DR_table7.csv") as csv_file:

    # login
    token1 = Authentication.login(LOGIN_NAME, LOGIN_PASSWORD)

    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for x in range(0, num_lines):
        row = next(csv_reader)
        if line_count == 0:
            line_count += 1
        else:
            ra_sx, dec_sx = row[1], row[2]
            c = SkyCoord(ra_sx, dec_sx, unit=u.degree)
            print(f'object {row[0]}', c)
            line_count += 1

            img = SkyServer.getJpegImgCutout(
                ra=c.ra.deg, dec=c.dec.deg, 
                width=IMG_WIDTH, height=IMG_HEIGHT, scale=0.1, 
                dataRelease='DR13')


            Image.fromarray(img).save(f"skyserver_fits/{row[0]}.png")

            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

            red = fits.PrimaryHDU(data=r)
            red.writeto(f"skyserver_fits/{row[0]}r.fits")

            green = fits.PrimaryHDU(data=g)
            green.writeto(f"skyserver_fits/{row[0]}g.fits")

            blue = fits.PrimaryHDU(data=b)
            blue.writeto(f"skyserver_fits/{row[0]}b.fits")

            # Conversion to FITS file - unfortunately, it doesn't work correctly
            #print(img.shape[0])
            #print(img.shape[1])
            #print(img.shape[2])
            #img = np.float32(img)
            #img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
            #io_mto.write_fits_file(data=[np.vectorize(process_val)(img)], filename=f"skyserver_fits/{row[0]}.fits")


    print(f"Read {line_count} lines")