#!/bin/bash
filename='stripe82/first-100-fits.lis'
echo Start
while read p; do 
    # strip base url and extension
    bn=${p##*/}
    bn=${bn%%.*}
    echo "Processing file $bn"
    python3 mto.py $p -par_out stripe82/parameters/$bn.csv -gz_filename stripe82/zoo2Stripe82Coadd1.csv
done < $filename