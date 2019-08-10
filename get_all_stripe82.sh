#!/bin/bash
filename='stripe82/sdss-wget-tohUxz-1.lis'
echo Press CTRL+Z to interrupt the script.
while read p; do 
    # strip base url and extension
    bn=${p##*/}
    bn=${bn%%.*}
    echo "Processing file $bn"
    python3 mto.py $p -par_out stripe82/parameters/$bn.csv -gz_filename stripe82/zoo2Stripe82Coadd1.csv -move_factor 1.5
done < $filename