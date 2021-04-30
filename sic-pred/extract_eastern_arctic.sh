#! /bin/bash -x

function process_one_month {
    # each output file is 42MB (after compression)
    # one year of files is .5GB
    outdir=$new_dir/${yr}
    # merge individual days into monthly files
    bname="ice_conc_nh_polstere-100_multi_${yr}${mon}"
    outfile=$outdir/${bname}.nc
    # don't overwrite if already there
    [[ -f $outfile ]] && return
    # subset spatially
    subset="-d xc,400,650 -d yc,450,870"

    # loop over daily files
    d2=$outdir/tmp
    mkdir -p $d2
    lst=()
    for f in $osisaf_dir/${yr}_nh_polstere/${bname}*.nc
    do
        [[ ! -f $f ]] && break
        g=$d2/$(basename $f)
        # decompress and subset spatially
        ncpdq -U -O $subset $f -o ${g}.0
        ncks -O --mk_rec_dmn time ${g}.0 -o $g
        ncatted -O -a _FillValue,,o,f,-32767 $g
        lst+=($g)
    done
    [[ ${#lst[@]} -eq 0 ]] && return

    # make the monthly files
    ncrcat ${lst[@]} -o $d2/${bname}.nc
    # compress
    ncpdq $d2/${bname}.nc -o $outfile
    #rm -r $d2
}


osisaf_dir=/cluster/projects/nn2993k/sim/data/OSISAF_ice_conc/polstere/
new_dir=$USERWORK/OSISAF_ice_conc_eastern_arctic_monthly
if [ 1 -eq 0 ]
then
    # test for one month
    yr=2018
    mon=01
    process_one_month
    exit 0
fi

for yr in 2018 2019 2020 2021
do
    for mon in 1 12
    do
        process_one_month
    done
done
