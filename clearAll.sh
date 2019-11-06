#!/bin/bash

# toy
cd toyModel
rm L2-gnm.t0-0.rc1.0.cm.oe
rm L2-gnm.t0-0.rc1.0.cm.oeL20W1*
cd ../

# GM12878
cd GM12878-chr5-90-100Mb-50kb
rm chr5_50kb.090-100Mb.ps
rm chr5_50kb.090-100Mb.oe*
#
cd MD
if [ -f chr5_50kb.090-100Mb.oeL400W2.km ]
then
    rm ./*km
fi
if [ -f gnm.t0.sys ]
then
    rm ./gnm.t0*
fi
cd ../
#
cd ../
