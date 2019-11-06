set size square
set tics front out scale 0.3; set cbtics in
unset key

# files
fa = 'L2-gnm.kij.reference'
fb = 'L2-gnm.t0-0.rc1.0.cm'
fc = 'L2-gnm.t0-0.rc1.0.cm.oe'
fd = 'L2-gnm.t0-0.rc1.0.cm.oeL20W1'
fe = 'L2-gnm.t0-0.rc1.0.cm.oeL20W1.km'
#
set term wxt 0 size 1200, 400
set xrange [-0.5:19.5]; set yrange [-0.5:19.5]
set xtics 0,5,20; set mxtics 5
set xtics 0,5,20; set mxtics 5
set multiplot layout 1,4
#
set title 'The true values of k_{ij}'
set palette defined(0 'white', 1'blue')
set cbrange [0:3]; set cbtics 0,1,3
plot fa matrix u 1:2:3 w image notitle
#
set title 'p_{ij} based on MD simulation (the input of HLM)'
set palette defined(0 'white', 1'red')
set cbrange [0:1]; set cbtics 0,1,1; set mcbtics 5;
plot fb matrix u 1:2:3 w image notitle
#
set title 'z_{ij}'
set palette defined(0 'blue', 0.5 'white', 1'red')
set cbrange [-1:1]; set cbtics -1,1,1; unset mcbtics; set cbtics add('2.7'1,'1'0,'0.4'-1);
plot fc matrix u 1:2:($3>0 && $1<$2 ? log($3) : NaN) w image notitle, \
     fd matrix u 1:2:($3>0 && $1>$2 ? log($3) : NaN) w image notitle
#
set title 'k_{ij} inferred from p_{ij} (the output of HLM)'
set palette defined(0 'white', 1'blue')
set cbrange [0:3]; set cbtics 0,1,3; set mcbtics 5
plot fe matrix u 1:2:3 w image notitle
#
unset multiplot
