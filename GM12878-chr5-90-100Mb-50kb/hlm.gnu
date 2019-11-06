set size square
set tics front out scale 0.3; set cbtics in
unset key

# files
fa = 'chr5_50kb.090-100Mb.Hi-C.cm'
fb = 'chr5_50kb.090-100Mb.oe'
fc = 'chr5_50kb.090-100Mb.oeL400W2'
fd = 'chr5_50kb.090-100Mb.oeL400W2.km.list'
fe = 'MD/chr5_50kb.090-100Mb.HLM-MD.cm'
mps= 0.3
#
set term wxt 0 size 1200, 400
set xrange [0:200]; set yrange [0:200]
set xtics 0,40,200; set mxtics 2
set ytics 0,40,200; set mytics 2
set xtics add('90'0, '92'40, '94'80, '96'120, '98'160, '100'200)
set ytics add('90'0, '92'40, '94'80, '96'120, '98'160, '100'200)
set multiplot layout 1,4
#
set title 'p_{ij} measured by Hi-C'
set palette defined(0 'white', 1'red')
set cbrange [-2.5:0]; set cbtics -3,1,0; set cbtics add('10^-3'-3, '10^-2'-2, '0.1'-1, '1'0)
plot fa matrix u 1:2:($3>0 ? log10($3) : NaN) w image notitle
#
set title 'z_{ij}'
set palette defined(0 'blue', 0.5 'white', 1'red')
set cbrange [-1:1]; set cbtics -1,1,1; unset mcbtics; set cbtics add('2.7'1,'1'0,'0.4'-1);
plot fb matrix u 1:2:($3>0 && $1<$2 ? log($3) : NaN) w image notitle, \
     fc matrix u 1:2:($3>0 && $1>$2 ? log($3) : NaN) w image notitle
#
set title 'k_{ij} inferred from p_{ij} (the output of HLM)'
set palette defined(0 'red', 0.5 'white', 1'blue')
set cbrange [-3:0]; set cbtics -3,1,0; set cbtics add('10^-3'-3, '10^-2'-2, '0.1'-1, '1'0)
#plot fd matrix u 1:2:($3>0 ? log10($3) : NaN) w image notitle
plot fd u 1:2:(log10($3)) w p lc palette ps mps pt 7 notitle
#
set title 'p_{ij} based on 3D structures'
set palette defined(0 'white', 1'red')
set cbrange [-2.5:0]; set cbtics -3,1,0; set cbtics add('10^-3'-3, '10^-2'-2, '0.1'-1, '1'0)
plot fe matrix u 1:2:($3>0 ? log10($3) : NaN) w image notitle
#
unset multiplot
