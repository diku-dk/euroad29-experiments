# Bar charts of Jacobian overhead relative to the primal, one per program.
#
#   ./extract-results.py results-c.json results-multicore.json results-hip.json
#   gnuplot plot-results.gp
#
# Each group of bars is one way of computing the Jacobian; within a group the
# three bars are the c, multicore and hip backends -- labelled CPU (seq),
# CPU (mt) and GPU -- each relative to its own primal.  The baselines thus
# differ between bars, which is deliberate: the comparison of interest is how
# the cost of the Jacobian scales with the backend, not how fast the backends
# are.
#
# Optional -e settings:
#   fmt='png'   raster output instead of PDF
#   scale=N     resolution multiplier for PNG output (default 3)
#   labelsize=N point size of the value printed above each bar (default 11)
#   logscale=1  logarithmic y axis; useful for ht, whose variants span 40x

if (!exists("fmt"))      fmt = "pdf"
if (!exists("logscale")) logscale = 0
if (!exists("scale"))    scale = 3
if (!exists("labelsize")) labelsize = 11   # font size of the value above each bar

programs = "ba ht greeks reaction-network batch-reactor"

# The canvas is wide relative to its height so that the value printed above
# each bar has room: adjacent bars in a cluster are 1/(nbackends+cluster_gap)
# apart, and at four characters the labels of two tall neighbours touch on a
# narrower plot.
#
# PNG is raster, so its resolution is the canvas size.  Scaling the canvas,
# the fonts and the line widths together keeps the layout identical and just
# renders it more finely; everything else is in character units and follows
# the fonts.  PDF is vector, so 'scale' does not apply to it.
if (fmt eq "png") {
  eval sprintf('set terminal pngcairo size %d,%d fontscale %g linewidth %g font "sans,11"', \
               900 * scale, 480 * scale, scale, scale)
} else {
  set terminal pdfcairo size 16cm,9cm font "sans,11"
}

# Three bars per group (one per backend), separated only between groups.
nbackends = 3
cluster_gap = 2
set style data histograms
# 'set style histogram' wants a literal, so build the command to keep
# cluster_gap the single source of truth for both the style and bar() below.
eval sprintf("set style histogram clustered gap %d", cluster_gap)
set style fill solid 0.85 border lt -1
set boxwidth 0.9 relative

# Where gnuplot puts the bars: a cluster is centred on its row number, each bar
# is 1/(nbackends+cluster_gap) wide, and bar j sits at (j - (nbackends-1)/2) widths from
# the centre.  Used below to place each bar's value above it.
width = 1.0 / (nbackends + cluster_gap)
bar(j) = (j - (nbackends - 1) / 2.0) * width

set grid ytics lc rgb "#dddddd"
set ylabel "Runtime relative to primal" offset 1.5, 0
set format y "%g×"
set xtics scale 0 nomirror
set key outside bottom center horizontal samplen 2

set lmargin 7.5
set rmargin 1.5
set tmargin 2.5                 # enough for the title
set bmargin 4                   # the method labels are two lines tall

# The primal itself: a bar reaching this line would mean the Jacobian cost no
# more than a single objective evaluation.
# Values above bars: no decimals once the number is large, so they stay narrow.
barlabel(x) = x >= 100 ? sprintf("%.0f", x) : sprintf("%.1f", x)

set arrow 1 from graph 0, first 1 to graph 1, first 1 nohead \
    dashtype 2 lw 2 lc rgb "#cc0000" front

if (logscale) {
  set logscale y
  set yrange [0.8:*]
  set offsets -0.55, -0.55, 0, 0
} else {
  set yrange [0:*]
  # Negative x offsets pull the axis in towards the outermost clusters, which
  # gnuplot otherwise pads by most of a full row width on each side.
  set offsets -0.55, -0.55, graph 0.08, 0
}

do for [p in programs] {
  set output sprintf("plots/%s.%s", p, fmt eq "png" ? "png" : "pdf")
  set title sprintf("%s - computing the full Jacobian", p) font "sans,12"
  plot sprintf("plots/%s.dat", p) using 2:xtic(1) title "CPU (seq)" lc rgb "#4878cf", \
                               '' using 3        title "CPU (mt)"  lc rgb "#ee854a", \
                               '' using 4        title "GPU"       lc rgb "#6acc64", \
     for [j=0:nbackends-1] '' using ($0 + bar(j)):(column(j+2)):(barlabel(column(j+2))) \
                              with labels notitle font sprintf("sans,%g", labelsize) offset 0, 0.5 \
                              textcolor rgb "#333333"
}

unset output
