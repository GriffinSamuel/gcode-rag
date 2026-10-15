(dialect: linuxcnc)
(request: Cut the outside contour of a 60 x 40 mm rectangle 3 mm deep with cutter radius compensation on the left)
(work zero: lower left corner of the rectangle, top of stock; tool 3 is a 6 mm end mill)
G21 G90 G17 G40 G49 G94
T3 M6
G43 H3
S8000 M3
G0 Z5
G0 X-15 Y-15
G1 Z-3 F150
G1 G41 D3 X0 Y-5 F400
G1 Y40
G1 X60
G1 Y0
G1 X-5
G1 G40 X-15 Y-15
G0 Z5
M5
M30
