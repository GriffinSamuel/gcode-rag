(dialect: linuxcnc)
(request: Engrave the letter L, 20 mm tall and 12 mm wide, 0.2 mm deep in a single stroke with a V-bit)
(work zero: lower left of the letter, top of stock)
G21 G90 G17 G40 G49 G94
T4 M6
G43 H4
S10000 M3
G0 Z5
G0 X0 Y20
G1 Z-0.2 F100
G1 Y0 F300
G1 X12
G0 Z5
M5
M30
