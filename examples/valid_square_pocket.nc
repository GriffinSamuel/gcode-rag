(dialect: linuxcnc)
(40 x 40 mm square pocket, 3 mm deep in two 1.5 mm passes, 6 mm end mill)
(work zero: pocket centre, top of stock)
G21 G90 G17 G40 G49 G94
T1 M6
G43 H1
S10000 M3
G0 Z5
G0 X0 Y0
(pass 1)
G1 Z-1.5 F150
G1 X-17 Y-17 F600
G1 X17
G1 Y17
G1 X-17
G1 Y-17
G1 X0 Y0
(pass 2)
G1 Z-3 F150
G1 X-17 Y-17 F600
G1 X17
G1 Y17
G1 X-17
G1 Y-17
G1 X0 Y0
G0 Z5
M5
G0 Z25
M30
