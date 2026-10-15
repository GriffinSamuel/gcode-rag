(dialect: linuxcnc)
(request: Face the top of a 100 x 60 mm block, 0.5 mm deep, with a 50 mm face mill)
(work zero: front left corner of the stock, top surface)
G21 G90 G17 G40 G49 G94
T1 M6
G43 H1
S3000 M3
G0 Z5
G0 X-30 Y15
G1 Z-0.5 F300
G1 X130 F600
G0 Z5
G0 Y45
G1 Z-0.5 F300
G1 X-30 F600
G0 Z5
M5
G0 Z25
M30
