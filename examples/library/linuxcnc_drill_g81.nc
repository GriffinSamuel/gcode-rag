(dialect: linuxcnc)
(request: Drill four 5 mm holes 8 mm deep at the corners of a 60 mm square centred on the origin using the simple drilling cycle)
(work zero: centre of the pattern, top of stock)
G21 G90 G17 G40 G49 G94
T2 M6
G43 H2
S2000 M3
G0 Z10
G0 X-30 Y-30
G98 G81 X-30 Y-30 Z-8 R2 F150
X30 Y-30
X30 Y30
X-30 Y30
G80
G0 Z25
M5
M30
