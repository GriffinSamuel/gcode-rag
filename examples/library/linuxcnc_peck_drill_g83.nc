(dialect: linuxcnc)
(request: Peck drill three 6 mm holes 20 mm deep in a row with 3 mm pecks to clear the chips)
(work zero: first hole, top of stock)
G21 G90 G17 G40 G49 G94
T3 M6
G43 H3
S1500 M3
M8
G0 Z10
G0 X0 Y0
G98 G83 X0 Y0 Z-20 R2 Q3 F100
X25
X50
G80
G0 Z25
M9
M5
M30
