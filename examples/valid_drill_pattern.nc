(dialect: linuxcnc)
(4-hole bolt pattern, 5 mm drill, 10 mm deep, peck drilling with G83)
G21 G90 G17 G94
T2 M6
G43 H2
S2500 M3
M8
G0 Z10
G0 X-30 Y-30
G98 G83 X-30 Y-30 Z-10 R2 Q3 F120
X30 Y-30
X30 Y30
X-30 Y30
G80
G0 Z25
M9
M5
M30
