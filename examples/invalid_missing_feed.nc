(dialect: linuxcnc)
(BROKEN ON PURPOSE: the first cutting move has no F word and no feed was set before it -> E005)
G21 G90 G17
T1 M6
S8000 M3
G0 Z5
G0 X0 Y0
G1 Z-1
G1 X30 Y0
G1 X30 Y30
G0 Z5
M5
M30
