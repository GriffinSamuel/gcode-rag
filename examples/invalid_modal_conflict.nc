(dialect: linuxcnc)
(BROKEN ON PURPOSE: G0 and G1 are both motion codes in one block, and G90 with G91 -> E004)
G21 G90 G17
T1 M6
S8000 M3
G0 Z5
G0 G1 X10 Y10 F300
G90 G91 X5
G0 Z5
M5
M30
