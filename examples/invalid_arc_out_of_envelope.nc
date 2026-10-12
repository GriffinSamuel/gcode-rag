(dialect: linuxcnc)
(BROKEN ON PURPOSE: G2 arc with no I/J/R -> E006, and an incremental move past X 400 mm -> E009)
G21 G90 G17
T1 M6
S8000 M3
F300
G0 Z5
G1 Z-1
G2 X20 Y20
G91
G1 X250
G1 X250
G90
G0 Z5
M5
M30
