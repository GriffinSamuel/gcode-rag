(dialect: linuxcnc)
(BROKEN ON PURPOSE: M13 is a Haas spindle-on-with-coolant code that LinuxCNC does not know -> E003)
G21 G90 G17
T1 M6
S8000 M13
G0 Z5
G1 Z-1 F150
G1 X20
G0 Z5
M5
M30
