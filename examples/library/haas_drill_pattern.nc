%
O00002 (dialect: haas)
(request: Drill four 0.25 inch holes 0.5 inch deep at the corners of a 1 inch square on the Haas mill using the simple drilling cycle)
G20 G90 G17 G40 G49 G80
T2 M06
G54
G43 H02 Z1.
S2500 M03
M08
G00 X0.5 Y0.5
G00 Z0.1
G98 G81 Z-0.5 R0.1 F10.
X1.5
Y1.5
X0.5
G80
G00 Z1.
M09
M05
G28 G91 Z0.
G90
M30
%
