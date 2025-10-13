log:
scaled
Info    : 0.00 < quality < 0.10 :     11923 elements
Info    : 0.10 < quality < 0.20 :      6401 elements
Info    : 0.20 < quality < 0.30 :      8258 elements
Info    : 0.30 < quality < 0.40 :     21325 elements
Info    : 0.40 < quality < 0.50 :     27924 elements
Info    : 0.50 < quality < 0.60 :     37442 elements
Info    : 0.60 < quality < 0.70 :     50327 elements
Info    : 0.70 < quality < 0.80 :     60515 elements
Info    : 0.80 < quality < 0.90 :     50822 elements
Info    : 0.90 < quality < 1.00 :     16861 elements
Info    : Done optimizing mesh (Wall 1.05189s, CPU 1.03473s)

stacked
Info    : 0.00 < quality < 0.10 :     12253 elements
Info    : 0.10 < quality < 0.20 :      7847 elements
Info    : 0.20 < quality < 0.30 :     18156 elements
Info    : 0.30 < quality < 0.40 :     20944 elements
Info    : 0.40 < quality < 0.50 :      9704 elements
Info    : 0.50 < quality < 0.60 :     11950 elements
Info    : 0.60 < quality < 0.70 :     22018 elements
Info    : 0.70 < quality < 0.80 :     33771 elements
Info    : 0.80 < quality < 0.90 :     19000 elements
Info    : 0.90 < quality < 1.00 :      4970 elements
Info    : Done optimizing mesh (Wall 1.06856s, CPU 1.06847s)


# hmin = no effect
#  use ball field = no effect
# --point-radius-mult = no effect
# --worst-frac = converges faster to same thing?


okay so i am looping through all points and doing setNode() on all points. can you write a simple script that also updates the mesh size somehow (with an additional command or anything you like). i would like the mesh size to be proportional to aspect ratio between 