#!/bin/bash
/usr/bin/time -a -o times.txt mpirun -n 1 python run_fixed_nslices.py
/usr/bin/time -a -o times.txt mpirun -n 2 python run_fixed_nslices.py
/usr/bin/time -a -o times.txt mpirun -n 5 python run_fixed_nslices.py
/usr/bin/time -a -o times.txt mpirun -n 10 python run_fixed_nslices.py
