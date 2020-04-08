#!/bin/bash
echo >> times.txt
echo ----------------------- >> times.txt
/usr/bin/time -a -o times.txt mpirun -n 1 python parallelFista.py
/usr/bin/time -a -o times.txt mpirun -n 2 python parallelFista.py
/usr/bin/time -a -o times.txt mpirun -n 5 python parallelFista.py
/usr/bin/time -a -o times.txt mpirun -n 10 python parallelFista.py
echo ----------------------- >> times.txt
echo >> times.txt
