# proj_split_mpi: a Parallel Implementation of Projective Splitting for the Lasso

This is a parallel implementation of the projective splitting algorithm, as seen 
here: ) using MPI (mpi4py in Python).

To use this, you need to install mpi4py (see here: )

To run a script
mpirun -n 4 python name_of_script.py

This will launch 4 MPI processes. 

The scripts that will actually run experiments begin with "run", i.e.
run_fixed_nslices.py
run_master_slave.py
run_ps_mpi_lasso.py

The actual implementations are in 
proj_split_mpi4py_sync_lasso.py
proj_split_mpi4py_sync_lasso_v1.py
ps_lasso.py
ps_master_slave.py

An exception to this is 
parallelFista.py
which will run our parallel version of FISTA and also includes the source code

The shell scripts 
get_times.sh
runTimesFista.sh
can be used for timing different number of processes and assesing the parallel speedup factor.