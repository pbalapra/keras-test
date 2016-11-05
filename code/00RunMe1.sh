qsub -q knl_7210 -t 0:10:00 -n 1 --env "PATH=/home/pbalapra/.virtualenvs/intel-tflow/bin:$PATH" --env "PYTHONPATH=/home/pbalapra/.virtualenvs/intel-tflow/lib64/python2.7/site-packages" --env "OMP_NUM_THREADS=32" --env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/soft/compilers/intel/mkl/lib/intel64 /usr/bin/numactl -m 0 python 02_runRegression.py miniamr_03


