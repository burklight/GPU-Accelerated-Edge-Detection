#!/bin/bash -l

# job name
#SBATCH -J myjob
# account
#SBATCH -A edu18.SF2568
# email notification
#SBATCH --mail-type=BEGIN,END
# 10 minutes wall-clock time will be given to this job
#SBATCH -t 00:10:00
# Number of nodes
#SBATCH --nodes=2
# set tasks per node to 24 in order to disablr hyperthreading
#SBATCH --ntasks-per-node=24

module add i-compilers intelmpi

for processes in {1..8}
do
  for values in {1..6}
    do
      mpirun -np $processes ./trySave $values
    done
done
