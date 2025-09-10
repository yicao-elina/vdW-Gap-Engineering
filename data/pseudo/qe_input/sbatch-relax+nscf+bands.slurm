#!/bin/bash -l
#SBATCH --account=pclancy3
#SBATCH --export=ALL

#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --job-name="s2t3-2d"
#SBATCH --output="slurm.%j"

#SBATCH --partition=parallel
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48

module reset

source /data/apps/go.sh

# bash commands go here. Don't forget to load the necessary modules!
ml intel
ml qe/7.2

ml
date
echo "This is sbatch-relax+nscf+bands.slurm"

if [ ! -d "pw-files" ]; then
    mkdir pw-files
fi

mpirun /data/apps/extern/qe/7.2/bin/pw.x < relax.in > relax.out
# mpirun /data/apps/extern/qe/7.2/bin/pw.x < '2-nscf.in' > '2-nscf.out'
# mpirun /data/apps/extern/qe/7.2/bin/pw.x < '3-bands.in' > '3-bands.out'
# mpirun /data/apps/extern/qe/7.2/bin/pw.x < aimd_s2t3_221.in > aimd_s2t3_221.out

# sbatch sbatch-all-post.slurm

date
