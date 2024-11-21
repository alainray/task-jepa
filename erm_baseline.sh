#!/bin/bash
#SBATCH --job-name=erm_base
#SBATCH -t 6-00:00                    # tiempo maximo en el cluster (D-HH:MM)
#SBATCH -o storage/jepa/exp_logs/%x_%j.out                 # STDOUT (A = )
#SBATCH -e storage/jepa/exp_logs/%x_%j.err                 # STDERR
#SBATCH --mail-type=END,FAIL         # notificacion cuando el trabajo termine o falle
#SBATCH --mail-user=afraymon@uc.cl    # mail donde mandar las notificaciones
#SBATCH --chdir=/user/araymond    # direccion del directorio de trabajo
#SBATCH --partition=ialab-high
#SBATCH --nodelist=grievous            # forzamos scylla
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # numero de nodos a usar
#SBATCH --ntasks-per-node=1          # numero de trabajos (procesos) por nodo
#SBATCH --cpus-per-task=1           # numero de cpus (threads) por trabajo (proceso)


export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv shell mini

cd storage/jepa
python run_expt.py --train_method erm --dataset shapes3d --seed $seed
echo "Finished with job $SLURM_JOBID"
