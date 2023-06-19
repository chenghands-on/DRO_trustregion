#/bin/bash
#BSUB -J DRO
#BSUB -e /nfsshare/home/lichenxi/log/DRO_%J.err
#BSUB -o /nfsshare/home/lichenxi/log/DRO_%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
python my_train.py --epoch 1