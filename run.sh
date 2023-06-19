#/bin/bash
#BSUB -J DRO
#BSUB -e /nfsshare/home/xiechenghan/DRO_trustregion/log/DRO_%J.err
#BSUB -o /nfsshare/home/xiechenghan/DRO_trustregion/log/DRO_%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
python my_train.py
# python my_train.py --optim drsom