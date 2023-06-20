#/bin/bash
#BSUB -J DRO
#BSUB -e /nfsshare/home/xiechenghan/DRO_trustregion/log/DRO_%J.err
#BSUB -o /nfsshare/home/xiechenghan/DRO_trustregion/log/DRO_%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
cd /nfsshare/home/xiechenghan/DRO_trustregion
python my_train.py
python my_train.py --optim sgd
python my_train.py --optim drsom --gamma 0.001