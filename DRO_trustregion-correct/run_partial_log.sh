#/bin/bash
#BSUB -J DRO
#BSUB -e /nfsshare/home/lichenxi/log/DRO_%J.err
#BSUB -o /nfsshare/home/lichenxi/log/DRO_%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
export DRSOM_VERBOSE=1;python my_train.py --epoch 10 --batch_size 4 --train_path '/nfsshare/home/lichenxi/DRO/data_partial' --val_path '/nfsshare/home/lichenxi/DRO/data_partial'