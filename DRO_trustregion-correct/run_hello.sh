#/bin/bash
#BSUB -J hello
#BSUB -e /nfsshare/home/lichenxi/log/hello_%J.err
#BSUB -o /nfsshare/home/lichenxi/log/hello_%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
python hello.py