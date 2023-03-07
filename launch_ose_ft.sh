#!/bin/bash

lstart=("2017-01-01" "2017-02-01" "2017-03-01" "2017-04-01" "2017-05-01" "2017-06-01" "2017-07-01" "2017-08-01" "2017-09-01" "2017-10-01" "2017-11-01" "2017-12-01")
lend=("2017-01-31" "2017-02-28" "2017-03-31" "2017-04-30" "2017-05-31" "2017-06-30" "2017-07-31" "2017-08-31" "2017-09-30" "2017-10-31" "2017-11-30" "2017-12-31")

lstart=("2017-01-01" "2017-02-01")
lend=("2017-01-31" "2017-02-28")

len=${#lstart[@]}

for (( i=0; i<$len; i++ )); do
  start="${lstart[$i]}"
  end="${lend[$i]}"
  start2=$(date -d "$start - 15 days" '+%Y-%m-%d')
  end2=$(date -d "$end + 12 days" '+%Y-%m-%d')
  HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_ose/duacs_free_ose_gf_wosst file_paths=jz_ose entrypoint=run entrypoint.max_epochs=10 +entrypoint.ckpt_path="../../ose/ckpt/4DVarNet_OI_OSSE_woSST.ckpt" datamodule.train_slices.0._args_.0='"'$start2'"' datamodule.train_slices.0._args_.1='"'$end2'"' datamodule.test_slices.0._args_.0='"'$start2'"' datamodule.test_slices.0._args_.1='"'$end2'"'
  N=`ls -d dashboard/df_ose_gf_wosst/version_* | cut -f2 -d'_' | wc -l`
  cp -rf dashboard/df_ose_gf_wosst/version_$((N-1))/metrics.txt $HOME/metrics_ft_${start}_${end}.txt
  rm -rf dashboard/df_ose_gf_wosst/version_$((N-1))
  HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_ose/duacs_free_ose_gf_wosst file_paths=jz_ose entrypoint=test +entrypoint.ckpt_path="../../ose/ckpt/4DVarNet_OI_OSSE_woSST.ckpt" datamodule.train_slices.0._args_.0='"'$start2'"' datamodule.train_slices.0._args_.1='"'$end2'"' datamodule.test_slices.0._args_.0='"'$start2'"' datamodule.test_slices.0._args_.1='"'$end2'"'
  N=`ls  dashboard/df_ose_gf_wosst/lightning_logs/ | cut -f2 -d'_' | head -1`
  cp -rf dashboard/df_ose_gf_wosst/lightning_logs/version_$((N))/metrics.txt $HOME/metrics_${start}_${end}.txt  
  rm -rf dashboard/df_ose_gf_wosst/lightning_logs/version_$((N))
done 
