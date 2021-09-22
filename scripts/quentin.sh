# # Commands to manage slurm jobs

# # SWOT with ROLL and NADIRS  : config_q/roll.py
# ## start training
# sbatch --parsable scripts/run.slurm q.roll >> scripts/roll.jobs
# ## See logs
# tail -f *$(tail -n1 scripts/roll.jobs)*
# ## Test latest ckpt
# sbatch --parsable scripts/test.slurm $(tail -n1 scripts/roll.jobs) q.roll >> scripts/roll.test_jobs
# ## tb latest test ckpt
# tensorboard --logdir=lightning_logs/version_$(tail -n1 scripts/roll.test_jobs) --bind_all

function link(){
	mkdir -p "dashboard/current/train"
	mkdir -p "dashboard/current/test"
	rm -f "dashboard/current/$1" 
	ln -s "../../../lightning_logs/version_$2" "dashboard/current/$1" 
}

function train(){
	JOBID=$(sbatch --parsable scripts/run.slurm $1)
	link "train/$1" $JOBID
	echo $JOBID
}

function test(){
	JOBID=$(sbatch --parsable scripts/test.slurm $1)
	link "test/$1" $JOBID
	echo $JOBID
}

function stash(){
	mv dashboard/current dashboard/$1
}

function train_cmds(){
	fd . config_q/$1 | sed -n 's?/?\.?g;s/\.py//;s/config_//;p' | grep -v __in | xargs -I{} echo train {}
}

function test_cmds(){
	fd . config_q/$1 | sed -n 's?/?\.?g;s/\.py//;s/config_//;p' | grep -v __in | xargs -I{} echo test {}
}
