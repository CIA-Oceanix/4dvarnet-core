# # Commands to manage slurm jobs

# utils
function link(){
	mkdir -p "dashboard/current/train"
	mkdir -p "dashboard/current/test"
	rm -f "dashboard/current/$1" 
	ln -s "../../../lightning_logs/version_$2" "dashboard/current/$1" 
}

# launch training of {config_name} using scripts/run.slurm and create or replace the symlink in dashboard/current/train/{config_name}
# example: train q.xp_icassp.direct_phi
function train(){
	JOBID=$(sbatch --parsable scripts/run.slurm $1)
	link "train/$1" $JOBID
	echo $JOBID
}

function icassp_train(){
	JOBID=$(sbatch --parsable scripts/icassp.slurm $1)
	link "train/$1" $JOBID
	echo $JOBID
}

function hydra_train(){
	JOBID=$(sbatch --parsable scripts/hydra_train.slurm $1)
	link "train/$1" $JOBID
	echo $JOBID
}

function hydra_test(){
	JOBID=$(sbatch --parsable scripts/hydra_test.slurm $1)
	link "test/$1" $JOBID
	echo $JOBID
}


# launch test on the latest checkpoint of {config_name} using scripts/test.slurm and create or replace the symlink in dashboard/current/test/{config_name}
function test(){
	JOBID=$(sbatch --parsable scripts/test.slurm $1)
	link "test/$1" $JOBID
	echo $JOBID
}

# move current folder to {arg}, useful to keep current xp results and not erasing them while launching new training
function stash(){
	mv dashboard/current dashboard/$1
}

# requires fd
function train_cmds(){
	fd . config_q/$1 | sed -n 's?/?\.?g;s/\.py//;s/config_//;p' | grep -v __in | xargs -I{} echo train {}
}

function test_cmds(){
	fd . config_q/$1 | sed -n 's?/?\.?g;s/\.py//;s/config_//;p' | grep -v __in | xargs -I{} echo test {}
}
