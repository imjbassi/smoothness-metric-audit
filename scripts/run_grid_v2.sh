#!/bin/bash
# Run every config in configs_v2/jobs.txt, P at a time, skipping runs that
# already finished (a DONE marker in results_train_v2/<name>/).
#   cd ~/robomimic && . .venv/bin/activate && bash run_grid_v2.sh 2
P=${1:-2}
mkdir -p results_train_v2/_logs
run() {
    cfg=$1; name=$(basename "$cfg" .json)
    [ -f "results_train_v2/$name/DONE" ] && exit 0
    rm -rf "results_train_v2/$name"
    if python robomimic/scripts/train.py --config "$cfg" \
           > "results_train_v2/_logs/$name.log" 2>&1; then
        touch "results_train_v2/$name/DONE"
        echo "$(date '+%F %T') done $name"
    else
        echo "$(date '+%F %T') FAILED $name"
    fi
}
export -f run
xargs -a configs_v2/jobs.txt -P "$P" -I{} bash -c 'run {}'
echo "$(date '+%F %T') grid finished"
