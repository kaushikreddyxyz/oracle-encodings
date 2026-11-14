#!/bin/bash
# pod side: run_sae.sh <tag> <site> <jobs.json> <nworkers> [extra args]   -> trains every job with <nworkers> concurrent processes on the one GPU
TAG=$1; SITE=$2; JOBS=$3; NW=$4; shift 4
cd /workspace/steelman; OUT=/workspace/out/$TAG; mkdir -p $OUT/sae-$SITE; rm -f $OUT/sae-$SITE/*.claim
python sae_sweep.py --tag $TAG --site $SITE --out $OUT --prep || exit 1
for w in $(seq 0 $((NW-1))); do
  ( python sae_sweep.py --tag $TAG --site $SITE --jobs $JOBS --out $OUT --worker $w --nworkers $NW "$@" > $OUT/sae-$SITE/worker$w.log 2>&1 ) &
done
wait
grep -h -E "^SAE|OOM|Traceback|Error" $OUT/sae-$SITE/worker*.log | tail -n 400
python sae_sweep.py --tag $TAG --site $SITE --out $OUT --merge --part ${PART:-p0}
