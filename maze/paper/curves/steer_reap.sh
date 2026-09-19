#!/bin/bash
# bash steer_reap.sh : pull results of finished shards (eval outputs, prep stats/SAEs, logs), then delete those pods.
HERE="$(cd "$(dirname "$0")" && pwd)"; R="$HERE/results/distance"; mkdir -p "$R/steer_out" "$R/steer_prep" "$R/steer_tune"
while read -r sh id ip port; do a=runpod-mzs-$sh; [ -n "${ALT:-}" ] && [ "$sh" = "${ALT%%:*}" ] && a="${ALT##*:}"
  grep -q "^$sh reaped" "$R/steer_logs/${REAPED:-reaped.txt}" 2>/dev/null && continue
  ssh -n -o ConnectTimeout=10 "$a" 'grep -q SHARD_DONE /workspace/steer.log' 2>/dev/null || continue
  want=$(( $(wc -l < "$SHARDS/$sh.dirs") * ${NCOND:-5} ))
  rsync -az -e "ssh -o ConnectTimeout=15" "$a:/workspace/steer_out/eval/" "$R/steer_out/" </dev/null || continue
  rsync -az -e "ssh -o ConnectTimeout=15" "$a:/workspace/steer_out/tune/" "$R/steer_tune/" </dev/null 2>/dev/null
  rsync -az -e "ssh -o ConnectTimeout=15" "$a:/workspace/steer/" "$R/steer_prep/" --exclude 'train-*.pt' </dev/null || continue
  scp -q "$a:/workspace/steer.log" "$R/steer_logs/steer${LOGTAG:-}-$sh.log" </dev/null
  have=$(ssh -n "$a" "ls /workspace/steer_out/eval/${ONLY:-*}/*.npz | wc -l")
  if [ "$have" -ge "$want" ]; then runpodctl pod delete "$id" </dev/null >/dev/null 2>&1 && ~/.claude/skills/runpod-spinup/deregister-ssh-alias.sh "${a#runpod-}" </dev/null >/dev/null 2>&1; echo "$sh reaped ($have files)" | tee -a "$R/steer_logs/${REAPED:-reaped.txt}"
  else echo "$sh: only $have/$want outputs, not deleting"; fi
done < "$R/steer_logs/${EP:-endpoints.txt}"
