#!/bin/bash
# Mac-side: poll the RunPod API-free way — wait until a pod's direct ssh is up; args: <alias>
n=0; until ssh -o ConnectTimeout=6 -o BatchMode=yes "$1" 'echo ready' 2>/dev/null | grep -q ready; do sleep 6; n=$((n+1)); [ $n -gt 60 ] && { echo "$1: no sshd"; exit 1; }; done; echo "$1 up"
