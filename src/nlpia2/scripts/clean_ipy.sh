#!/usr/bin/env bash
# 0. hist -o -p
# 1. expand '\n' in strings to actual newlines
# 2. summarize (head -n 5) lengthy output from ipython console commands 
if [[ -n "$1" ]] ; then
    FD=$1
    echo "found argv[1]"
else
    FD=<&0
    echo "found stdin"
fi
echo "FD=$FD"
cat $FD | sed 's/\\n/\n/g' | egrep '^[.]{3}[ ]|^>>>[ ]' -A5  | sed 's/^--/.../g'
