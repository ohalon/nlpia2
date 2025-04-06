#!/usr/bin/env bash
# rename all files from *.asc to *.adoc
for f in *.asc; do 
    mv -- "$f" "${f%.asc}.adoc"
done
