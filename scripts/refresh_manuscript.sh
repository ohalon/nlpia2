#!/usr/bin/env bash
# Delete all *.adoc files in nlpia2/data/manuscript/adoc 
# and replaces with hard links from nlpia-manuscript/manuscript/adoc/

# FIXME: gitlab doesn't like hard links to unchanged files (prereceive hook)
# TODO: make this a for loop

# rm -f src/nlpia2/data/manuscript/adoc/Chapter*.adoc
# git commit -am 'rm Chapter*'
SCRIPT_NAME="nlpia2/scripts/refresh_manuscript.sh"
DEST="src/nlpia2/data/manuscript/adoc"

PREFIX='Chapter'
git add $DEST
git commit -am "$SCRIPT_NAME: git-commit -a before refresh of $DEST/$PREFIX*.adoc"
rm -f "$DEST/$PREFIX"*.adoc
cp -f "../nlpia-manuscript/manuscript/adoc/${PREFIX}"*.adoc $DEST
# git commit -am 'hard link Chapter*'
git add $DEST 
git commit -am "$SCRIPT_NAME: git-commit -a AFTER refresh of $DEST/$PREFIX*.adoc"

PREFIX='Appendix'
git add $DEST
git commit -am "$SCRIPT_NAME: git-commit -a before refresh of $DEST/$PREFIX*.adoc"
rm -f "$DEST/$PREFIX"*.adoc
cp -f "../nlpia-manuscript/manuscript/adoc/${PREFIX}"*.adoc $DEST
# git commit -am 'hard link Chapter*'
git add $DEST 
git commit -am "$SCRIPT_NAME: git-commit -a AFTER refresh of $DEST/$PREFIX*.adoc"

PREFIX='TOC'
git add $DEST
git commit -am "$SCRIPT_NAME: git-commit -a before refresh of $DEST/$PREFIX*.adoc"
rm -f "$DEST/$PREFIX"*.adoc
cp -f "../nlpia-manuscript/manuscript/adoc/${PREFIX}"*.adoc $DEST
# git commit -am 'hard link Chapter*'
git add $DEST 
git commit -am "$SCRIPT_NAME: git-commit -a AFTER refresh of $DEST/$PREFIX*.adoc"

PREFIX='outline'
git add $DEST
git commit -am "$SCRIPT_NAME: git-commit -a before refresh of $DEST/$PREFIX*.adoc"
rm -f "$DEST/$PREFIX"*.adoc
cp -f "../nlpia-manuscript/manuscript/adoc/${PREFIX}"*.adoc $DEST
# git commit -am 'hard link Chapter*'
git add $DEST 
git commit -am "$SCRIPT_NAME: git-commit -a AFTER refresh of $DEST/$PREFIX*.adoc"

PREFIX='Glossary'
git add $DEST
git commit -am "$SCRIPT_NAME: git-commit -a before refresh of $DEST/$PREFIX*.adoc"
rm -f "$DEST/$PREFIX"*.adoc
cp -f "../nlpia-manuscript/manuscript/adoc/${PREFIX}"*.adoc $DEST
# git commit -am 'hard link Chapter*'
git add $DEST 
git commit -am "$SCRIPT_NAME: git-commit -a AFTER refresh of $DEST/$PREFIX*.adoc"
