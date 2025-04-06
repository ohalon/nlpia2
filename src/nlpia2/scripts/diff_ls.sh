# diff the file list on another branch relative to master
export current_branch=master
echo "$current_branch"
git checkout "$current_branch"
export other_branch=before-maria
if [ $# -eq 1 ] ; then
   export other_branch=$1
fi
diff <(ls -1) <(git checkout $other_branch && ls -1)
git checkout $current_branch
