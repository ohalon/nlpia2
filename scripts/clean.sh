full_path=$(realpath $0)
echo $0
echo "Running $full_path ..."
dir_path=$(dirname $full_path)
echo $full_path
set -e
echo 'find "'$dir_path'"/.. -name '"*.pyc"' -exec rm -rf "{}" \;'
# find "$dir_path"/.. -name '*.egg-info' -exec rm -rf "{}" \;
# rm -rf "$dir_path"/../build
# rm -rf "$dir_path"/../.eggs

rm -rf src/qary.egg-info/
rm -rf .eggs/ dist build
find . -name '*.egg-info' -exec rm -rf "{}" \;
find . -name '*.pyc' -exec rm -rf "{}" \;
