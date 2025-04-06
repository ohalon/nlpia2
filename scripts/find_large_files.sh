SIZE="+10M"
if [ -n "$1" ]; then
   SIZE="$1"
fi
sudo find . -xdev -type f -size $SIZE



