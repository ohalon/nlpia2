mkdir -p ../xml
mkdir -p ../html


for p in *.adoc ; do
  echo "path: $p";
  asciidoctor -d book -b html5 "$p" -o ../html/
done

for p in *.html ; do
  echo "path: $p";
  pandoc -t xml -o ../xml/"$p".xml --indent=2 "$p"
done
