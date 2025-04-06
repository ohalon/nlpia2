egrep -h '^= ' manuscript/adoc/Chapter*.adoc > manuscript/adoc/TOC.automatic.adoc
sed 's/= /### /g' manuscript/adoc/TOC.automatic.adoc > docs/headings.md
# sed 's/= /* /g' docs/headings.adoc > docs/outline.md
sed -i 's/= /=== /g' manuscript/adoc/TOC.automatic.adoc
# rm -f headings.adoc

# pandoc --atx-headers     --verbose     --wrap=none     --toc     --reference-links     -s -o outline.adoc     outline.md


