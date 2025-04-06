egrep -i -h '^([=]+ |:Chapter: [0-9]+)' manuscript/adoc/Chapter*.adoc > docs/headings.adoc
# python -c '\
#   with open("docs/headings.adoc") as fin: \
#     lines = fin.readlines()
#   ch_lines = []
#   for i, line in enumerate(lines):
#     if line.lower().startswith(':chapter:'):
#       toks = line[-1].split()
#       ch_lines[-1] = ' '.join(toks[:1] + [line.strip()] + toks[1:])
#     else:
#       ch_lines.append(line)
#   with open("docs/headings.adoc", 'w') as fout: \
#     fout.writeines(lines)
#   '
newline=$'\n'


sed 's/=/#/g' docs/headings.adoc > docs/headings.md
sed 's/==== /      * /g' docs/headings.adoc > docs/outline.md
sed -i 's/=== /    * /g' docs/outline.md
sed -i 's/== /  * /g' docs/outline.md
sed -i 's/= /'"\\${newline}"'* /g' docs/outline.md

# rm -f headings.adoc

# pandoc --atx-headers     --verbose     --wrap=none     --toc     --reference-links     -s -o outline.adoc     outline.md


