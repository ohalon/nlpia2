filepath=./manuscript/adoc/TOC.automatic.adoc
mdpath="docs/outline.md"
echo << EOF > $filepath
= contents
Hobson Lane, Maria Dyshel
Lane / Natural Language Processing in Action 2nd Edition
:doctype: book

:unnumbered:

[[chapter_id_0]]
[role="available free"]
== Acknowledgments

[[part_id_1]]
= Part 1: Wordy Machines (Vector Models of Natural Language)
EOF

egrep -i -h '^([=]+[ ][a-zA-Z0-9] |:Chapter: [0-9]+)' manuscript/adoc/Chapter*.adoc > $filepath
egrep -i -h '^([=]+[ ][a-zA-Z0-9] |:Chapter: [0-9]+)' manuscript/adoc/Appendix*.adoc >> $filepath
egrep -i -h '^([=]+[ ][a-zA-Z0-9] |:Chapter: [0-9]+)' manuscript/adoc/Resource*.adoc >> $filepath
egrep -i -h '^([=]+[ ][a-zA-Z0-9] |:Chapter: [0-9]+)' manuscript/adoc/Glossary*.adoc >> $filepath
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


sed 's/=/#/g' $filepath > $mdpath
sed 's/==== /      * /g' docs/headings.adoc > $mdpath
sed -i 's/=== /    * /g' $mdpath
sed -i 's/== /  * /g' $mdpath
sed -i 's/= /'"\\${newline}"'* /g' $mdpath

# rm -f headings.adoc

# pandoc --atx-headers     --verbose     --wrap=none     --toc     --reference-links     -s -o outline.adoc     outline.md


