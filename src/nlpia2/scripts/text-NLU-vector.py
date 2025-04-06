    g = Digraph(name)
    g.attr(rankdir='LR')
from graphviz import Digraph
dir(g)
    g = Digraph(name)
    g.attr(rankdir='LR')
    g = Digraph('text-NLU-vector')
    g.attr(rankdir='LR')
g.edge('Text\n(natural language)', 'NLU\n(rules, patterns, encoder)')
g.edge('NLU\n(rules, patterns, encoder)', 'Vector\n(tensor, embedding, encoding)')
g.save('text-NLU-vector.svg')
ls -al
firefox text-NLU-vector.svg
!firefox text-NLU-vector.svg
g.save()
!firefox text-NLU-vector.svg
g.format('svg')
g.format?
g.format
g.format = 'svg'
g.save()
!firefox text-NLU-vector.svg
ls -al
g.save(format='svg')
g.save?
g.format?
g.attr?
g.render(filename='text-NLU-vector.svg', cleanup=1, view=0)
g.render(filename='text-NLU-vector', cleanup=1, view=0)
!firefox text-NLU-vector.svg
g.render(filename='text-NLU-vector', cleanup=1, view=0, format='png')
history -f 'text-NLU-vector.py'
