import re

re.findall(r'Hannes|Hobson|Cole', 'Hobson Lane, Cole Howard, and Hannes Max Hapke')

re.findall(r'H|Hobson|Cole', 'Hobson Lane, Cole Howard, and Hannes Max Hapke')

import re

match = re.match(r'(kitt|dogg)y', "doggy")

match.group()

match.group(0)

match.groups()

match = re.match(r'((kitt|dogg)(y))', "doggy")  # <1>

match.groups()

match.group(2)
