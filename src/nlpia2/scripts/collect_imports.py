import os
import subprocess
import re
import pandas as pd
from pathlib import Path
import argparse

DIRS 		= ['nlpia-manuscript', 'nlpia2']
DF_COLUMNS  = ['Module', 'Line', 'Line #', 'Path', 'Repo Root', 'Pattern']

## IDENTIFY GIT REPO ROOT
# This ensures the script will work even if it's moved around within the repository
REPO_COMMAND = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True)
CURRENT_REPO_ROOT =Path( REPO_COMMAND.stdout.strip(b'\n').decode('utf-8') )

REPOS = [ (CURRENT_REPO_ROOT.parent / directory) for directory in DIRS]
MANUSCRIPT_ADOCS = CURRENT_REPO_ROOT.parent / 'nlpia-manuscript' / 'manuscript' / 'adoc'


## REGULAR EXPRESSION PATTERNS
FROM_R 		= r'[>\s]*from\s+(?P<from_import>\w+[\w\.]*)\s+import\s+.*#*.*'
#https://regex101.com/r/afZx6f/1

IMPORT_AS_R = r'[>\s]*import\s+(?P<import_as>\w+)\.?\w*\s+(\sas\s)?.*#*.*'
#https://regex101.com/r/ALfXkh/1

IMPORT_R 	= r'[>\s]*import\s+(?P<import_direct>(\w+[,\s*]?)*)\s*[#]*.*'
#https://regex101.com/r/x2gxek/1

PATTERNS_R 	= (FROM_R, IMPORT_AS_R, IMPORT_R)


def inspect_files(root, pattern, expanded=False):
	"""
	Expects a local path and a pattern of documents for glob
	Inspects all the files that match the pattern, and extracts the import statements into a dataframe.
	Prints on screen a set of all imported modules found
	"""
	print('\n'*3, '*'*30, str(root), sep="")
	root=str(root)
	import_lines=[]
	terms_rc = re.compile(r'|'.join(PATTERNS_R))
	for filepath in Path(root).rglob(pattern):
		with open(filepath) as f:
			for number, line in enumerate(f):
				line = line.strip('\n ')
				m = terms_rc.match(line) 
				if m:
					modules = m.group('from_import') or m.group('import_as') or m.group('import_direct') 
					import_lines.append( (modules, line,  number+1, str(filepath).split(root)[1], root, pattern) )
	modules_df =  pd.DataFrame(import_lines, columns = DF_COLUMNS)
	modules_df['Module'] = modules_df['Module'].apply(lambda x: x.split('.')[0] if '.' in x else x)
	modules_df['Module'] = modules_df['Module'].apply(lambda x: [y.strip() for y in x.split(',')] if ',' in x else x)
	modules_df = modules_df.explode(['Module'], ignore_index=True)
	if expanded==True:
		with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
			print( modules_df.groupby(by=['Module', 'Path', 'Line #', 'Line'])['Line'].agg('count'))
	print('\nModules identified, with counts: ')
	print ( modules_df['Module'].value_counts())
	
	print("\nList of Modules: \n\t", end="")
	print( * sorted(set( modules_df['Module'])), sep = ", ")
	
	return modules_df


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="%(prog)s recursively traverses repositories or subrepo directory structures, \
		inspecting files whose names match a given name pattern, to collect package names that appear in import statements. The output \
		lists such packages along with their counts. With the optional --expand flag, %(prog)s additionally outputs the line numbers and filenames where \
		the imports were found.")
	parser.add_argument("-a", "--all", dest="all", action='store_true',
		default = False, 
		help="exhaustively traverse both repos (nlpia-manuscript, nlpia2)")
	parser.add_argument("-p", "--pattern", dest="pattern", metavar="",
		default = "*.adoc",
		help="pattern of filenames for glob to match, default is '*.adoc'" )
	parser.add_argument("-e", "--expand", dest="expand", action='store_true',
		default=False,
		help="additionally prints each occurence of a package mention, along with file name and line number", )
	args = parser.parse_args()

	if args.all == False:													# checks only the manuscript_adocs directory
		repo_root = MANUSCRIPT_ADOCS
		df = inspect_files(repo_root, args.pattern, args.expand)
	else:																	# otherwise, check all repos in the DIRS list
		for repo_root in REPOS: 
			df = inspect_files(repo_root, args.pattern, args.expand)
