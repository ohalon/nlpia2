# ch11 information extraction 

## 2022-07-22 Vish sprint planning

- clone the manuscript repo: https://gitlab.com/tangibleai/nlpia-manuscript
- create a vish-ch11 branch
- add your manuscript/Chapter 11....adoc file to the repo
- push your changes to your branch
- open a merge request to master
- fix heading formatting
- put newline character between sentences in all adoc natural language text
- display dataframes as copy pasted snippets of text
- split the table and code blocks
- commit often (several times a day)
- push to gitlab daily
- remove !pip install commands from code blocks
- always use the `en_core_web_md` model in spacy
- clone the code repo: https://gitlab.com/tangibleai/nlpia2
- put code snippets into a py file or jupyter notebook here on the main branch: https://gitlab.com/tangibleai/nlpia2/-/tree/main/src/nlpia2/ch11
- install the nlpia2 package from source by running nlpia2/scripts/install_editable.sh` from the nlpia2 directory (`./scripts/install_editable.sh`)
- run your code snippets
- minimize new dependencies and have a good justification why readers of this book need to install this package in order to understand how to apply what they learn in the real world
- add new dependencies to nlpia2/requirements.txt and nlpia2/pyproject.toml
- install asciidoctor syntax highlighter and linter in sublime/vscodium
- create a weekly meeting on Thurs 5 pm for sprint planning and retro