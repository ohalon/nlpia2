ln -s code ../nlpia2/src/nlpia2
source code/scripts/conda_install.sh

conda activate nlpia2

sudo apt install -y inkscape build-essential

conda install -y -c conda-forge -c defaults asciidoctor xmltodict pandoc

git config --global user.name "${USER}"
git config --global core.autocrlf True
git config --global core.eol lf
git config --global mergetool.meld.path /usr/bin/meld
git config --global merge.tool meld
git config --global diff.tool meld

git remote add all git@gitlab.com:hobs/nlpia-manuscript.git
git remote add upstream	git@git.manning.com:agileauthor/lane3.git
git remote add origin git@gitlab.com:hobs/nlpia-manuscript.git
git remote add manning git@git.manning.com:agileauthor/lane3.git

# push to two repos using all rempte
git remote set-url all git@gitlab.com:hobs/nlpia-manuscript.git
git remote set-url --add all git@git.manning.com:agileauthor/lane3.git
git config branch.master.remote all
git pull all master
git push -u all master

git config --list --show-origin > docs/git-config.new.txt
