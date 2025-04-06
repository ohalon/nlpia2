#!/usr/bin/env bash
# release.sh
USAGE="./release.sh [MESSAGE [| VERSION]]"
DEPENDENCIES=(twine wheel setuptools pip keyring rfc3986)
# set -e
PKG_NAME=nlpia2
PKG_NAME=$( (echo $PWD | rev | cut -d'/' -f1 | rev) || ls src/ | head -n1)
echo "Building and pushing a new minor version of package named $PKG_NAME..."

USAGE="./release.sh [VERSION | [VERSION MESSAGE]]"
DEPENDENCIES=(twine wheel setuptools pip keyring rfc3986)

if [ -z "$1" ] ; then
    echo $USAGE
    exit 1
fi


ORIGIN="origin"
MAIN="main"

BRANCH=$(git branch | grep -o -E '\*\ [-/a-zA-Z]+' | cut -c3-)
echo "On branch $BRANCH"
if [[ "$BRANCH" == "$MAIN" ]] ; then
    echo "Looks like you're on the $BRANCH branch which is the main branch, so bumping the version to create a release."
else
    echo "You're not on the $MAIN branch, so exiting without creating a release."
    return 1
fi

CMD=$'try:\n  import twine\n  print("1")\nexcept ImportError:\n  print("")\n\n\n'
TWINE_INSTALLED=$(python -c "$CMD")
if [ -n "$TWINE_INSTALLED" ] ; then
   echo "twine is already installed"
else
   echo "Installing dependencies: ${DEPENDENCIES[*]}"
   conda install -c conda-forge -c defaults -y ${DEPENDENCIES[*]}
   # pip install --upgrade twine wheel setuptools
fi


if [ -n "$2" ] ; then
    VERSION=$2
    echo "VERSION=$VERSION"
else
    VERSION_TAG=$(git tag --list | sort -h | tail -n1)
    echo "Found git version tag $VERSION_TAG"
    VERSION_PIP=$(pip install $PKG_NAME && pip show $PKG_NAME | tee | head -n2 | tail -n1 | cut -d' ' -f2)
    echo "Found pypi.org version $VERSION_GIT"
    if [ -n $VERSION_PIP ] ; then
        VERSION=$VERSION_PIP
    elif [ -n $VERSION_TAG ] ; then
	VERSION=$VERSION_TAG
    echo "Old version was $VERSION"
    else
	echo "ERROR: Unable to determine the appropriate current version of the package. Aborting release."
        return 1
    fi
    VERSION=$(echo $VERSION | python -c "import sys; ver = sys.stdin.readline().split('.'); ver[-1] = str(int(ver[-1])+1); print('.'.join(ver));")
    echo "New version is $VERSION"
fi

if [ -n "$VERSION" ] ; then
    echo "New minor version: $VERSION"
else
    echo "ERROR: Unable to determine the appropriate current version of the package. Aborting release."
    echo "You may also specify a release version on the command line."
    echo "$USAGE"
    echo ""
    echo "VERSION_TAG=$VERSION_TAG"
    echo "VERSION_PIP=$VERSION_PIP"
    return 1
fi

sed -i -E 0,/'^version = [0-9]{0,3}[.]?[0-9]{0,3}[.]?[0-9]{0,3}[abrc0-9]{0,7}'/s//"version = $VERSION"/g setup.cfg
NEW_VERSION=$(grep -E 'version = [0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}' setup.cfg)
if [[ "$NEW_VERSION" == "version = $VERSION" ]] ; then
    echo "Successfully modified setup.cfg to contain '$NEW_VERSION'"
else
    echo "ERROR: Unable to modify setup.cfg to contain the new 'version = $VERSION'. Instead it contains '$NEW_VERSION'."
    return 1
fi

if [ -n "$1" ] ; then
    MESSAGE=$1
else
    MESSAGE=$(git log | head -n5 | tail -n1)
fi
echo "Release notes (commit message): $MESSAGE"

git commit -am "Version $VERSION release: $MESSAGE"
git push
git tag -a "$VERSION" -m "$MESSAGE" || (git tag --delete $VERSION && git push --delete $ORIGIN $VERSION && git tag -a "$VERSION" -m "$MESSAGE" ) || echo "ERROR: Unable to tag with version '$VERSION'"


rm -rf build
rm -rf dist
# find . -type f -size +10M -exec rm -f {} \;

python setup.py sdist bdist_wheel

if [ -z "$(which twine)" ] ; then
    echo 'Unable to find `twine` so installing it with pip.'
    conda install -c conda-forge -c defaults -y twine
fi

twine check dist/*
twine upload dist/"$PKG_NAME-$VERSION"*"-py"* --verbose
twine upload dist/"$PKG_NAME-$VERSION"*".tar"* --verbose
git push --tag

pip install -e .
