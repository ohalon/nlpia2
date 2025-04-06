# # cat ~/.pypirc
# [pypi]
# username = __token__
# password = <the token value, including the `pypi-` prefix>

rm -rf dist/*.tar.gz  # --sdist
rm -rf dist/*.whl  # --wheel
python -m pip install build twine
python -m build --sdist --wheel
twine check dist/*
twine upload dist/*
