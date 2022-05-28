rm -rf docs/build
python build_scripts/update_docs.py
sphinx-build -W -b html docs docs/build

