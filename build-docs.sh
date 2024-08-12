rm -rf docs/build
rm -rf docs/jupyter_execute
jupyter-book config sphinx docs/
python docs/autogen_rst.py
python docs/create_toc.py
sphinx-build -W -b html docs docs/build
