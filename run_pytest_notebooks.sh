if ! git lfs pull; then
  printf "\n\nERROR: git lfs pull failed\n\n"
  exit
fi
export PYTHONPATH="`realpath src`"
pytest notebooks