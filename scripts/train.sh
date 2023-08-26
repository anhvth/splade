export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_NAME="config_splade_vi.yaml"
python -m splade.train config.checkpoint_dir=outputs/splade_vi 
