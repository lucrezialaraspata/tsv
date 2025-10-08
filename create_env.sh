uv venv --python 3.11.5
source .venv/bin/activate
uv pip install -r requirements.txt

huggingface-cli download --repo-type dataset truthfulqa/truthful_qa
#huggingface-cli download meta-llama/Llama-3.1-8B