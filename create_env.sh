uv venv --python 3.11.5
source .venv/bin/activate
uv pip install -r requirements.txt

hf download --repo-type dataset truthfulqa/truthful_qa
hf download meta-llama/Llama-3.1-8B