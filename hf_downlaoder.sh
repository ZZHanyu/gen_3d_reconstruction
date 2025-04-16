export HF_ENDPOINT=https://hf-mirror.com

# download model
huggingface-cli download --resume-download gpt2 --local-dir gpt2

# download dataset
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
