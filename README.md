# LLama3 project
This soucre code is the inference pipeline of LLama3 which can run in Linux locallay
## Prequisites
- Install and activate a virtual environment:
```bash
conda create -n llama3 python=3.11
conda activate llama3
```
- Install some requirements. Note that the transofrmer and accelerate lib are only used for
 exporting original model to torchScript model. For inference, we don't need those
```bash
pip install -r requirements.txt
```
## How to run:
- To export the original model weight file to torch script(Optinoal because I already exported):
    - Convert original checkpoitn (can be downloaded at [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main/original) ) to safetensors format to be loaded by Autotokenizer (required at least 16GB RAM)
        ```bash
        python3 utils/convert_llama_weights_to_hf.py \
    --input_dir weights --model_size 8B --output_dir weights --llama_version 3
        ```
    - Export to torch script:
     ```bash
     python3 utils/export_torchScript
     ```
- Run inference:
```bash
python3 main.py --tokenizer tokenizer.model --max_gen_len 10 --temperature 0.3 
```
You can test with another tokenizer file (cl100k_base.tiktoken (for GPT4))