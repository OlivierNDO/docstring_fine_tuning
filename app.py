from dotenv import load_dotenv
load_dotenv()


from flask import Flask, render_template, request, jsonify
import threading
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class TinyLlamaDocStringInference:
    def __init__(self, model_path: str = "./tiny_llama_finetuned", use_quantization: bool = True, local_files_only: bool = False):
        self.model_path = model_path
        self.use_quantization = use_quantization
        self.local_files_only = local_files_only
        self.prompt_prefix = "Generate a Python docstring for the following function:\n\n"

        is_finetuned = os.path.exists(os.path.join(model_path, "adapter_config.json"))

        if is_finetuned:
            print(f"Loading fine-tuned model with LoRA adapter from: {model_path}")
            base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

            bnb_config = None
            if use_quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=local_files_only,
                trust_remote_code=True
            )

            self.model = PeftModel.from_pretrained(base_model, model_path)

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                local_files_only=local_files_only,
                trust_remote_code=True
            )

        else:
            print(f"Loading base model from: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=local_files_only
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                trust_remote_code=True
            )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_id = self.tokenizer.eos_token_id

    def format_prompt(self, instruction: str) -> str:
        prompt = self.prompt_prefix + instruction
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def submit_prompt(self, instruction: str, max_new_tokens: int = 512) -> str:
        input_text = self.format_prompt(instruction)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        output_text = generated.strip()
        if '<|end|>' in output_text:
            output_text = output_text.split("<|end|>")[0]
        return output_text


app = Flask(__name__)


class InferenceManager:
    def __init__(self):
        self.base_model = None
        self.ft_model = None
        self.models_loaded = False

    def load_models(self):
        try:
            print("Loading base model...")
            self.base_model = TinyLlamaDocStringInference(
                model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )

            print("Loading fine-tuned model...")
            if not os.path.exists("./tiny_llama_finetuned"):
                raise FileNotFoundError("Fine-tuned model directory not found.")
            self.ft_model = TinyLlamaDocStringInference(
                model_path="./tiny_llama_finetuned"
            )

            self.models_loaded = True
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Failed to load models: {e}")
            self.models_loaded = False

    def run_inference(self, model_obj, code, model_name):
        try:
            start = time.time()
            output = model_obj.submit_prompt(code)
            duration = int((time.time() - start) * 1000)
            return {
                'response': output,
                'time_ms': duration,
                'model': model_name
            }
        except Exception as e:
            return {
                'response': f"Error generating response: {str(e)}",
                'time_ms': 0,
                'model': model_name
            }


inference_manager = InferenceManager()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare', methods=['POST'])
def compare():
    if not inference_manager.models_loaded:
        return jsonify({'error': 'Models are still loading'}), 503

    data = request.get_json()
    code = data.get('code', '').strip()
    if not code:
        return jsonify({'error': 'No code provided'}), 400

    base_out = inference_manager.run_inference(
        inference_manager.base_model, code, "Original TinyLlama"
    )
    ft_out = inference_manager.run_inference(
        inference_manager.ft_model, code, "Fine-tuned TinyLlama"
    )

    return jsonify({'original': base_out, 'finetuned': ft_out})


@app.route('/status')
def status():
    return jsonify({'models_loaded': inference_manager.models_loaded})


@app.route('/test_models')
def test_models():
    if not inference_manager.models_loaded:
        return jsonify({'error': 'Models not loaded yet'}), 500

    code = "def get_user_by_id(user_id):\n    return database.query(f'SELECT * FROM users WHERE id = {user_id}')"

    base_out = inference_manager.run_inference(
        inference_manager.base_model, code, "Original TinyLlama"
    )
    ft_out = inference_manager.run_inference(
        inference_manager.ft_model, code, "Fine-tuned TinyLlama"
    )

    return jsonify({
        'original': base_out,
        'finetuned': ft_out
    })


def load_models_async():
    inference_manager.load_models()


if __name__ == '__main__':
    print("Starting background model load...")
    thread = threading.Thread(target=load_models_async)
    thread.daemon = True
    thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
