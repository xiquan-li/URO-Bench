import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import transformers
from openai import OpenAI
import os
import jsonlines
import logging


class NaiveAssistant:
    def __init__(self, whisper_path):
        self.whisper_path = whisper_path
        self.asr = self.load_asr()
        self.llm = self.load_llm()

    def load_asr(self):
        model_id = self.whisper_path
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir="./cache",
        )
        model.to("cuda:0")
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda:0",
        )
        return pipe

    def load_llm(self):
        # set your api key
        client = OpenAI()
        return client

    def generate_audio(self, audio):
        transcript = self.asr(audio, generate_kwargs={})["text"].strip()
        completion = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": transcript},
            ],
        )
        response = completion.choices[0].message.content
        return response

    def generate_text(self, text):
        completion = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
            ],
        )
        response = completion.choices[0].message.content
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["audio", "text"],
        help="Input modality: 'audio' or 'text'",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--whisper_path", type=str, required=True)
    args = parser.parse_args()

    # Set log
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # inference
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    pred_text = os.path.join(output_dir, "pred_text.jsonl")
    question_text = os.path.join(output_dir, "question_text.jsonl")
    gt_text = os.path.join(output_dir, "gt_text.jsonl")

    logging.info("<========loading model========>")
    assistant = NaiveAssistant(args.whisper_path)

    logging.info("<========inference starts========>")
    with open(args.dataset, "r") as f, jsonlines.open(
        pred_text, mode="w"
    ) as pt, jsonlines.open(question_text, mode="w") as qt, jsonlines.open(
        gt_text, mode="w"
    ) as gt:
        for step, item in enumerate(jsonlines.Reader(f)):
            input_path = os.path.join(os.path.dirname(args.dataset), item["source_wav"])
            input_text = item["source_text"]
            if "target_text" in item:
                target_text = item["target_text"]
            else:
                target_text = item["source_text"]
            if args.modality == "audio":
                # audio = torchaudio.load(input_path)[0].squeeze().numpy()
                response = assistant.generate_audio(input_path)
                logging.info(f"Input text: {input_text}")
                logging.info(f"Output text: {response}")
                pt.write({str(step).zfill(4): response})
                qt.write({str(step).zfill(4): input_text})
                if isinstance(target_text, list):
                    gt.write({str(step).zfill(4): " / ".join(target_text)})
                else:
                    gt.write({str(step).zfill(4): target_text})
            else:
                text = input_text.strip()
                response = assistant.generate_text(text)
                logging.info(f"Input text: {text}")
                logging.info(f"Output text: {response}")
                pt.write({str(step).zfill(4): response})
                qt.write({str(step).zfill(4): text})
                if isinstance(target_text, list):
                    gt.write({str(step).zfill(4): " / ".join(target_text)})
                else:
                    gt.write({str(step).zfill(4): target_text})


if __name__ == "__main__":
    main()

# python inference_lm.py --input input.wav --modality audio
# python inference_lm.py --input input.txt --modality text
