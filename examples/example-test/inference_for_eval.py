import argparse
import os
import jsonlines
import logging


def load_sdm():
    """
    load your SDM
    you can pass the necessary parameters as needed
    """
    pass


def respond(input_audio, output_path):
    """
    input_audio: the path of the input audio
    output_path: the path to write the output audio
    you can pass the necessary parameters as needed
    remember to return the textual response
    """
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # Set log
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # inference
    output_dir = args.output_dir
    output_audio_dir = os.path.join(args.output_dir, "audio")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir, exist_ok=True)
    pred_text = os.path.join(output_dir, "pred_text.jsonl")
    question_text = os.path.join(output_dir, "question_text.jsonl")
    gt_text = os.path.join(output_dir, "gt_text.jsonl")

    logging.info("<========loading model========>")
    sdm = load_sdm()

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
            output_path = os.path.join(output_audio_dir, f"{step:04d}.wav")
            response = respond(input_path, output_path)
            logging.info(f"Input text: {input_text}")
            logging.info(f"Output text: {response}")
            logging.info(f"Output audio saved to {output_audio_dir}/{step:04d}.wav")
            pt.write({str(step).zfill(4): response})
            qt.write({str(step).zfill(4): input_text})
            if isinstance(target_text, list):
                gt.write({str(step).zfill(4): " / ".join(target_text)})
            else:
                gt.write({str(step).zfill(4): target_text})


if __name__ == "__main__":
    main()
