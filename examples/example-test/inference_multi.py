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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_text = os.path.join(output_dir, "output_with_text.jsonl")

    logging.info("<========loading model========>")
    sdm = load_sdm()

    logging.info("<========inference starts========>")
    with open(args.dataset, "r") as f, jsonlines.open(output_text, mode="w") as ot:
        for data in jsonlines.Reader(f):
            dialogue = data["dialogue"]
            last_response = ""
            conversation = []
            id_dir = os.path.join(output_dir, str(data["id"]))
            os.makedirs(id_dir, exist_ok=True)
            for turn in dialogue:
                input_path = os.path.join(
                    os.path.dirname(args.dataset), turn["source_wav"]
                )
                output_path = os.path.join(id_dir, f"chat_{turn['round']}.wav")
                response = respond(input_path, output_path)
                last_response = response.strip()
                conversation.append(
                    {
                        "round": turn["round"],
                        "source_wav": turn["source_wav"],
                        "source_text": turn["source_text"],
                        "target_text": turn["target_text"],
                        "output_text": last_response,
                    }
                )

            output_data = {
                "id": data["id"],
                "num_round": len(dialogue),
                "dialogue": conversation,
            }
            logging.info(f"sample{data['id']} finished")
            ot.write(output_data)


if __name__ == "__main__":
    main()
