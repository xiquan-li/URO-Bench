import jiwer
from tqdm import tqdm
from argparse import ArgumentParser
import logging
import os
import jsonlines
import string
import multiprocessing as mp
import numpy as np
import torch.nn.functional as F
from normalizers.english import EnglishTextNormalizer
from funasr import AutoModel
import cn2an


# get wer for en
def wer_en(s1, s2):
    english_normalizer = EnglishTextNormalizer()
    text1 = english_normalizer(s1.strip().lower())
    text2 = english_normalizer(s2.strip().lower())
    try:
        wer = jiwer.wer(text1, text2)
    except:
        wer = 1
    return wer


# get cer for zh
def wer_zh(s1, s2):
    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation

    truth = s1
    hypo = s2

    for x in punctuation_all:
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    text1 = cn2an.transform(truth.strip().lower(), "an2cn")
    text2 = cn2an.transform(hypo.strip().lower(), "an2cn")

    try:
        cer = jiwer.cer(text1, text2)
    except:
        cer = 1
    return cer


# get wer or cer
def get_wer(lang, s1, s2):
    if lang not in ["zh", "en"]:
        raise NotImplementedError("lang support only 'zh' and 'en' for now.")
    if lang == "en":
        return wer_en(s1, s2)
    elif lang == "zh":
        return wer_zh(s1, s2)


def set_emotion2vec():
    model_id = "iic/emotion2vec_plus_large"

    model = AutoModel(
        model=model_id,
        hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
    )

    return model


# get the prob that the audio contains the emotion
def infer_emotion(model, item):
    """
    Using the finetuned emotion recognization model

    rec_result contains {'feats', 'labels', 'scores'}
        extract_embedding=False: 9-class emotions with scores
        extract_embedding=True: 9-class emotions with scores, along with features

    9-class emotions:
    iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large (May. 2024 release)
    iic/emotion2vec_base_finetuned (Jan. 2024 release)
        0: angry
        1: disgusted
        2: fearful
        3: happy
        4: neutral
        5: other
        6: sad
        7: surprised
        8: unknown
    """

    emo2num = {
        "angry": 0,
        "disgusted": 1,
        "fearful": 2,
        "happy": 3,
        "neutral": 4,
        "other": 5,
        "sad": 6,
        "surprised": 7,
        "unknown": 8,
    }

    wav_file = item["audio"]
    if os.path.exists(wav_file):
        rec_result = model.generate(
            wav_file,
            output_dir="./outputs",
            granularity="utterance",
            extract_embedding=False,
        )
        return rec_result[0]["scores"][emo2num[item["emotion"]]]
    else:
        return 0


# eval for GenEmotion
def eval_ge(args):
    output_file = os.path.join(args.output_dir, "result_ge.jsonl")
    sum_score = 0
    model = set_emotion2vec()
    with open(args.question, "r") as f:
        length = sum([1 for _ in f])
    with open(args.answer, "r") as pt, open(args.reference, "r") as gt, open(
        args.dataset_path, "r"
    ) as dt, jsonlines.open(output_file, mode="w") as ot:
        for i, (answer, reference, source_data) in tqdm(
            enumerate(
                zip(jsonlines.Reader(pt), jsonlines.Reader(gt), jsonlines.Reader(dt))
            ),
            total=length,
        ):
            file_path = os.path.join(args.audio_dir, str(i).zfill(4) + ".wav")
            item = {
                "answer": answer[str(i).zfill(4)],
                "reference": reference[str(i).zfill(4)],
                "emotion": source_data["emotion"],
                "audio": file_path,
            }
            posibility_emo = infer_emotion(model, item)
            wer = min(get_wer(args.language, item["answer"], item["reference"]), 1)
            item["posibility_emo"] = posibility_emo
            item["WER"] = wer
            item["score"] = posibility_emo * (1 - wer) * 100
            sum_score += item["score"]
            ot.write(item)
        ot.write({"final_score": sum_score / length})
    # save results
    logging.info(f"saving result to {output_file}")
