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


# eval asr-wer or asr-cer
def eval_wav(args):
    wers = []

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.language == "en":
        output_file = os.path.join(args.output_dir, "result_wer.jsonl")
    elif args.language == "zh":
        output_file = os.path.join(args.output_dir, "result_cer.jsonl")

    logging.info("<------start wer/cer eval------>")
    with open(args.answer, "r") as f:
        length = sum([1 for _ in f])

    if args.mode == "multi" or args.mode == "sa":
        with open(args.answer_text, "r") as gt, open(
            args.answer, "r"
        ) as pt, jsonlines.open(output_file, mode="w") as ot:
            for i, (text, asr) in tqdm(
                enumerate(zip(jsonlines.Reader(gt), jsonlines.Reader(pt))), total=length
            ):
                item = {"round": text["num_round"]}
                multi_wer = []
                for j in range(item["round"]):
                    tmp = {
                        "text": text["dialogue"][j]["output_text"],
                        "asr": asr["dialogue"][j]["output_text"],
                    }
                    wer = get_wer(args.language, tmp["text"], tmp["asr"])
                    multi_wer.append(wer)
                    item["text" + str(j)] = text["dialogue"][j]["output_text"]
                    item["asr" + str(j)] = asr["dialogue"][j]["output_text"]
                    item["WER" + str(j)] = wer
                avg_wer = np.mean(multi_wer)
                item["avg.WER"] = avg_wer
                wers.append(avg_wer)
                ot.write(item)
            wer = np.mean(wers)
            ot.write({"final_WER": wer})
    else:
        with open(args.answer_text, "r") as gt, open(
            args.answer, "r"
        ) as pt, jsonlines.open(output_file, mode="w") as ot:
            for i, (text, asr) in tqdm(
                enumerate(zip(jsonlines.Reader(gt), jsonlines.Reader(pt))), total=length
            ):
                item = {"text": text[str(i).zfill(4)], "asr": asr[str(i).zfill(4)]}
                wer = get_wer(args.language, item["text"], item["asr"])
                item["WER"] = wer
                wers.append(wer)
                ot.write(item)
            wer = np.mean(wers)
            ot.write({"final_WER": wer})

    logging.info(f"Total {len(wers)} samples")
    logging.info(f"WER: {round(wer * 100, 3)}%")
    logging.info(f"saving result to {output_file}")


# eval repeat or repeat-zh
def eval_repeat(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.language == "en":
        output_file = os.path.join(args.output_dir, "result_repeat_wer.jsonl")
    elif args.language == "zh":
        output_file = os.path.join(args.output_dir, "result_repeat_cer.jsonl")

    sum_wer = 0
    ok_num = 0
    with open(args.question, "r") as f:
        length = sum([1 for _ in f])
    with open(args.answer, "r") as pt, open(args.reference, "r") as gt, jsonlines.open(
        output_file, mode="w"
    ) as ot:
        for i, (answer, reference) in tqdm(
            enumerate(zip(jsonlines.Reader(pt), jsonlines.Reader(gt))), total=length
        ):
            item = {
                "answer": answer[str(i).zfill(4)],
                "reference": reference[str(i).zfill(4)],
            }
            wer = get_wer(args.language, item["answer"], item["reference"])
            item["WER"] = wer
            if wer <= 0.5:
                ok_num += 1
                sum_wer += item["WER"]
            ot.write(item)
        ok_rate = ok_num / length if length != 0 else 0
        final_WER_for_ok_case = sum_wer / ok_num if ok_num != 0 else 0
        ot.write({"ok_rate": ok_rate, "final_WER_for_ok_case": final_WER_for_ok_case})
    # save results
    logging.info(f"saving result to {output_file}")
