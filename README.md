# TSV


Source code for ICML 2025 paper [Steer LLM Latents for Hallucination Detection](https://arxiv.org/abs/2503.01917) by Seongheon Park, Xuefeng Du, Min-Hsuan Yeh, Haobo Wang, and Yixuan Li

---

## Requirements

```bash
conda env create -f tsv.yml
```
---

## LLM response generation

Generate responses for each question to construct an unlabeled QA dataset in the wild.

```bash
bash gen.sh
```

---

## GT generation

Generate [BLEURT](https://arxiv.org/abs/2004.04696) score for each QA pair


```bash
bash gt.sh
```

---

## Train TSV

Train TSV for hallucination detection.

```bash
bash train.sh
```

---

## Citation

```
@inproceedings{
park2025steer,
title={Steer {LLM} Latents for Hallucination Detection},
author={Seongheon Park and Xuefeng Du and Min-Hsuan Yeh and Haobo Wang and Yixuan Li},
booktitle={Forty-second International Conference on Machine Learning},
year={2025}
}
```

---

## Acknowledgement

We gratefully acknowledge [HaloScope](https://arxiv.org/abs/2409.17504), [ITI](https://arxiv.org/abs/2306.03341), and [ICV](https://arxiv.org/abs/2311.06668) for their inspiring ideas and open-source contributions, which served as valuable foundations for this work.
