# Scalable Machine Learning and Deep Learning
# Lab 2 : Swedish Transcriber
[[Blog]](https://openai.com/blog/whisper)
[[Paper]](https://cdn.openai.com/papers/whisper.pdf)
[[Model card]](model-card.md)
[[Colab example]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

## Model: Whisper

Whisper is a general-purpose speech recognition model developped by OpenAi. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.

Its main idea is based on a trained transformer sequence-to-sequence modelon various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. All of these tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing for a single model to replace many different stages of a traditional speech processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.

## Available models and languages

There are five model sizes, four with English-only versions, offering speed and accuracy tradeoffs. Below are the names of the available models and their approximate memory requirements and relative speed. 


|  Size  | Parameters | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |      `large`       |    ~10 GB     |       1x       |

## Approach


We used the small Whisper model with 255M parameters and we followed the same steps as in the colab example.
First, we loaded the pre-trained FeatureExtractor and the pre-trained tokenizer and used it for fine-tuning without any further modifications. We have just set the target language to swedish and the task to transcription. These arguments inform the tokenizer to prefix the language and task tokens to the start of encoded label sequences.Then, we combined both into one processor to make it easier. 
Next, we created the DataCollector treating the data in an independent schema composed of inputs and labels, and we the WER metric to evaluate the performance of the fine-tuned model at the end.
Finaly, we loaded the pre-trained small whisper model from the openAi checkpoint.


## Model Fine-Tuning

