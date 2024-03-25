# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the data filtering mechanism of Whisper."""

from concurrent import futures
import logging

import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, pipeline
from transformers.generation.configuration_utils import GenerationConfig
import tokenizers

import cProfile
import pstats

import os
import math
import re
import time
import editdistance

from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.audio import decode_audio, pad_or_trim

import argparse
import copy
import json

from tqdm import tqdm

from sequence_matching import find_best_match_segment

import numpy as np
import ctranslate2

CHUNK_SIZE = 3000
SECONDS_PER_CHUNK = 30
FRAMES_PER_SECONDS = 1000
SAMPLING_RATE = 16000
TOKEN_MARGIN = 10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_lists', type=str, default=None)
    return parser.parse_args()

def calculate_cer(ref, hyp):
    cer = editdistance.eval(ref.replace(" ", ""), hyp.replace(" ", "")) / len(ref)
    return cer

def remove_first_match(original_string, substring):
    pattern = re.escape(substring)    
    return re.sub(pattern, '', original_string, count=1)

def combine_texts_and_timestamps(text, timestamps_start, timestamps_end):
    start = "<|{:.2f}|>".format(timestamps_start)
    end = "<|{:.2f}|>".format(timestamps_end)
    return " ".join([start, text, end])

class fast_pipeline():
    def __init__(self, model):
        self.model = model
        self.epd_margin = 0.1

    def __call__(self, filepath, ref, audio_duration, generate_configs):
        if "condition_on_prev_tokens" in generate_configs:
            generate_configs["condition_on_previous_text"] = generate_configs.pop("condition_on_prev_tokens")
        if "logprob_threshold" in generate_configs:
            generate_configs["log_prob_threshold"] = generate_configs.pop("logprob_threshold")

        # Get segment info
        segments, info = self.model.transcribe(filepath, beam_size=5, without_timestamps=False, **generate_configs)
        
        if info.language != "ko":
            filter_type = "lang_detect"
            return None, None, None, None, None, None, None, filter_type

        texts = []
        starts = []
        ends = []
        tokens = []
        encoder_outputs = []
        segment_sizes = []
        for segment in segments:
            if float(segment.start) >= float(audio_duration) - self.epd_margin:
                break
            texts.append(segment.text)
            starts.append(segment.start)
            ends.append(segment.end)
            tokens.append(segment.tokens)
            encoder_outputs.append(segment.encoder_output)
            segment_sizes.append(segment.segment_size)
        
        hyp = "".join(texts)
        if len(hyp) == 0:
            filter_type = "low_cer"
        elif len(ref) == 0:
            filter_type = "empty_trans"
        else:
            cer = calculate_cer(hyp, ref)
            if cer > 0.3:
                filter_type = "low_cer"
            else:
                filter_type = None

        return starts, ends, texts, tokens, encoder_outputs, segment_sizes, cer, filter_type

class ASR():
    def __init__(self, pipe_type='fast-pipe'):
        # Initialize and load the model and pipeline here, so it's done only once
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32

        tokenizer_file = "tokenizer.json"
        self.model = WhisperModel('large-v3', device="cuda", compute_type="float16")
        self.tokenizer = Tokenizer(
            tokenizers.Tokenizer.from_file(tokenizer_file),
            True,
            task="transcribe",
            language="ko",
        )
        
        self.pipe_type = pipe_type
        self.pipe = fast_pipeline(model=self.model)

    def split_audio_and_text(self, original_audio_path, timestamped_texts):
        """Splits the audio and its timestamped text into segments.

        Args:
            original_audio_path (str): Path to the original audio file.
            timestamped_texts (list): List of timestamped texts for each audio segment.

        Returns:
            list: A list of tuples, each containing the path to the split audio file and its corresponding text.
        """
        splits = []
        waveform, sample_rate = torchaudio.load(original_audio_path)

        prev_end_sec = 0.00
        for i, timestamped_text in enumerate(timestamped_texts):
            # Extract start and end times from the timestamped text
            times = [float(time[2:-2]) for time in re.findall(r"<\|\d+\.\d+\|>", timestamped_text)]
            start_sec = times[0] + prev_end_sec
            end_sec = times[-1] + prev_end_sec

            # Calculate start and end samples
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)

            # Split waveform
            split_waveform = waveform[:, start_sample:end_sample]

            # Define new audio path
            split_audio_path = original_audio_path.split('.wav')[0] + f'-{i}.wav'

            # Save split audio
            torchaudio.save(split_audio_path, split_waveform, sample_rate)
            
            # Add split audio path and text to splits
            splits.append((split_audio_path, timestamped_text))

            prev_end_sec += end_sec

        return splits

    def transcribe_speech(self, filepath, json_dump_file, filter_dump_file):
        temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        no_speech_threshold = 0.55
        logprob_threshold = -0.3

        generate_config = {"task": "transcribe"}
        generate_config["max_new_tokens"] = 256
        generate_config["condition_on_prev_tokens"] = False
        generate_config["no_speech_threshold"] = no_speech_threshold
        generate_config["temperature"] = temperature
        generate_config["logprob_threshold"] = logprob_threshold
        generate_config["return_encoder_output"] = True
        generate_config["return_segment_size"] = True
        
        # with open(filepath, 'r') as f:
        #     contents = json.load(f)

        contents = ['a']
        new_contents = []
        filtered_contents = []
        for content in tqdm(contents):
            # t = "이 문장 안에 사랑이 정의돼 있어요? 정리되지 않아요 전혀 정의되어 있지 않죠. 조용히 계시다가 사랑 이야기가 나오니까 그렇죠. 그런 그럴 수 있어요. 사랑 그런 전제가 있죠. 그런데 이 문장 형식은 문장 형식은 정의 내려지는 형식이 아니죠."
            # content = {'audio': "/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech/Training/D12/G02/S001024/000143.wav", 'text': t}
            t = "그렇겠죠. 왜그러냐 하면은 그 공자가 극복하려고 하는 이 방향이 노자가 봤을 때는 문제점이 있잖아요."
            content = {'audio': "/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech/Training/D12/G02/S001024/000022.wav", 'text': t}
            # t = "만들어 논 건 사실이었어요. 그다음에 와 조선인들아. 너희들 민족의 언어로 한글로 신문을 창간할수 있게끔 해줄게. 와 출판의 자유, 언론의 자유 주네? 실제로 조선일보 동아일보가 이 당시에 창간될 수 있었어요. 이것도 맞아요 그리고 조선인들과 일본인을 차별하지 않고 교육하겠다며 조선 교육령을 1922년에 2차로 다시 발표합니다. 1차가 10년대였으면 이때는 2차인데 여기에 핵심 내용은 조선인과 일본인 학교도 똑같이 6년 다니세요."
            # content = {'audio': "/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech/Training/D05/H01/S000134/000196.wav", 'text': t}
            # t = "떠나는 길에 니가 내게 말했지 너는 바라는 게 너무나 많아 잠깐이라도 널 안 바라보면 머리에 불이 나버린다니까 나는 흐르려는 눈물을 참고 하려던 얘길 어렵게 누르고 그래 미안해 라는 한 마디로 너랑 나눈 날들 마무리했었지 달디달고 달디달고 달디단 밤양갱 밤양갱 내가 먹고 싶었던 건 달디단 밤양갱 밤양갱이야 떠나는 길에 니가 내게 말했지 너는 바라는 게 너무나 많아 아냐 내가 늘 바란 건 하나야 한 개뿐이야 달디단 밤양갱 달디달고 달디달고 달디단 밤양갱 밤양갱 내가 먹고 싶었던 건 달디단 밤양갱 밤양갱이야 상다리가 부러지고 둘이서 먹다 하나가 쓰러져버려도 나라는 사람을 몰랐던 넌 떠나가다가 돌아서서 말했지 너는 바라는 게 너무나 많아 아냐 내가 늘 바란 건 하나야 한 개뿐이야 달디단 밤양갱"
            # content = {'audio': "data/byg.mp3", 'text': t}
            # t = "꼬맹아 집사가 나타나기만 하면 무조건 도망치는 고양이가 있다? 남 보다 못하지 않냐 그리고 쟤 저러다가 일나면 어떡하냐 꼬맹아 사람과 눈도 마주치기 싫다 사람의 손길은 더 싫다 어떻게 든 다가가 보려는 집사와 무조건 피하는 냥이 숨막히는 숨바꼭질이 벌써 사 년째년째 여기가 감옥이라고 생각안 하고 하지만 녀석은 점점 자신의 동굴 속으로 들어가 버린다는데 집사들도 가끔 우리 고양이가 맞나 의심이 든다 요 녀석 과연 무슨 사연이 있는 걸까? 봄꽃이 한창이지만 마음 한켠은 아직 겨울이라는 이댁 집사님들 저희도 반갑습니다. 우리 집에는 냥이가 여섯 마리가마리가 있고요. 큰 아이가 나봉이 오 이 집에 맏형 둘째가 연지 부끄러움이 많은 친구래요. 세 번째가 꼬질이 제 딴에는 꼭꼭 숨은 거라구 그다음에 이부 자칭 타칭 순둥이 마지막으로 내가 본 세리 한 마리는 여기 옷장에 있어요 여기 낯선 제작진의 방문이 불편했던 걸까요 조심스럽게 옷장 문을 열어보는데 자자 다시 한번 볼까요? 오호라 요 녀석이 오늘의 문제냥 꼬맹이랍니다 얼굴 보기 참 힘드네요. 지금 한 사 년이년이 넘은 것 같은데 얘를 안아본 횟수가 손에 꼽을 만큼밖에 안되고 손톱도 못 깎아주고 세상에 집사들도 사 년년 동안 거의 볼 수가 없었다는데요 옷장 문을 여는 앞발 기술 보셨나요? 아니 캄캄한 옷장을 스스로 열고 들어가 왜 안나오는지 집사들도 답답할 뿐이랍니다. 꼬맹이는 옷장 속에 들어가서 나오지를 않아요. 아예 눈만 마주쳐도 피하고 꼬맹이는 이댁에 다른 양이들과 차원이 다르답니다. 집사들과 함께 뛰어 놀고 살 부비며 사는 게 집냥이들의 행복이건만 꼬맹이는 이십사 시간시간 늘 경계를 늦추지 않고 옷장 밖으로는 나올 생각이 없다고 합니다. 집사도 피하는 마당에 낯선 제작진을 반가워할 턱이 없겠죠 꼬맹이의 일상을 관찰해 보기로 했습니다. 노릇노릇 식빵을 보며 쉬고 있는 녀석 그런데 그 순간 집사의 움직임 소리가 들리자 한껏 몸을 낮추며 긴장하는 꼬맹이 오 이번엔 친구가 놀러왔네요. 누구니 꼬질이 냐옹 사람만큼 경계하진 않지만 나올 생각은 없는 듯 그런데 잠시 후 맛나게 먹는 소리가 들립니다. 아이구 꼬맹이 입맛을 다시는데요 꼬맹아 나가서 밥 먹어 옷장을 차지하고 들어와 앉은 꼬맹이 때문에 집사들도 불편한 점이 한두 가지가가지가 아닙니다. 이게 진짜 많이 때진 거예요 여기 요 가죽 자켓도 여기 보면은 요런데도 다 뜯어놨어요. 지금 여섯 번은번은 넘게버린 거 같아요 요기만 때가 타 있어요. 애 손 때고 꼬맹이를 옷장에서 나오게 하기 위해 집사들 정말 안 해본 게 없다고 합니다. 억지로 걔가 스트레스 너무 많이 받을 정도로 잡아 가지고 한번씩 안아보는데 그것도 방법이 아닌 것 같 더라고요. 고양이가 좋아하는 음악도 틀어도 보고 캣잎도 막 그 주변을 다 뿌려주고 해도 어 아무것도 안 해요. 집사들 출근시간 엄마 집사 출근 후 딸 집사도 급히 출근 준비를 합니다. 그런데 일 분 일 초가분 초가 급한 이 시간에 엥 뭐하시는 거예요 바닥에 간식을 짜기 시작 하는데요 설마 꼬맹이를 위해서 그렇게 짜시는 이유가 있어요? 나오다가 먹기라도 하라고 맛있는 간식길만 걸어라는 집사의 마음인데요 혹시나 배고프면 꼬맹이가 나와서 먹지 않을까 싶어 사료도 넉넉히 부어둡니다. 드디어 집사들이 모두 출근하고 집에는 고양이들 뿐 꼬맹이 반응이 궁금해지는데요. 오호 옷장 밖으로 한 발발 한 발발 내딛습니다 나올 때도 조심조심 주변을 살펴보고 나오는 꼬맹이 꼬맹아 누나가 너 먹으라고 놓은거야 맛 좀 봐봐 하지만 매사 늘 경계하고 조심하는 꼬맹이 녀석 그냥 지나쳐 버리네요. 집사님 서운하겠다 했는데 기다렸다는 듯 세리가 나타나 진공청소기로 흡입하듯 깨끗이 간식을 모두 먹어치우네요. 그렇게 집사가 없는 집에서 꼬맹이는 따사로운 햇볕 아래 제 세상이라도 만난 듯 일광욕을 즐기더니 좌로 딩굴 우로 딩굴 편안한 시간을 보내네요. 그러고는 어딜 가나 했더니 화장실로 직행 긴 시간 동안 옷장에 있었으니 볼 일도 당연히 봐야겠죠. 쉬이 얼마나 참았던 거니 소변 오래 참으면 병 되는데 걱정이다 야 비록 옷장에 살고 있지만 화장실 에티켓은 잊지 않았네요. 거실을 지나 녀석의 다음 목적지는 역시나 밥입니다. 주변을 두리번 거리며 살펴보더니 다시 한번 아무도 없는 걸 확인하고 나서야 먹기 시작하는데요. 꼬맹아 배고팠지 아이고 괜찮아 천천히 먹어 그렇게 볼일 다 보고 집사들이 퇴근하지도 않았는데 옷장으로 들어가는 녀석 이 공간이 얼마나 편했으면 여기서 그루밍을 하네요. 사람이 없어도 옷장 밖은 불안한 모양입니다 배도 부르고 긴장도 풀리고 녀석 그대로 긴 잠에 빠집니다. 어둠이 내리고 밤이 되서야 집에 불이 켜집니다. 꼬맹아 꼬맹이 진짜 없는데 있어야 할 옷장에 꼬맹이가 없다는데요 녀석이 감쪽같이 사라졌습니다. 꼬맹이는 와서 찾아야 되니까 네 확인은 해요 어디에 있나 보통 이제 퇴근해 오시면은 항상 꼬맹이는 이런식으로 열어보시고 찾으 대체 어디로 사라진 건지 집안 구석구석을 살펴보지만 할머니 장롱에 봤어 엄마? 없는데? 아니 항상 웃장에만 있던 녀석이 대체 어디로 사라진 걸까요? 어디 있어요? 그런데 녀석이 관찰 카메라에 포착됐습니다 시간은 바야흐로 삼십 분분 전 집사의 갑작스러운 퇴근에 녀석이 미처 옷장으로 가지 못하고 엄마 집사 방으로 숨어버린 것 아니 숨을만한 곳이 아니라 집안 전체를 다 뒤졌잖아 없으면은 집에 없는 거잖아 이제 남은 곳은 단 하나 베란다뿐입니다. 어 미칠 것 같애 꼬맹아 여기 안쪽이요 보여요 여기가 맞았네 얼굴만 숨기면다니 집사들이 발견하자 부리나케 도망가는 녀석 어디로 갔어 또 어 그럼 됐어 하지만 세상에나 나와 너 진짜 꼬맹아 저 비좁은 틈에 어떻게 들어간 건지 그런데 꼬맹아 거기 숨어 있는 거 다 보여 얼른 나와 아니 아니 이해는 하지 말구 어찌나 잽싸게 숨는지 매번 집사를 피하는 꼬맹이 녀석과의 인연은 어떻게 시작된 걸까요? 저희가 이제 가게에 건너편 쪽에 실외기 뒤에서 구조를 하게 됐는데 옥상에서 떨어져 실외기와 벽 사이에 끼어서 어미도 없이 이틀을 울었던 꼬맹이 구조에 트라우마였던 것 같애요 구조 걔는 엄마랑 계속 있고 싶어 했던 것 같은데 약간 우리를 엄마랑 떼어내는 나쁜 사람으로 생각했던 거 같애 그래서 데꼬 왔을 때도 제가 처음에 진정을 시키려고 계속 안고 이랬는데 그때도 항상 어미 찾듯이 계속 울더라고요. 그러다가 혼자 마음의 문을 닫아버린 거같애요. 어둠이 내리고 굳게 닫혀 있던 옷장에서 슬그머니 나오는 녀석 집사가 잠이 들자 조용히 움직이기 시작합니다. 이렇게 목이 마를 땐 실컷 마시며 좋으련만 마음이 놓이지 않는지 작은 움직임에도 황급히 몸을 숨기기 바쁜 꼬맹이 입양했을 때부터 마음을 닫은 꼬맹이는 처음 이댁에 왔을 땐 침대 밑이나 가구 밑에 들어가 있었고 육 개월이개월이 지난 후 자신에게 조금 힘이 생기자 옷장 안으로 들어온 것이라는데요. 지금까지 사 년년 앞으로 얼마나 더 옷장에서 지내지 모르는 상황 시원한 비가 한바탕 쏟아진 아침 눈 뜨자마자 집사들이 냥이 밥에 약을 타기 시작합니다. 맘마 먹자 배 아파 어 그래 근데 어디가 아픈건가요? 최근에 고양이들이 감기에 걸린 적이 있었어요 근데 그게 고양이들이 여섯 마리니까마리니까 전부 다 이렇게 옮게 되는 거예요. 꼬맹이가 계속 기침을 하는 상황이고 약을 맥이거나 병원 데꼬 가고 싶은데 잡히질 않으니까 너무 걱정이 되는 거예요 애들이나 하루가 다르게 좀 수척해져 가고 꼬맹이도 예외는 아니랍니다 옷장 생활로 얼굴도 제대로 못 보니 약을 먹일 엄두도 내지 못한다는데요. 미야옹철 쌤께 꼬맹이의 상황을 미리 보여드렸습니다. 문제는 보호자랑 대면하는 순간에 무조건 이 상황을 회피해 버릴려고 하기 때문에 조금 더 안정적인 큰 옷장 공간 이면서도 보호자랑 대면해서 계속해서 교육을 할 수 있는 좀 한정된 구역이 필요합니다. 근데 이 집에서는 바로 베란다가 그런 공간입니다. 집사들이 꼬맹이를 위해 두 팔 걷고 나섰는데요 옷장인듯 옷장 아닌 옷장 같은 꼬맹이만의 공간이 완성됐습니다. 꼬맹아 이게 얘네들 집이에요. 오늘은 절친 꼬질이가 놀러 와 있네요 옷장 생활은 이제 그만 집사님 꼬질이 먼저 꺼내는데요 하지만 예민한 꼬맹이는 벌써 사태 파악 끝 한바탕 소동이 벌어졌습니다. 꼬맹이를 꺼내기 위해 온갖 방법을 동원했지만 됐다 사람에 대한 경계가 얼마나 심한지 바로 쫓아갔지만 이미 자취를 감춰버렸는데요 어디에도 녀석은 보이지 않습니다. 여기 공간으로 가는 거 말고는 없는 것 같애 싱크대 하부장 공간을 해체하는데요. 꼬맹이 여기있어 빨리 와 봐 어디 어디있나요? 아 안 돼 안 돼 안 돼 절로 도망갔어 저쪽으로 갔어 저쪽으로 아이구 세상에 여기서 이쪽에 지금 빼 갖고 얘를 들어 나올수 있게끔 해야 돼 이것 좀 빨리 도와줘 빨리 본격적으로 싱크대를 해체하는 모녀 드디어 하부장이 열리고 꼬맹이를 찾았습니다. 하지만 가만히 있을 녀석이 아니죠 이번엔 침대로 숨어든 녀석을 유인해 보는데요. 어 하다가 꼬맹이가 스스로 베란다에 들어가 줬습니다 두 시간여의 사투끝에 성공하는데요. 꼬맹아 어이구 어이구 꼬맹이 그 갖고 놀아 봐 안식처라고 생각이 들 수 있을 정도만 였으면 좋겠어요. 여기가 감옥이라고 생각안하고 과연 꽁꽁 닫힌 꼬맹이의 마음을 열 수 있을까요 냥이들의 슈퍼맨 가족들의 오랜 숙제를 풀기 위해 한달음에 달려와 주셨는데요. 선생님 오늘도 잘 부탁드립니다. 이쪽에다가 꼬맹이 전용 공간을 만들어줬거든요. 일단은 본인이 좀 혼자 생활할 만한 그래도 공간배치가 나오네요. 아 다행이에요. 하지만 여전히 불안해 보이는 꼬맹이 녀석을 위해 선생님이 준비한 것이 있는데데요. 일단 지금 꼬맹이가 지금 사람 자체를 되게 대면 하는 것조차도 지금 힘들어하는 상황이고 그 기간이 너무 길어요 사 년년 동안 그랬기 때문에 일단은 지금 꼬맹이의 이런 트라우마적인 공포심을 조금 낮춰 주기 위해서 항불안제를 조금 맛있는 간식이랑 섞어서 한번 시도를 해 보구요. 먼저 안정을 찾아주는 것이 급선무 아니나다를까 안절부절 도망치 기 바쁜데요. 먹는 거야 침착하게 꼬맹이에게 다가서다가 아무 말 없이 조용히 나옵니다. 근데 제가 느끼기에는 전보다는 조금 많이 빠졌어요. 사실 그렇게 마르지 않았어요 정상 체형이어서 그리고 아예 지금 기본 건강 상태에 대해서 크게 걱정할만한 어떤 요인이 없기는 하거든요. 근데 쟤 그 접종은 애기 때 애기 때는 좀 사실 잡기가 조금 더 편 했잖아요 그때 한 번만번만 하고 그때 예방접종을 하고 지금까지 한 번도번도 안 했고 근데 걱정이 되신 거에 대해서 아이가 치료를 받을 수 있는 치료를 받아들일 수 있는 상황일까요? 병원 가는 건 시기상조라는 거 아니죠. 아니죠 그러면 지금 상황에서는 치료를 받아들일 수 있는 상황까지 갔는게 우선이죠. 사 년년 동안 옷장 속에서 숨어 살았던만큼 지금부터 서서히 집사와의 마음의 거리를 좁히는게 우선이라는데요. 완전 동굴이잖아요 자기가 안 돼 아마 이제 이런 경우에는 어렸을때 트라우마 상황까지 같이 겹치는 경우가 많아요. 정말 안 나올려고 하는 거를 끄집어 냈거든요 왜냐하면 거기서 이틀 동안 울고 있었던 걸로 알았어요. 고양이 입장에서 이게 나를 찔러 죽이려고 하는 건지 아니면 나를 여기서 꺼내줄려고 하는 건지 그걸 판단 할 수는 없거든요. 괜히 사람의 심정으로만 안타까우니까 근데 구조를하는 건 맞기는 한데 그 상황을 그 고양이는 엄청나게 큰 외상후 증후군처럼 그 상황을 그렇게 인지할 수밖에 없는 거예요. 그날의 사건으로 사람에게서 더 멀어지게 된 겁니다. 생사의 기로에서 느꼈던 그 극한의 공포심은 우리는 살면서 한 번도번도 겪어보지 못한 공포심 어느 정도 이제 거리를 두고 조금씩 먹을 걸 던져준다거나 이런 것들이 돼야 되는데 이제 그 부분 없이 시간이 그냥 쭉 집 안에 길고양이처럼 생활하는 거예요. 그니까요. 그래서 그게 진짜 그래서 안타까운 거 같애요. 결국 꼬맹이는 사람의 손길을 피해 고립된 공간으로 숨어버린 건데요. 평소 꼬맹이의 모습을 좀 더 살펴보기로 했습니다 집사님들 뭘 보고 그렇게 놀라신 거예요? 집사들이 없을 때 밥 먹고 볼일 정도 보겠지 했는데 숨기 바빴던 녀석이 여유롭게 뒹굴뒹굴 할 줄도 안다는 게 신기하다는 것 얘가 편안한 걸 처음 보니까 이렇게 보는 게 거의 없어요 아예 처음 봐요. 더 재밌는건 관찰 카메라에 잡힌 꼬맹이 얼굴에 대한 반응 근데 사진 이렇게 보니까 우리 집에서 제일 미남이야 원래 맨날 이렇게 도망가는 모습에서 얼굴을 한 번도번도 본 적이 없는데 세상에 꼬맹이가 그루밍하는 것도 사 년만에년만 에 처음 본다구 그러면 집사님들 이 장면 보시면 더욱 깜짝 놀라시겟는걸요? 어머 너한테 올라왔어? 어 내 침대에 오 설마 여기까지 온 거야 제가 잘못 진짜 많이 했어요. 무슨 잘못을요? 너 키워주고 너무 이러냐면서 막 혼내기도 하고 갖다 버린다 그랬거든요. 아니 그 말을 해도 얘는 그럴 마음이 없는대도 이제 너무 애가 곁을 안 주니까 근데 지금은 그나마 약간의 희망이 보이는 거는 보호자분이 자고 있을때 이불 밑을 조금 바꿔주는 정도까지 왔다는 것도 희망이 사실은 많이 뭔가 용기 낸 거거든 꼬맹이와 가까워지기 위해서는요 평소 갑자기 튀어나오는 녀석 때문에 집사님들 놀랄 일이 참 많았죠. 가능하면 꼬맹이 근처에 갔을 때는 최대한 좀 의연하게 큰 제스처 없이 반응하는 것에 대해서 좀 연습을 하셔야 돼요. 안그러면 관계가 오히려 조금 생기다가도 깨질 수 있거든요. 꼬맹이는 만약 자기가 정말 궁지에 몰리면 저를 공격할 것 같은 생각이 저도 모르게 비명을 꼬맹이를 직접 손으로 만지지 않는 이상 그런 일은 생기지 않을 거야 이제는 침착하게 꼬맹이와 마주하기로 약속해요 마지막으로 꼬맹이의 마음의 문을 여는 것만 남았습니다. 거리를 한 일 점 오 미터. 정도 유지하고 그 상황에서 눈 인사 한 번번 이름 불러주고 간식 코앞에다가 이렇게 놔두고 그냥 돌아서 나오는 거예요. 그 사람이 너한테 이걸 주고 가 이것만 알려주는 거거든요. 절대 함부로 만져서도 안되고 먹으라고 강요해서도 안 된다는 것 조금 더 가까이 가셔도 괜찮아요 지금 캣타워 있는 그 위치까지 한 걸음만걸음만 더 눈 인사를 가볍게 하시고 이름 불러주고 고 앞에다가 간신히 내려놓고 돌아서서 나올게요. 미안해 미안해 누나가 뭐 하는 거 아니고 간식 주러 온 거야 누나 갈게 아이구 꼬맹이 녀석 비록 얼음이 됐지만 숨거 도망치지 않았다는 게 포인트 오늘 조금 길었어요. 아 더 짧게요? 더 짧게 처음 시작은 정말 눈 인사하고 꼬맹아 간식이야 정말 노출 시간은 짧고 보상만 딱 두고 오는 거예요. 일단 중요한 건 매일 하루에 네 번번 네 번에번에 나눠서 먹을 것 사료를 제한 급식을 조금씩 가져다주고 이게 그냥 가장 기본 시작이에요. 순화 과정이 얼마나 걸릴지 꼬맹이 같은 경우에는 오래 걸릴 거예요 지금 피해서 다닌 시간이 이미 사 년년 가까이 흘렀잖아요. 사람으로 치면은 뭐 십 년년 이상의 시간을 계속해서 그 공포심 때문에 숨어다니던 친구인데 이게 뭐 하루아침에 일 주일주일 안에 훨씬 더 갑자기 좋아진다 이거는 사실 불가능에 가깝고 하지만 중요한 거는 그 시간 동안에 조금이라도 더 가까워지는게 사실은 중요한 거거든 마지막 꿀팁 대 방출 여기를 좀 더 넓은 옷장 공간처럼 느끼게 해주면 돼요 이런 시각적인 자극 부분도 지금 차단을 해 줄 거고 나중에는 저희가 커튼을 이렇게 살짝 열어서 이렇게 봐도 피하지 않고 편안하게 먹을 거 먹고 있는 그때는 좀 더 사람하고 유대감이 쌓였다라는 증거로 보죠. 멀리 돌아왔지만 이제 길을 알았으니 지금도 늦지 않았답니다. 구조했다는 일 차적인차적인 생각뿐이었고 오히려 더 집 안에서 길냥이로 만들었다는게 그냥 미안한 마음이 제일 큰 것 같애요. 여기가 이렇게 안식처가 돼서 그냥 정말 편안하게 행복하게 살았으면 좋겠어요. 꼬맹아 여기도 불편하지는 않지 어둡고 외로웠던 혼자만의 생활과 이젠 작별하고 앞으로 집사님들과 꽃길만 걷자 꼬맹이의 묘생 이 막막 집사님들 잘 부탁드립니다."
            # content = {'audio': "/home/ubuntu/Workspace/DB/korean_db/workspace/faster-whisper/tests/data/SLAAO21000001.wav", 'text': t}

            audio = decode_audio(content['audio'], sampling_rate=SAMPLING_RATE)
            audio_duration = audio.shape[0] / SAMPLING_RATE

            # transcribe audio & get segment ranges
            starts, ends, texts, tokens, encoder_outputs, segment_sizes, cer, filter_type = self.pipe(
                content['audio'],
                content['text'],
                audio_duration,
                generate_config
            )

            # Data filtering
            if filter_type is not None:
                filtered_content = copy.deepcopy(content)
                filtered_content['filter_type'] = filter_type
                filtered_contents.append(filtered_content)
                continue
            
            # set json dump variable
            new_content = copy.deepcopy(content)

            # Generate timestamps-added texts
            # There are 3 sizes of segments: chunk (30s), vad (utterence segment), word
            # chunk > vad > word
            timestamped_texts = []
            segmented_texts = []
            chunk_idx = -1
            chunk_onset_gap = 0.00
            ref_tokens = self.tokenizer.encode(t)
            for num_vad_processed, (timestamps_start, timestamps_end, enc_output, segment_size, token) in enumerate(zip(starts, ends, encoder_outputs, segment_sizes, tokens)):
                # Filter wierd timestamped audio files (alignment early end)
                if len(t) == 0:
                    timestamped_text = content['text']
                    break

                if timestamps_end // SECONDS_PER_CHUNK > chunk_idx:
                    chunk_onset_gap = 0.00
                    chunk_idx += 1

                # Get alignment path
                ref_tokens = ref_tokens[:len(token) + TOKEN_MARGIN]
                alignments = self.pipe.model.find_alignment(self.tokenizer, ref_tokens, enc_output, segment_size)

                timestamps_start, timestamps_end = timestamps_start % (SECONDS_PER_CHUNK), timestamps_end % (SECONDS_PER_CHUNK)
                timestamps_start, timestamps_end = timestamps_start + chunk_onset_gap, timestamps_end + chunk_onset_gap

                if timestamps_start > timestamps_end:
                    chunk_onset_gap = SECONDS_PER_CHUNK - timestamps_start
                    timestamps_start = 0.00
                    timestamps_end += chunk_onset_gap

                # iterate through the speech-words alignment
                prev_end_margin = SECONDS_PER_CHUNK
                words = []
                for num_word_processed, alignment in enumerate(alignments):
                    word_end = alignment['end']
                    end_margin = math.fabs(timestamps_end - word_end)
                    if prev_end_margin < end_margin or num_word_processed == len(alignments) - 1:
                        if num_word_processed == len(alignments) - 1:
                            words.append(alignment['word'])

                        # form a new text segment with timestamps
                        text = "".join(words).strip(" ")
                        timestamped_texts.append(combine_texts_and_timestamps(text, timestamps_start, timestamps_end))
                        segmented_texts.append(text)

                        # set data for next vad segment determination
                        t = remove_first_match(t, text).strip(" ")

                        ref_tokens = self.tokenizer.encode(t)
                        alignments = alignments[num_word_processed:]
                        words = []
                        break
                    else:
                        prev_end_margin = end_margin
                        words.append(alignment['word'])

            # Filter wierd timestamped audio files (alignment cut)
            segmented_text = " ".join(segmented_texts)
            if segmented_text != content['text']:
                timestamped_text = content['text']
            else:
                timestamped_text = " ".join(timestamped_texts)

            # Perform audio segmentation
            timestamped_texts = timestamped_text.split("<|0.00|>")
            timestamped_texts = [text.strip() for text in timestamped_texts if text.strip() != ""]
            timestamped_texts = ["<|0.00|> " + text for text in timestamped_texts]

            # Check if there's more than one segment
            if len(timestamped_texts) > 1:
                # Split the audio and text
                splits = self.split_audio_and_text(content['audio'], timestamped_texts)
                
                for split_audio_path, split_text in splits:
                    new_content['audio'] = split_audio_path
                    new_content['text'] = split_text
                    new_contents.append(copy.deepcopy(new_content))
            else:
                new_content['text'] = timestamped_text
                new_contents.append(new_content)

        json_dump_file = "test.json"
        with open(json_dump_file, 'w', encoding='utf-8') as f:
            json.dump(new_contents, f, ensure_ascii=False, indent=4)
        exit()

        # with open(filter_dump_file, 'w', encoding='utf-8') as f:
        #     json.dump(filtered_contents, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # train_dir = "/home/ubuntu/Workspace/DB/korean_db/data/KlecSpeech/train.json"
    # valid_dir = "/home/ubuntu/Workspace/DB/korean_db/data/KlecSpeech/validation.json"

    # with open(train_dir, 'r') as f:
    #     jsons = json.load(f)
    
    # split_count = len(jsons) // 80
    # for idx in range(80):
    #     if idx == 79:
    #         temp_json = jsons[split_count*idx:]
    #     else:
    #         temp_json = jsons[split_count*idx:split_count*(idx+1)]
        
    #     with open(f'datas/train_{idx}.json', 'w', encoding='utf-8') as f:
    #         json.dump(temp_json, f, ensure_ascii=False, indent=4)

    # for i in range(5):
    #     with open(f"list{i}.txt", 'w') as f:
    #         for idx in range(i*16, (i+1)*16):
    #             f.write(f'datas/train_{idx}.json\n')
        
    args = get_args()
    asr = ASR(pipe_type='fast-pipe')
    args.json_lists = "/home/ubuntu/Workspace/DB/korean_db/workspace/datas/train_0.json"
    
    with open(args.json_lists, 'r') as f:
        json_lists = f.readlines()

    fname = args.json_lists.split("/")[-1]
    json_dump_file = f"new_datas/{fname}"
    filter_dump_file = f"filtered/{fname}"

    asr.transcribe_speech(fname, json_dump_file, filter_dump_file)

    # asr.transcribe_speech(None, None, None)