import os
import json
import time
import pickle
import pathlib
import logging
import requests
from io import StringIO 
from pandas import read_json
from typing import Optional, Callable, List, Any
from dataclasses import dataclass
from langchain.schema import Document

@dataclass
class BatchSettings:
    api_key: str
    log_file_path: os.PathLike
    result_file_path: os.PathLike
    batch_cache_file_path: os.PathLike
    quality_control_path: os.PathLike = "archives"


DATASET_FILE_PATH = "corpus.jsonl"
QUALITY_CONTROL_FOLDER = "benchmark"

SETTINGS = BatchSettings(
    api_key="FDGdifg09u0u09SGO6SO75s",
    result_file_path="container/batch-01.jsonl",
    batch_cache_file_path="container/batches/SMA_BATCH_0.cache",
    quality_control_path="container/archives",
    log_file_path="container/logs/primary.log",
)

logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

def load_batch_cache_file(path: str) -> str:
    with open(pathlib.Path(path), "rb") as file:
        return file.read()

def convert_string_to_object(string: str) -> Any:
    return pickle.loads(string)

chunks: List[Document] = convert_string_to_object(
    load_batch_cache_file(SETTINGS.batch_cache_file_path)
)

JSONL_FORMAT_EXAMPLE = """
JSONL EXAMPLE:
{"instruction": "Jelaskan kegunaan memaknai sarung tangan keselamatan!", "context": "Sarung tangan keselamatan biasanya digunakan pada pekerjaan elektronika atau listrik agar tidak terjadi bahaya kena arus listrik dan termasuk jenis-jenis pekerjaan yang menggunakan bahan kimia.", "response": "Sarung tangan keselamatan digunakan untuk melindungi tangan dari benda tajam dan berbahaya, mencegah kecelakaan kerja yang fatal, membuat perlindungan dari benda panas, melindungi dari cairan kimia berbahaya, dan membuat pengguna tidak terpeleset. Selain itu, sarung tangan keselamatan juga dapat digunakan pada kegiatan pekerjaan listrik untuk mencegah kenaikan arus listrik dan pada kegiatan elektronika yang melibatkan bahan kimia berbahaya seperti membersihkan papan PCB.", "category": "closed_qa"}
{"instruction": "Apa yang dimaksud dengan budaya kerja?", "context": "Guru juga dapat menugaskan peserta didik untuk melakukan simulasi penerapan budaya kerja 5R di rumah atau di kamar sendiri. Peserta didik kemudian melaporkan perkembangan penerapan ini kepada guru dalam bentuk jurnal harian.", "response": "Budaya kerja adalah standar yang dipraktikkan berulang kali oleh anggota (karyawan) dalam suatu organisasi, yang mencerminkan sikap terhadap perilaku, keyakinan, cita-cita, pendapat, dan tindakan, yang menunjukkan pekerjaan atau bertindak sebagai motivator untuk meningkatkan moral atau kinerja di tempat kerja.", "category": "closed_qa"}
{"instruction": "Jelaskan kegunaan memaknai safety shoes?", "context": "Safety shoes digunakan pada pekerjaan yang melibatkan benda tajam dan berbahaya.", "response": "Safety shoes digunakan untuk melindungi dari benda tajam dan berbahaya, mencegah kecelakaan kerja yang fatal, membuat perlindungan dari benda panas, melindungi dari cairan kimia berbahaya, dan membuat pengguna tidak terpeleset.", "category": "closed_qa"}
{"instruction": "Jaka pepetrilista memang budaya kerja?", "context": "Budaya kerja 5R merupakan singkatan dari Ringkas, Rapi, Resik, Rawat, dan Rajin.", "response": "Budaya kerja 5R merupakan singkatan dari Ringkas, Rapi, Resik, Rawat, dan Rajin, yang merupakan modifikasi dari program 5S yang dikembangkan di Jepang.", "category": "closed_qa"}
{"instruction": "Membuat pertanyaan-pertanyaan terkait penggunaan APD!", "context": "Sistem atau Soal dalam format Soal Formatif sederhana.", "response": "1. Jika sepatu pengaman tidak digunakan, apa yang akan terjadi? 2. Jika tidak memakai sarung tangan keselamatan, apa yang akan terjadi? 3. Jika tidak memakai helm pengaman, apa yang akan terjadi? 4. Jika tidak memakai masker atau respirator, apa yang akan terjadi?", "category": "closed_qa"}
{"instruction": "Jelaskan kegunaan warna kuning pada rambu peringatan!", "context": "Rambu peringatan biasanya digunakan pada pekerjaan yang melibatkan benda berbahaya.", "response": "Warna kuning pada rambu peringatan menunjukkan potensi bahaya atau sebagai peringatan agar waspada.", "category": "closed_qa"}
{"instruction": "Jelaskan kegunaan tanda seru pada rambu peringatan!", "context": "Tanda seru dalam rambu peringatan berarti waspada.", "response": "Tanda seru dalam rambu peringatan berarti waspada.", "category": "closed_qa"}
{"instruction": "Jelaskan kegunaan tulisan \"Warning\" pada rambu peringatan!", "context": "Tulisan \"Warning\" pada rambu peringatan menegaskan agar pembaca berhati-hati.", "response": "Tulisan \"Warning\" pada rambu peringatan menegaskan agar pembaca berhati-hati.", "category": "closed_qa"}
{"instruction": "Pertanyaan untuk menguji pengetahuan peserta didik!", "context": "Sistem atau Soal dalam format Soal Formatif sederhana.", "response": "1. Apa yang dimaksud dengan budaya kerja? 2. Apa itu budaya kerja 5R? 3. Bagaimana cara membuat tempat kerja rapi? 4. Apa kegunaan safety shoes? 5. Apa yang dimaksud dengan rambu peringatan? 6. Apa yang dimaksud dengan rambu larangan?", "category": "closed_qa"}
{"instruction": "Sesuaikan kebutuhan", "context": "Sesuaikan dengan kebutuhan!", "response": "Sesuai kebutuhan dan kondisi!", "category": "closed_qa"}
{"instruction": "Kasus atau contoh soal untuk menguji pengetahuan peserta didik!", "context": "Sistem atau Soal dalam format Soal Formatif sederhana.", "response": "1. Seorang pekerja elektronika tidak mengenakan sarung tangan keselamatan saat bekerja. Apa yang akan terjadi? 2. Seorang pekerja tidak memakai helm pengaman saat bekerja. Apa yang akan terjadi? 3. Seorang pekerja tidak memakai masker atau respirator saat bekerja. Apa yang akan terjadi?", "category": "open_qa"}
{"instruction": "Kasus atau contoh soal untuk menguji pengetahuan peserta didik!", "context": "Sistem atau Soal dalam format Soal Formatif sederhana.", "response": "1. Seorang pekerja tidak memakai safety shoes saat bekerja di tempat yang berbahaya. Apa yang akan terjadi? 2. Seorang pekerja tidak memakai sarung tangan keselamatan saat membersihkan papan PCB. Apa yang akan terjadi? 3. Seorang pekerja tidak memakai helm pengaman saat bekerja di atas bangunan tinggi. Apa yang akan terjadi?", "category": "open_qa"}
"""
RULES = """
RULES TO FOLLOW:
- Respond in pure jsonl no texts outside the jsonl!
- Ensure high quality data!
- Remove unnecessary information and invalid characters!
- Respond as long and detail as possible!
"""

def set_interval_no_thread_no_argument_function(func: Callable, sec: int, iterations: Optional[int] = None):
    count = 0
    while True:
        func()
        time.sleep(sec)
        count += 1
        if iterations is not None and count >= iterations:
            break

def gen_message(data: str):
    return [
        {
            "role": "system",
            "content": f"You are a gonna help the fine-tuning process by generating jsonl like this {JSONL_FORMAT_EXAMPLE} and {RULES}!"},
        {
            "role": "user",
            "content": f"I want you to generate jsonl about a complex topic in Indonesian Language: {data}!"
        },
    ]


def get_response_from_chunk(chunk: str):
    res = requests.post(
        f"https://model-.api.baseten.co/production/predict",
        headers={"Authorization": f"Api-Key "},
        json={
            "prompt": gen_message(chunk),
            "stream": True,
            "max_tokens": 16384,
            "temperature": 0.9
        },
        stream=False
    )
    return res.text


def put_to_quality_control(data: str, title: str):
    with open(title, "w") as file:
        file.write(str(data))


def need_quality_control(data: str, index: int) -> bool:
    try:
        read_json(StringIO(data), lines=True)
    except ValueError:
        logger.error(f"Invalid JSONL syntax: {index} -> Quality Control")
        return True
    return False


def start_process(settings: BatchSettings):
    with open(pathlib.Path(settings.result_file_path), "a") as file:
        for index, chunk in enumerate(chunks):
            logger.info(f"[STARTING] chunk number: {index}")
            response = get_response_from_chunk(chunk.page_content)
            time.sleep(60) #! NOTE: IMPORTANT DO NOT DELETE (THE IMPLEMENTATION IS SPECIFIC FOR RPM LIMITS)
            if need_quality_control(response, index):
                put_to_quality_control(response, pathlib.Path(f"{QUALITY_CONTROL_FOLDER}/{index}-QC.jsonl"))
                logger.warning(f"Chunk number: {index} [FINISHED] -> [QC]")
            else:
                file.write(str(json.loads(response)))
                logger.info(f"Chunk number: {index} [FINISHED] -> [SUCCESS]")
    logger.info("ALL TASKS HAVE BEEN FINISHED!")
