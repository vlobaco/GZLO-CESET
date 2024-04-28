
from typing import List, Any
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
import nltk
import re

accents_dict = {
    'á':'a',
    'é':'e',
    'í':'i',
    'ó':'o',
    'ú':'u',
    'Á':'A',
    'É':'E',
    'Í':'I',
    'Ó':'O',
    'Ú':'U',
}
stop_words = nltk.corpus.stopwords.words('spanish')

def partition_file(file_path: str) -> List[Element] | Any:
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == 'pdf':
        partition = partition_pdf(
            file_path,
            # strategy = "hi_res",
            # hi_res_model_name = "yolox",
            # languages = ["es"],
            #infer_table_structure = True
            )
    elif file_extension == 'docx':
        partition = partition_docx(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    return partition

def mine_text_from_partition(partition: List[Element]) -> str:
    return ' '.join(element.text for element in partition)

def mine_text_from_pdf(file_path: str) -> str:
    partition = partition_file(file_path)
    return mine_text_from_partition(partition)

def mine_text_from_docx(file_path: str) -> str:
    partition = partition_file(file_path)
    return mine_text_from_partition(partition)

def preprocess_text(
        text: str, 
        remove_stop_words = False, 
        remove_punctuation = False,
        remove_accents = True,
        to_lower = True
        ) -> str:
    if to_lower:
        text = text.lower()
    if remove_accents:
        for accent, no_accent in accents_dict.items():
            text = text.replace(accent, no_accent)
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.split()
    if remove_stop_words:
        text = [word for word in text if word not in stop_words]
    return text
