import os
import re
import spacy
from transformers import pipeline
from bs4 import BeautifulSoup

# 加载spaCy中文模型
nlp = spacy.load('zh_core_web_sm')

# 使用Hugging Face Transformers进行事件提取
event_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 设定文件目录
directory = r'C:\Users\chan\Desktop\研究\危险品数据集相关\危化品数据集相关\no_tag_html'

def process_text(text):
    # 删除“16. 其它信息”部分及之后的内容
    text = re.sub(r'16\.\s*其它信息[\s\S]*', '', text)
    return text

def extract_sentences(text):
    # 使用spaCy进行中文分句
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def extract_entities(text):
    # 使用spaCy进行NER识别
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities[ent.text] = ent.label_
    return entities

def generate_events(sentences):
    events = []
    for i, sentence in enumerate(sentences):
        # 使用Transformers的zero-shot-classification来识别事件类型
        result = event_model(sentence, candidate_labels=["危险化学品信息", "应急处理", "健康危害", "环境危害"])
        event_type = result['labels'][0]  # 获取最可能的事件类型
        events.append({
            "recguid": str(i + 1),
            "event_type": event_type,
            "arguments": {
                "sentence": sentence
            }
        })
    return events

def generate_data_structure(file_path):
    # 读取HTML文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text()

    # 处理文本，删除不需要的部分
    cleaned_text = process_text(text_content)

    # 提取句子
    sentences = extract_sentences(cleaned_text)

    # 提取实体
    entities = extract_entities(cleaned_text)

    # 提取事件
    events = generate_events(sentences)

    # 获取实体范围 (ann_mspan2dranges)
    ann_mspan2dranges = {}
    for ent in entities:
        ann_mspan2dranges[ent] = [(sentences.index(sentence), sentence.find(ent), sentence.find(ent) + len(ent))
                                  for sentence in sentences if ent in sentence]

    # 推断实体类型 (ann_mspan2guess_field)
    ann_mspan2guess_field = {ent: "Chemical" if label == "ORG" else "Location" for ent, label in entities.items()}

    # 构建最终的数据结构
    doc_structure = [{
        "doc_1": {
            "sentences": sentences,
            "recguid_eventname_eventdict_list": events,
            "ann_mspan2dranges": ann_mspan2dranges,
            "ann_mspan2guess_field": ann_mspan2guess_field
        }
    }]

    return doc_structure

# 遍历目录中的HTML文件并处理
all_docs = []
for filename in os.listdir(directory):
    if filename.endswith('.html'):
        file_path = os.path.join(directory, filename)
        doc_structure = generate_data_structure(file_path)
        all_docs.extend(doc_structure)

# 输出处理后的数据结构
print(all_docs)
