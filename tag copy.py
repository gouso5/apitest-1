from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from utils import create_chunks
from kiwipiepy import Kiwi
import re


# 유사도 검사 / 내용 추출 프롬프트
tagging_template = '''You are a model classifying the meanings of sentences.

When I input a paragraph, please tag each sentence with its meaning related to the following items:

Background Description

Classification Code: TG-1
Description: This refers to the definition and explanation of diseases, products, technologies related to the invention, research purpose, related research trends, etc.
Industrial Needs

Classification Code: TG-2
Description: This pertains to the related industry, problems and limitations of existing products and technologies, needs, etc.
Technology Introduction

Classification Code: TG-3
Description: This refers to the detailed explanation of the internal structure, components, operating principles, theories, models, methodologies, etc., of the invention.
Experiment and Implementation Examples

Classification Code: TG-4
Description: This refers to the manufacturing and verification process related to the invention, experiments and implementation examples, comparison examples, etc.
Utilization of the Invention

Classification Code: TG-5
Description: This refers to the content about various application plans such as pharmaceutical compositions, foodstuffs, other processed products, product lines, applications, etc. that can be manufactured through the invention.
Effects of the Invention

Classification Code: TG-6
Description: This refers to the positive effects that can be obtained through the invention, features of the invention, advantages, performance, expected effects, etc.

Please classify the following content. Generate a classification name and classification code. If it does not belong to the classification items, please mark it as null.


답변은 한글로 작성해줘.
===
'''

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k", 
    messages=[{
        "role": "user", 
        "content": tagging_template,
    }]+[{"role": "user", 
        "content": chuncks[p]}], 
    functions=[
        {
            "name": "tagging", 
            "description": "Tag each sentence", 
            "parameters": {
                "type": "object", 
                "properties": {
                    "Sentences": {
                        "type": "array",
                        "description": "Please input the sentence classified within the paragraph.",
                        "items" : {
                            "type": "string"
                            }
                        }, 
                    "Classification Codes": {
                        "type": "array",
                        "description": "Please input the classification code corresponding to the sentence.",
                        "items" : {
                            "type": "string"
                            }
                        }
                    }, 
                "required": ["Sentences", "Classification Codes"]
            }, 
        }, 
    ], 
    function_call="auto"
)
completion["choices"][0]["message"]["function_call"]["arguments"]["Sentence"]

def tag_text(
        text: str
    ) -> list:
    texts = []
    kiwi = Kiwi()
    temp = kiwi.split_into_sents(text=text.replace("\n", ""))
    for m in range(0, len(temp)):
        texts.append(temp[m][0])
    chuncks = create_chunks(texts, max_chunk_length=2000)
    chuncks = [x for x in chuncks if len(x) != 0]
    tagged_value = []
    text_pattern = r'Classification Code: (.*?)\n'
    for p in range(len(chuncks)) :
        cl_result = LLMChain(prompt=tagging_prompt, llm=llm).run({"return_docs" :  chuncks[p]})
        text_pattern_data = re.findall(text_pattern, cl_result+'\n')
        if len(text_pattern_data) == 0 :
            tagged_value.extend([cl_result])
        else :
            tagged_value.extend(text_pattern_data)
    return texts, tagged_value

#################################################################################################
