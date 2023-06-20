from chromadb.config import Settings
import json

import os
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from typing import List, Optional
from googletrans import Translator
from googletrans.client import Translated
import openai

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma

from pydantic import BaseModel


load_dotenv()

os.chdir("./질문하기")

from chat import extract_pdf, refine_text
from summarize import summarize_doc
from tag import tag_text
from db import auth

############## Models ##############
class Chat(BaseModel):
    question: str
    answer: str

class TextRequest(BaseModel):
    chat_history: List[Chat]
    new_question: str

class TextResponse(BaseModel):
    text: str

class TagResponse(BaseModel):
    tagged_value: list
    texts: list

class SummarizeRequest(BaseModel):
    tag: List[TagResponse]

class SummarizeResponse(BaseModel):
    summary_result: list
############## Models ##############

app = FastAPI()

@app.get('/', status_code=status.HTTP_200_OK)
def healthcheck():
    return { 'message': 'Everything OK!' }

## 3가지 방식(문서입력방식 / 문서선택방식 / 챗봇방식) 중 하나를 통해 특허에 관련된 내용을 입력
@app.get("/api/v1/docs/content", dependencies=[Depends(auth)])
async def read_content(path : str):
    text = extract_pdf(path = path)
    return TextResponse(text=text)

@app.get("/api/v1/docs/content", dependencies=[Depends(auth)])
async def load_content(content : str):
    return TextResponse(text=content)

@app.post("/api/v1/docs/content", dependencies=[Depends(auth)])
async def chat(body: TextRequest) -> TextResponse:
    refine_text()
    question = body.new_question
    chat_history = [(history.question, history.answer) for history in body.chat_history]
    result = chat_chain.run({"question": question, "chat_history": chat_history})
    return TextResponse(text=result['answer']) 

## 입력된 정보를 기반으로 각 줄의 tag 생성
@app.get("/api/v1/docs/tag", dependencies=[Depends(auth)])
async def tagging_text(text : str) :
    texts, tagged_value = tag_text(text = text)
    return TagResponse(tagged_value = tagged_value,texts = texts)

## tag 항목 중 내용이 부족한 부분을 질문
#
#

## 요약된 내용을 기반으로 
@app.post("/api/v1/docs/summarize", dependencies=[Depends(auth)])
async def summary_tag(body: SummarizeRequest) -> SummarizeResponse:
    summary_result = summarize_doc(texts = body.texts, tagged_value = body.tagged_value)
    return SummarizeResponse(summary_result=summary_result)

##사용자가 재생성버튼을 누르면 내용을 다시 요약

##사용자가 계속버튼을 누르면 생성하기로 이동