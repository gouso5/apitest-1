from langchain import LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory,ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.document_loaders import PyPDFLoader
import re
import os



############## Setup QA Chain ##############

def refine_text() -> str:
    Chat_template = """You: First, ask the user what patent they would like to write a technical for. Next, organize what you need to write a technical script for your patent. After this you ask the user a question about each item. You should ask your questions sequentially, one at a time, but clearly and concisely so that non-experts can understand them.
        And at the end of the last question, please provide a final summary of the questions and answers. And when presenting the final summary, display the value '[state: True]' and ask the user to write a transcript.
        If the user gives an answer unrelated to the question, display [test: true] and ask the user a second question.
        Always reply in Korean.
        Please do not repeat what I said or include unnecessary information such as [Note].

        ===
        history : {chat_history}
        ===
        Human: {question}
        ===
        AI:
        ===""" 
    llm = ChatOpenAI(temperature=0.2,
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4")
    Chat_prompt = PromptTemplate(
        input_variables=["chat_history", "question"], template=Chat_template
    )
    chat_chain = LLMChain(
        llm=llm, 
        verbose=False, 
        prompt=Chat_prompt
    )
    return chat_chain

# 주어진 문자열을 패턴에 맞게 정제하여 반환
def refine_text(pattern: str, text: str) -> str:
    pattern_data = re.findall(pattern, text)
    regex_pattern = r"|".join([re.escape(item) for item in pattern_data])
    refined_text = re.sub(regex_pattern, "", text)
    return refined_text

# 주어진 문자열에서 불필요한 텍스트를 삭제하여 반환
def remove_dump(text: str) -> str:
    return refine_text(
        pattern=r'공개특허 \d+-\d+-\d+\n-\d+-|등록특허 \d+-\d+-\d+\n-\d+-|공개특허 \d+-\d+\n-\d+-|등록특허 \d+-\d+\n-\d+-', 
        text=text
    )

# 주어진 문자열에서 줄번호를 삭제하여 반환
def remove_line_n(text: str) -> str:
    return refine_text(
        pattern=r' \[\d+\]', 
        text=text
    )

def extract_pdf(path: str) -> str:
    file_path = os.path.join(path)
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    page_texts = [remove_dump(page.page_content) for page in pages]
    combined_text = " ".join(page_texts)
    result = remove_line_n(combined_text)
    text = re.sub("(발명의 명칭|배 경 기 술|기 술 분 야|과제의 해결 수단|발명의 내용|발명의 효과|도면의 간단한 설명|발명을 실시하기 위한 구체적인 설명|해결하려는 과제)",'',result)
    return text

############## Setup QA Chain ##############