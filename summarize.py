from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from utils import create_chunks
from kiwipiepy import Kiwi


# 유사도 검사 / 내용 추출 프롬프트

def summarize_doc(
        texts: list,
        tagged_value: list
    ) -> str:
    kiwi = Kiwi() # 한국어 문장 분할
    summary_result = []
    summary_list = ['TG-1','TG-2','TG-3','TG-4','TG-5', 'TG-6']
    summary_prompt = """
    아래 내용을 요약하지 말고 쉽게 풀어서 설명해줘. 대답의 내용은 중복되면 안돼.
    만약 서열 번호나 제품의 종류, 화학식 등 특정 항목에 대한 요소들을 길게 나열한 정보가 입력되면 해당 내용의 자세한 요소들은 제외하고 하나의 항목으로 요약해줘.
    
    ===
    {return_docs}
    ===
    대답 : 
    """
    summary_template = PromptTemplate(template=summary_prompt, input_variables=["return_docs"])

    llm3000 = ChatOpenAI(
            model_name='gpt-4',
            temperature="0",
            request_timeout=3600,
            max_retries=30,
            streaming=True,
            max_tokens=4000,
    )

    for k in range(3) :
        if len(splitted_texts[(tagged_value == summary_list[k*2]) | (tagged_value == summary_list[k*2+1])]) == 0 :
            summary_result.append('\n')
        else :
            texts = []
            temp = kiwi.split_into_sents(text="".join(splitted_texts[(tagged_value == summary_list[k*2]) | (tagged_value == summary_list[k*2+1])]))
            for m in range(0, len(temp)):
                texts.append(temp[m][0])
            chuncks = create_chunks(texts, max_chunk_length=2000)
            chuncks = [x for x in chuncks if len(x) != 0]
            for l in range(len(chuncks)) :
                if l == 0 :
                    summary_result.append(LLMChain(prompt=summary_template,
                    llm = llm3000).run({"return_docs" :  chuncks[l]}))
                else :
                    summary_result[summary_list.index(summary_list[k])] = summary_result[summary_list.index(summary_list[k])] + LLMChain(prompt=summary_template, llm=llm3000).run({"return_docs" :  chuncks[l]})

    
    return summary_result

#################################################################################################
