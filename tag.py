from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from utils import create_chunks
from kiwipiepy import Kiwi
import re


# 유사도 검사 / 내용 추출 프롬프트
tagging_template = """
    You play the role of classifying sentences in patent specifications or papers according to their context and extracting the necessary content for creating a technical promotional video script. The script includes the following components: introduction, main body, and conclusion.

    Introduction: Background explanation, problems and limitations
    Main body: Operating principle of the invention, internal structure, components, experimental information
    Conclusion: Features of the invention, advantages, application methods, expected effects

    The following classification items are available:

    Background Explanation
    Classification Code: TG-1
    Description: Definitions and explanations related to the invention, such as diseases, products, technologies, research objectives, and research trends.

    Industry Needs
    Classification Code: TG-2
    Description: Problems, limitations, and needs of related industries, existing products, and technologies.

    Technology Introduction
    Classification Code: TG-3
    Description: Detailed explanations of the internal structure, components, operating principles, theories, models, and methodologies of the invention.

    Existing Research and Experiments
    Classification Code: TG-4
    Description: The production and validation process, experiments, and comparative examples related to new research.

    Invention Applications
    Classification Code: TG-5
    Description: Various application methods, such as pharmaceutical compositions, food products, other processed products, product lines, and applications that can be manufactured through the invention.

    Invention Effects
    Classification Code: TG-6
    Description: Positive effects that can be obtained through the invention, features, advantages, performance, and expected effects.

    Generate classification code for sentence. If it does not fall into a classification item, display null.
    Show me just one classified code. 
    Example ) Classification Code: 


    Content: {return_docs}"""

tagging_prompt = PromptTemplate(template=tagging_template, input_variables=["return_docs"])

llm = ChatOpenAI(
  model_name='gpt-4',
  temperature="0",
  request_timeout=3600,
  max_retries=30,
  streaming=True,
  max_tokens=4000,
)

def tag_text(
        text: str
    ) -> list:
    texts = []
    kiwi = Kiwi()
    temp = kiwi.split_into_sents(text=text.replace("\n", ""))
    for m in range(0, len(temp)):
        texts.append(temp[m][0])
    tagged_value = []
    text_pattern = r'Classification Code: (.*?)\n'
    for p in range(len(texts)) :
        cl_result = LLMChain(prompt=tagging_prompt, llm=llm).run({"return_docs" :  texts[p]})
        text_pattern_data = re.findall(text_pattern, cl_result+'\n')
        if len(text_pattern_data) == 0 :
            tagged_value.extend([cl_result])
        else :
            tagged_value.extend(text_pattern_data)   
    return texts, tagged_value
#################################################################################################
