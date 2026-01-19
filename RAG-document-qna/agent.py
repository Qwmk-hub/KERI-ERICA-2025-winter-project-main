import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 설정 ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_PATH = './bge-m3'
DB_PATH = 'vectorstore/db_faiss'

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 정의되어 있지 않습니다.")

def get_agent():
    # 1. 임베딩 모델 로딩
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. FAISS DB 로드
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"벡터 DB가 '{DB_PATH}'에 없습니다. ingest.py를 먼저 실행하세요.")
    
    vectorstore = FAISS.load_local(
        DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True 
    )

    # 3. LLM 설정
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    # 4. 프롬프트 템플릿
    template = """당신은 기술 지원 전문가입니다. 아래 제공된 [참고 문서] 내용만을 바탕으로 답변하세요.
문서에 내용이 없다면 "해당 내용은 문서에서 찾을 수 없습니다"라고 답하세요.

[참고 문서]
{context}

[질문]
{question}

전문가 답변:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 5. RAG 체인 구성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def main():
    print("KERI 에이전트 초기화 중...")
    try:
        agent = get_agent()
    except Exception as e:
        print(f"초기화 실패: {e}")
        # 어떤 모듈이 부족한지 구체적으로 확인하기 위한 출력
        import traceback
        traceback.print_exc()
        return

    print("\n연결 완료. (종료: q)")
    while True:
        query = input("\n[질문]: ")
        if query.lower() == 'q': break
        
        print("답변 생성 중...")
        try:
            response = agent.invoke({"query": query})
            print("\n" + "="*60)
            print(f"[답변]: {response['result']}")
            print("="*60)
            
            print("\n[참고한 문서 조각]")
            for doc in response['source_documents']:
                source = os.path.basename(doc.metadata.get('source', '알 수 없음'))
                print(f"- {source}: {doc.page_content[:50]}...")
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()