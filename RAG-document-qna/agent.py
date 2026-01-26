import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- 설정 ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_PATH = './bge-m3'
DB_PATH = 'vectorstore/db_faiss'

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 정의되어 있지 않습니다.")

def get_agent():
    # 1. 임베딩 모델 로딩 (가용한 경우 GPU 사용)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. FAISS DB 로드
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"벡터 DB가 '{DB_PATH}'에 없습니다.")
    
    vectorstore = FAISS.load_local(
        DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True 
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 3. LLM 설정 (모델명 확인 필요)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    # 4. 프롬프트 템플릿 (시스템 메시지 활용)
    system_prompt = (
        "당신은 기술 지원 전문가입니다. 아래 제공된 [참고 문서] 내용만을 바탕으로 답변하세요. "
        "문서에 내용이 없다면 '해당 내용은 문서에서 찾을 수 없습니다'라고 답변하세요.\n\n"
        "[참고 문서]\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. RAG 체인 구성 (최신 LCEL 방식)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def main():
    print("KERI 에이전트 초기화 중...")
    try:
        agent = get_agent()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return

    print("\n연결 완료. (종료: q)")
    while True:
        query = input("\n[질문]: ")
        if query.lower() == 'q': break
        
        print("답변 생성 중...")
        try:
            # create_retrieval_chain은 'input' 키를 사용합니다.
            response = agent.invoke({"input": query})
            
            print("\n" + "="*60)
            print(f"[답변]: {response['answer']}")
            print("="*60)
            
            print("\n[참고한 문서 조각]")
            for doc in response['context']:
                source = os.path.basename(doc.metadata.get('source', '알 수 없음'))
                print(f"- {source}: {doc.page_content[:50].strip()}...")
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()