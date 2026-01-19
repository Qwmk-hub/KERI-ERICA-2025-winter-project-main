import os
from tqdm import tqdm  # 진행 바 라이브러리
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 설정 ---
DATA_PATH = './docs' 
DB_PATH = 'vectorstore/db_faiss'
MODEL_PATH = './bge-m3'

def create_vector_db():
    # 1. PDF 파일 목록 확인
    if not os.path.exists(DATA_PATH):
        print(f"오류: {DATA_PATH} 폴더가 없습니다.")
        return

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"알림: {DATA_PATH} 폴더 내에 PDF 파일이 없습니다.")
        return

    # 2. 문서 로딩 (파일 단위 프로그레시브 바)
    all_documents = []
    print("\n[1/3] PDF 문서 로딩 중...")
    for file in tqdm(pdf_files, desc="문서 읽기"):
        pdf_path = os.path.join(DATA_PATH, file)
        try:
            loader = PyMuPDFLoader(pdf_path)
            all_documents.extend(loader.load())
        except Exception as e:
            print(f"\n파일 로드 오류 ({file}): {e}")
    
    if not all_documents:
        return

    # 3. 텍스트 분할
    print("\n[2/3] 텍스트 분할 중...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_documents)

    # 4. 임베딩 모델 로드
    print("\n[3/3] 임베딩 모델 로드 및 벡터 DB 생성 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 5. 벡터 DB 생성 (배치 단위로 처리하여 진행 바 표시)
    try:
        # 처음 1개 데이터로 초기 DB 생성
        vectorstore = FAISS.from_documents(texts[:1], embeddings)
        
        # 나머지 데이터를 배치 단위로 추가하며 진행률 표시
        batch_size = 10  # 10개씩 묶어서 처리
        for i in tqdm(range(1, len(texts), batch_size), desc="벡터화 작업"):
            batch = texts[i : i + batch_size]
            vectorstore.add_documents(batch)

        # 저장
        if not os.path.exists('vectorstore'):
            os.makedirs('vectorstore')
        vectorstore.save_local(DB_PATH)
        print(f"\n성공: 벡터 DB가 '{DB_PATH}'에 저장되었습니다.")
        
    except Exception as e:
        print(f"\n벡터 DB 생성 오류: {e}")

if __name__ == "__main__":
    create_vector_db()