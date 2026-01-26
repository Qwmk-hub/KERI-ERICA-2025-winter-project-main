import os
import hashlib
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 설정 ---
DATA_PATH = './docs' 
DB_PATH = 'vectorstore/db_faiss'
MODEL_PATH = './bge-m3'

def get_file_hash(file_path):
    """파일의 내용을 읽어 SHA-256 해시값 생성"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def get_existing_hashes(vectorstore):
    """DB 메타데이터에 저장된 모든 해시값 추출"""
    if vectorstore is None:
        return set()
    
    hashes = set()
    for doc_id, doc in vectorstore.docstore._dict.items():
        if 'file_hash' in doc.metadata:
            hashes.add(doc.metadata['file_hash'])
    return hashes

def create_vector_db():
    # 1. 임베딩 모델 로드
    print(f"\n[1/4] 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. 기존 DB 로드
    vectorstore = None
    existing_hashes = set()
    if os.path.exists(DB_PATH):
        vectorstore = FAISS.load_local(
            DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        existing_hashes = get_existing_hashes(vectorstore)

    # 3. 신규 파일 판별 (해시 기준)
    if not os.path.exists(DATA_PATH):
        print(f"오류: {DATA_PATH} 폴더가 없습니다.")
        return

    all_pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith('.pdf')]
    files_to_process = []

    for file in all_pdf_files:
        file_path = os.path.join(DATA_PATH, file)
        f_hash = get_file_hash(file_path)
        
        if f_hash not in existing_hashes:
            files_to_process.append((file, f_hash))

    if not files_to_process:
        print("모든 문서가 이미 최신 상태입니다.")
        return

    print(f"새로 처리할 문서: {len(files_to_process)}개")

    # 4. 문서 로딩 및 해시 정보 삽입
    new_documents = []
    for file, f_hash in tqdm(files_to_process, desc="문서 읽기"):
        pdf_path = os.path.join(DATA_PATH, file)
        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            # 각 조각에 해시값 강제 주입
            for d in docs:
                d.metadata['file_hash'] = f_hash
                d.metadata['source'] = file # 파일명도 업데이트
            new_documents.extend(docs)
        except Exception as e:
            print(f"\n파일 로드 오류 ({file}): {e}")

    # 5. 분할 및 DB 업데이트
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(new_documents)

    print(f"\n[4/4] {len(texts)}개의 조각 추가 중...")
    if vectorstore:
        vectorstore.add_documents(texts)
    else:
        vectorstore = FAISS.from_documents(texts, embeddings)

    vectorstore.save_local(DB_PATH)
    print(f"완료: 새로운 내용이 성공적으로 추가되었습니다.")

if __name__ == "__main__":
    create_vector_db()