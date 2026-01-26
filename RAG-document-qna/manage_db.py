import os
from langchain_community.vectorstores import FAISS
# 실제 모델 대신 가짜 임베딩 클래스 사용
from langchain_core.embeddings import Embeddings

# 가짜 임베딩 클래스 정의 (아무 동작도 하지 않음)
class EmptyEmbeddings(Embeddings):
    def embed_documents(self, texts): return []
    def embed_query(self, text): return []

# --- 설정 ---
DB_PATH = 'vectorstore/db_faiss'

def load_db():
    if not os.path.exists(DB_PATH):
        print("에러: 벡터 DB가 존재하지 않습니다.")
        return None
    
    # 실제 모델 경로 대신 EmptyEmbeddings() 사용
    return FAISS.load_local(
        DB_PATH, 
        EmptyEmbeddings(), 
        allow_dangerous_deserialization=True
    )

def list_documents():
    vectorstore = load_db()
    if not vectorstore: return

    sources = set()
    doc_dict = vectorstore.docstore._dict
    for doc_id in doc_dict:
        metadata = doc_dict[doc_id].metadata
        if 'source' in metadata:
            sources.add(os.path.basename(metadata['source']))
    
    print(f"\n현재 DB 내 문서 목록: {sorted(list(sources))}")

def delete_document(file_name):
    """특정 파일과 관련된 모든 벡터 삭제"""
    vectorstore = load_db()
    if not vectorstore:
        return

    # 삭제할 파일의 원본 소스 경로와 일치하는 ID 찾기
    doc_dict = vectorstore.docstore._dict
    ids_to_delete = [
        doc_id for doc_id, doc in doc_dict.items()
        if os.path.basename(doc.metadata.get('source', '')) == file_name
    ]

    if not ids_to_delete:
        print(f"알림: '{file_name}'에 해당하는 문서를 찾을 수 없습니다.")
        return

    # ID 기반 삭제 및 DB 업데이트
    vectorstore.delete(ids_to_delete)
    vectorstore.save_local(DB_PATH)
    print(f"성공: '{file_name}' 관련 데이터 {len(ids_to_delete)}개를 삭제했습니다.")

def main():
    while True:
        print("\n[벡터 DB 관리 도구]")
        print("1. 목록 보기")
        print("2. 문서 삭제")
        print("q. 종료")
        
        choice = input("\n선택: ").strip().lower()
        
        if choice == '1':
            list_documents()
        elif choice == '2':
            file_to_del = input("삭제할 파일명(확장자 포함) 입력: ").strip()
            delete_document(file_to_del)
        elif choice == 'q':
            break
        else:
            print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()