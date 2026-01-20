# RAG-document-qna

RAG, Langchain 이용 KERI 기술문서 QnA AI 에이전트 소스 코드


## 준비 (Windows)

1. 아래 명령어로 `Git`을 설치 후 LFS을 활성화합니다.
   `winget install --id Git.Git -e --source winget`
   `git lfs install`

2. 아래 명령어로 텍스트 임베딩 모델을 다운받아 스크립트와 동일한 디렉토리에 저장합니다.
   `git clone --depth 1 https://huggingface.co/BAAI/bge-m3`

3. 요구 패키지들을 설치합니다.
   `pip install -r requirements.txt`

5. `./docs` 폴더 내에 참조 자료 (pdf)들을 저장하고 아래 명령어로 일괄 분석 및 저장을 시행합니다.
   `python ./ingest.py`

6. AI 에이전트를 실행하고 질의응답을 수행합니다.
   `python ./agent.py`
