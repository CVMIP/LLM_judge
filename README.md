# ⚖️ LLM-as-a-Judge (v2)

LLM-as-a-Judge v2는 의료 도메인(EMR 분석, 퇴원 요약지 타당성 검증 등)에 특화된 최첨단 AI 답변 평가 및 검증 도구입니다. 이 어플리케이션은 원본 텍스트에 기반한 Groundedness 검증, 다중 모델 앙상블(Ensemble) 평가, 사용자 피드백 로그 기능을 통합하여 신뢰성 있고 객관적인 LLM 성능 분석 워크플로우를 제공합니다.


## ✨ V2 주요 기능

- **Context-Aware Groundedness 검증:** 원본 EMR 문서나 근거 자료(Source Context)를 주입하여, 대상 LLM이 생성한 텍스트가 원본 데이터에 입각하여 정확하게 작성되었는지 환각(Hallucination) 여부를 판별합니다.
- **다수결 / 다중 모델 앙상블 평가 (Ensemble Judging):** GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Flash, Local-vLLM 등 여러 모델을 한 번에 다중 선택하여, 각 모델의 스코어를 평균하거나 다수결(Tie-breaking)로 승자를 결정하여 판정의 객관성을 극대화합니다.
- **Human-in-the-Loop (인간 피드백 결합):** LLM 판사의 평가 결과를 검토하고 '👍 동의' / '👎 비동의' 항목을 클릭할 수 있으며, 비동의 시 구체적인 사유(Reasoning)를 입력받아 데이터를 축적할 수 있습니다.
- **로컬 LLM (OLLAMA / vLLM) 완전 지원:** 로컬망에 띄워진 보안 LLM이나 자체 Fine-Tuned 모델의 `API Base URL`을 입력하여 의무기록과 같이 민감한 데이터의 외부 유출 없이 로컬 생태계 내에서 폐쇄적으로 평가를 진행할 수 있습니다.
- **유연한 UI View 및 JSON 내보내기 (Export):** Dual Source-Target 뷰를 지원하여 시각적으로 쉽게 모델 답변을 대조할 수 있으며, 클릭 한 번으로 통계 그래프 확인 및 데이터셋 구축을 위한 세부 JSON 익스포트 기능을 지원합니다.
- **고속 병렬 Batch 프로세싱:** CSV 및 JSONL 대량 데이터셋을 입력 시 멀티스레드 기반 처리로 수백, 수천 건의 프롬프트를 일괄적으로 자동 평가 및 산술 요약(히스토그램 차트)할 수 있습니다.

## 🚀 빠른 시작 가이드 (Quick Start)

### 1. 환경 구성
Python 3.10 이상의 환경을 권장합니다.
```bash
# 가상환경 생성 및 활성화
python -m venv judge_tool_venv_2
source judge_tool_venv_2/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
최상위 경로에 `.env` 파일을 생성하거나 로컬 브라우저 UI 설정(톱니바퀴 아이콘)에서 직접 API Key를 기입할 수 있습니다.
```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

### 3. 서버 실행
FastAPI 서버 및 내장 UI를 실행합니다.
```bash
python -m judge_tool.cli.main ui
```
서버 구동 후, 웹 브라우저에서 `http://127.0.0.1:8000` 으로 접속하여 평가 도구를 이용합니다.

## 📂 프로젝트 구조

```
judge_tool_v2/
├── judge_tool/
│   ├── cli/
│   │   └── main.py          # Command Line 인터페이스 및 서버 실행 진입점
│   ├── core/
│   │   ├── batch.py         # 멀티스레드 기반 데이터셋 대량 병렬 평가기
│   │   ├── judge.py         # LiteLLM/GenAI 연동 Scorer 및 예외 처리
│   │   └── prompts.py       # Base Prompt 및 Context 주입 템플릿 관리
│   ├── models/
│   │   └── schemas.py       # Pydantic 기반의 데이터 클래스 모델링 (Input/Result)
│   └── web/
│       ├── app.py           # FastAPI 라우터 및 백엔드 비즈니스 로직
│       └── static/
│           └── index.html   # TailwindCSS 기반의 Single Page Application 프론트엔드
├── configs/
│   ├── helpfulness.yaml     # 유용성 평가 프롬프트 Rubric
│   └── safety.yaml          # 안전성 평가 프롬프트 Rubric
├── .env                     # (Ignored) 환경 변수 저장
├── .gitignore
├── requirements.txt
└── README.md
```

## 🛠 사용 방법 상세

1. **설정 연동**: 우측 상단 `⚙️` 아이콘을 눌러 보유한 API Key와 로컬 API Base(필요시)를 입력합니다. 데이터는 귀하의 로컬 스토리지에만 저장됩니다.
2. **단일 평가 (Single Evaluation)**: Source Context (예: 응급실 초진 기록)와 Prompt, 그리고 Target LLM이 반환한 답변을 입력한 후, 다수의 판사 모델을 체크하여 평가합니다.
3. **쌍비교 (Pairwise Comparison)**: 두 가지 다른 모델의 답변(A/B)을 동시에 입력하여, Ground Truth 근거 문서와 대조한 후 우승자를 판별합니다.
4. **배치 평가 (Batch Process)**: 사전에 수집된 데이터셋 세트를 업로드하여, 각 Rubric(채점 기준)에 따른 모델들의 성능 평균을 차트로 시각화하고 최종 JSON을 다운로드합니다. 이 데이터는 향후 지식 증류(Knowledge Distillation) 기반 학습 데이터로 사용 가능합니다.
