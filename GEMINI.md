# LLM-as-a-Judge Evaluation Tool - Progress & Mandates

이 파일은 프로젝트의 현재 상태, 구현된 핵심 로직, 그리고 유지보수를 위한 필수 규칙을 기록합니다.

## 📌 현재 진행 상황 (2026-04-06 기준)

### 1. 핵심 아키텍처
- **모델 엔진**: Gemini 모델 전용으로 `google-genai` SDK를 사용하며, 그 외 모델은 `litellm`을 통해 처리합니다.
- **평가 모드**:
    - **Absolute Scoring**: 루브릭 기준에 따른 1-5점 절대 평가.
    - **Pairwise Comparison**: 두 답변(A, B) 간의 비교 평가 및 개별 점수(score_a, score_b) 산출.
    - **Batch Process**: 대량의 CSV/JSONL 파일 순차 처리.

### 2. 구현된 핵심 기능
- **Rate Limiting**: Gemini API의 5 RPM 제한을 준수하기 위해 모든 호출 간 **최소 12.1초 간격**을 강제하는 전역 락(Lock) 로직 적용.
- **Sequential Processing**: 배치 처리 시 병렬성으로 인한 Quota 초과를 막기 위해 모든 처리를 **선형(Linear) 순차 방식**으로 통일.
- **프롬프트 관리**: `judge_tool/core/prompts.py`에서 함수형 템플릿으로 분리 관리.
- **웹 UI 강화**:
    - Chart.js를 이용한 점수 분포 및 승률 시각화 대시보드.
    - 결과 데이터의 **JSON 복사 및 다운로드** 기능 (모든 모드).
    - **Human Feedback**: 판사 결과에 대한 👍동의/👎비동의 데이터를 JSON에 포함.
    - **Reference Answer**: 모든 평가 모드에서 모범 답안 주입 가능.

## 🛠 실행 및 운영 지침

### 실행 방법
```bash
# 로컬 전용
PYTHONPATH=. python3 -m judge_tool.cli.main ui

# 외부 기기 접속 허용 (공유기 내)
PYTHONPATH=. python3 -m judge_tool.cli.main ui --host 0.0.0.0
```

### 필수 의존성
- `google-genai`: Gemini 2.5 Flash 모델 호출용
- `litellm`: 범용 LLM 인터페이스
- `fastapi`, `uvicorn`: 웹 서버
- `pandas`: 데이터 처리
- `chart.js`: UI 시각화 (CDN 사용)

## ⚠️ 개발 및 수정 규칙 (Mandates)

1. **모델 고정**: 현재 판사 모델의 기본값은 `gemini-2.5-flash`입니다.
2. **속도 제한 준수**: API 호출부(`_call_llm`) 이전의 `_wait_for_rate_limit()` 호출을 절대로 제거하지 마십시오.
3. **데이터 포맷**: 결과 JSON 구조는 `type`, `model`, `prompt`, `response`, `result` (내부에 score, reasoning 등 포함) 형식을 엄격히 준수해야 합니다.
4. **배치 처리**: 성능 향상을 위해 `ThreadPoolExecutor`를 다시 도입하더라도, RPM 제한 로직과 충돌하지 않도록 주의해야 합니다. (현재는 안전을 위해 Sequential 방식 사용 권장)
5. **루브릭 관리**: 새로운 평가 기준은 `configs/*.yaml` 추가만으로 작동하도록 설계되었으므로, UI 드롭다운에 항목을 추가할 때 파일명과 매칭되도록 하십시오.

---
*이 문서는 프로젝트의 상태가 변경될 때마다 업데이트되어야 합니다.*
