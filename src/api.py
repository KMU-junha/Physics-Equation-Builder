"""
Physics Equation Builder — FastAPI Backend
실시간 GA 실행 스트리밍 + CSV 업로드 + 결과 관리
"""
import os, sys, re, uuid, json, threading, asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UI_DIR   = BASE_DIR / "ui"
PYTHON   = sys.executable

app = FastAPI(title="Physics Equation Builder API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 정적 파일 서빙 (ui/ 디렉터리)
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

# ── 실행 중인 Run 관리 ───────────────────────────────────────────────
RUNS: dict = {}

# ── 히스토리 파일 관리 ───────────────────────────────────────────────
HISTORY_FILE = DATA_DIR / "history.json"

def _load_history():
    if not HISTORY_FILE.exists(): return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_history(hist):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

def append_to_history(run_id, started_at, csv_name, pop_size, generations, penalty_weight, hints, result_eq, result_score, elapsed_time):
    hist = _load_history()
    hist.insert(0, {
        "run_id": run_id,
        "timestamp": started_at,
        "dataset_name": csv_name,
        "settings": {
            "pop_size": pop_size,
            "generations": generations,
            "penalty_weight": penalty_weight,
            "hints": hints
        },
        "result": {
            "equation_str": result_eq,
            "score": result_score,
            "elapsed_time_sec": round(elapsed_time, 2)
        }
    })
    _save_history(hist)

_ANSI = re.compile(r'\x1b\[[0-9;]*m')
def strip_ansi(s: str) -> str:
    return _ANSI.sub('', s)

def parse_line(line: str) -> Optional[dict]:
    s = strip_ansi(line).strip()
    if not s:
        return None

    # 세대 완료
    m = re.search(r'깊이\s*(\d+)\s*[|│]\s*세대\s*(\d+)\s*완료.*?누적 방출:\s*(\d+)개.*?최고점:\s*([\d.]+)', s)
    if m:
        return {"type": "generation", "depth": int(m.group(1)),
                "gen": int(m.group(2)), "eval_count": int(m.group(3)),
                "score": float(m.group(4)), "raw": s}

    # 실시간 평가 진행률
    m_prog = re.search(r'\[진행률\]\s*깊이\s*(\d+)\s*[|│]\s*세대\s*(\d+)\s*[|│]\s*(\d+)개', s)
    if m_prog:
        return {"type": "progress", "depth": int(m_prog.group(1)), "gen": int(m_prog.group(2)), "eval_count": int(m_prog.group(3)), "raw": s}

    # 플롯 데이터 캡처 (수치형 트랙)
    m_plot = re.search(r'\[PLOT_DATA\]\s*(\{.*\})', s)
    if m_plot:
        try:
            import json
            return {"type": "plot_data", "data": json.loads(m_plot.group(1)), "raw": ""}
        except Exception:
            pass

    # 점수가 포함된 모든 로그 캡처
    m_score = re.search(r'(?:적합도|Score|최고점|점수)[^0-9]*([\d.]+)', s, re.IGNORECASE)
    
    # 최종 수식 명시적 매칭 (main.py에서 "최종수식:" 접두어를 붙인 경우)
    m_final = re.search(r'최종수식:\s*(.+)', s)
    if m_final:
        return {"type": "best_kernel", "equation": m_final.group(1).strip('` '), "raw": s}

    # 쇼트컷 완료 수식 매칭
    m_shortcut = re.search(r'직접 추출 완료.*?:\s*(.+)', s)
    if m_shortcut:
        return {"type": "best_kernel", "equation": m_shortcut.group(1).strip('` '), "score": 100.0, "raw": s}

    # 전역 최고해 갱신 직후의 방정식 (Integral, Eq, Derivative 포함)
    if 'Integral(' in s or 'Eq(' in s or 'Derivative(' in s:
        idx = -1
        for k in ['Integral(', 'Eq(', 'Derivative(']:
            if k in s:
                idx = s.find(k)
                break
        if idx != -1:
            eq = s[idx:].strip('` ')
            return {"type": "best_kernel", "equation": eq, "score": float(m_score.group(1)) if m_score else None, "raw": s}

    # 완벽한 해 발견
    if '완벽한 해 발견' in s or '조기 종료' in s:
        return {"type": "solution_found", "raw": s}

    if m_score:
        return {"type": "score_update", "score": float(m_score.group(1)), "raw": s}

    return {"type": "log", "raw": s}


# ── 엔드포인트 ────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(UI_DIR / "index.html")

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """CSV 파일 업로드 및 미리보기"""
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    # 저장
    dest = DATA_DIR / file.filename
    dest.write_bytes(content)
    return {
        "filename": file.filename,
        "rows": max(0, len(lines) - 1),
        "columns": lines[0].split(",") if lines else [],
        "preview": lines[:5],
    }

@app.get("/api/sample/{dataset_type}")
async def generate_sample(dataset_type: str):
    import pandas as pd
    import numpy as np

    dest_filename = f"sample_{dataset_type}.csv"
    dest = DATA_DIR / dest_filename
    
    if dataset_type == "sym_integral":
        data = [
            {"input": "exp(-a*t**2)", "output": "sqrt(pi/a) * exp(-w**2/(4*a))", "roc": "re(a) > 0"},
            {"input": "exp(-a*t) * Heaviside(t)", "output": "1/(a + I*w)", "roc": "re(a) > 0"},
            {"input": "exp(-a*Abs(t))", "output": "2*a/(a**2 + w**2)", "roc": "re(a) > 0"},
            {"input": "1/(a**2 + t**2)", "output": "(pi/a) * exp(-a*Abs(w))", "roc": "re(a) > 0"}
        ]
        df = pd.DataFrame(data)
    elif dataset_type == "sym_derivative":
        data = [
            {"input": "sin(k*x)", "output": "k*cos(k*x)"},
            {"input": "exp(k*x)", "output": "k*exp(k*x)"},
            {"input": "x**2", "output": "2*x"},
            {"input": "log(x)", "output": "1/x"}
        ]
        df = pd.DataFrame(data)
    elif dataset_type == "sym_algebraic":
        data = [
            {"input": "sin(t)", "output": "2*sin(t) + 3"},
            {"input": "exp(t)", "output": "2*exp(t) + 3"},
            {"input": "t**2", "output": "2*t**2 + 3"},
        ]
        df = pd.DataFrame(data)
    elif dataset_type == "num_integral":
        # 1. 푸리에 파형 (사인 혼합) + 선형 증가 + 노이즈
        input_data = np.linspace(0, 10, 150)
        true_output = 3.0 * np.sin(2.0 * input_data) + 1.5 * input_data
        noise = np.random.normal(0, 0.5, size=len(input_data))
        df = pd.DataFrame({"input": input_data, "output": true_output + noise})
    elif dataset_type == "num_derivative":
        # 2. 감쇠 진동 (Damped Oscillator) + 노이즈 (SINDy 커널에 매칭됨)
        input_data = np.linspace(0, 10, 200)
        true_output = 5.0 * np.exp(-0.4 * input_data) * np.sin(4.0 * input_data)
        noise = np.random.normal(0, 0.1, size=len(input_data))
        df = pd.DataFrame({"input": input_data, "output": true_output + noise})
    elif dataset_type == "num_algebraic":
        # 3. 대수 다항식 + 큰 스케일의 가우시안 노이즈
        input_data = np.linspace(-5, 5, 150)
        true_output = 2.5 * input_data**2 - 1.2 * input_data + 3.0
        noise = np.random.normal(0, 2.0, size=len(input_data))
        df = pd.DataFrame({"input": input_data, "output": true_output + noise})
    else:
        return JSONResponse({"error": "Unknown dataset type"}, status_code=400)

    df.to_csv(dest, index=False)
    
    with open(dest, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        
    return {
        "filename": dest_filename,
        "rows": max(0, len(lines) - 1),
        "columns": lines[0].split(",") if lines else [],
        "preview": lines[:5],
    }

@app.post("/api/run")
async def start_run(body: dict):
    """GA 실행 시작 → run_id 반환"""
    run_id = str(uuid.uuid4())[:8]
    csv_name   = body.get("filename", "fraunhofer_k_symbolic.csv")
    pop_size   = int(body.get("pop_size", 1000))
    generations = int(body.get("generations", 15))
    penalty_weight = float(body.get("penalty_weight", 1.0))
    hints = body.get("hints", {})

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    RUNS[run_id] = {
        "status": "running",
        "queue": q,
        "result": None,
        "started_at": datetime.now().isoformat(),
        "filename": csv_name,
    }

    def engine_thread():
        import subprocess
        import time
        start_time = time.time()
        csv_path = str(DATA_DIR / csv_name)
        main_py  = str(BASE_DIR / "src" / "main.py")
        cmd = [PYTHON, main_py, csv_path,
               "--pop-size", str(pop_size),
               "--generations", str(generations),
               "--penalty-weight", str(penalty_weight),
               "--hints", json.dumps(hints)]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["PAUSE_FILE"] = str(DATA_DIR / f".pause_{run_id}")
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1, cwd=str(BASE_DIR), env=env
            )
            RUNS[run_id]["process"] = proc
            result_eq = None
            result_score = None
            current_score = 0.0
            for raw_line in proc.stdout:
                ev = parse_line(raw_line)
                if ev:
                    # 점수가 갱신되면 로컬에 저장
                    if "score" in ev and ev["score"] is not None:
                        current_score = ev["score"]
                        result_score = current_score
                        
                    # best_kernel 이벤트인데 점수가 없다면 직전 점수 주입
                    if ev["type"] == "best_kernel" and ev.get("score") is None:
                        ev["score"] = current_score
                        
                    if "equation" in ev and ev["equation"]:
                        result_eq = ev["equation"]
                    loop.call_soon_threadsafe(q.put_nowait, ev)
            
            if RUNS[run_id]["status"] == "cancelled":
                loop.call_soon_threadsafe(q.put_nowait, {"type": "cancelled"})
                return
                
            proc.wait()
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "raw": str(e)})

        # 만약 점수 갱신이 없었다면 기본값 처리
        if result_score is None: result_score = 100.0

        RUNS[run_id]["status"] = "done"
        RUNS[run_id]["result"] = {"equation": result_eq, "score": result_score}
        
        elapsed_time = time.time() - start_time
        append_to_history(
            run_id=run_id,
            started_at=RUNS[run_id]["started_at"],
            csv_name=csv_name,
            pop_size=pop_size,
            generations=generations,
            penalty_weight=penalty_weight,
            hints=hints,
            result_eq=result_eq,
            result_score=result_score,
            elapsed_time=elapsed_time
        )

        loop.call_soon_threadsafe(q.put_nowait,
            {"type": "done", "equation": result_eq, "score": result_score})

    threading.Thread(target=engine_thread, daemon=True).start()
    return {"run_id": run_id}

@app.get("/api/stream/{run_id}")
async def stream_run(run_id: str):
    """SSE 스트림 — GA 진행 상황 실시간 전송"""
    if run_id not in RUNS:
        return JSONResponse({"error": "not found"}, status_code=404)
    q = RUNS[run_id]["queue"]

    async def generator():
        while True:
            ev = await q.get()
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if ev.get("type") == "done":
                break

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/cancel/{run_id}")
async def cancel_run(run_id: str):
    if run_id in RUNS:
        RUNS[run_id]["status"] = "cancelled"
        if RUNS[run_id].get("process"):
            RUNS[run_id]["process"].kill()
            
        pause_file = DATA_DIR / f".pause_{run_id}"
        if pause_file.exists():
            pause_file.unlink()
        return {"status": "cancelled"}
    return JSONResponse({"error": "not found"}, status_code=404)

@app.post("/api/pause/{run_id}")
async def pause_run(run_id: str):
    if run_id in RUNS:
        (DATA_DIR / f".pause_{run_id}").touch(exist_ok=True)
        return {"status": "paused"}
    return JSONResponse({"error": "not found"}, status_code=404)

@app.post("/api/resume/{run_id}")
async def resume_run(run_id: str):
    if run_id in RUNS:
        pause_file = DATA_DIR / f".pause_{run_id}"
        if pause_file.exists():
            pause_file.unlink()
        return {"status": "resumed"}
    return JSONResponse({"error": "not found"}, status_code=404)

@app.get("/api/result/{run_id}")
async def get_result(run_id: str):
    if run_id not in RUNS:
        return JSONResponse({"error": "not found"}, status_code=404)
    return RUNS[run_id].get("result") or {}

@app.get("/api/history")
async def get_history():
    return _load_history()

@app.delete("/api/history/all")
async def clear_history():
    _save_history([])
    return {"status": "cleared"}

@app.delete("/api/history/{run_id}")
async def delete_history(run_id: str):
    hist = _load_history()
    hist = [h for h in hist if h["run_id"] != run_id]
    _save_history(hist)
    return {"status": "deleted"}
