"""
Cortador automático de vídeos com IA - YouTube & TikTok
Novidades: Transcrição Whisper + Análise de visão (Claude) para detetar momentos relevantes

Dependências:
    pip install yt-dlp opencv-python ffmpeg-python openai-whisper torch anthropic Pillow

Requer: ffmpeg instalado no sistema
"""

import os
import shutil
import subprocess
import json
import base64
import time
from pathlib import Path
from dataclasses import dataclass, field

try:
    import yt_dlp
    YT_DLP_OK = True
except ImportError:
    YT_DLP_OK = False

try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import whisper
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False

try:
    import anthropic
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False


# ─────────────────────────────────────────────
# ESTRUTURA DE DADOS
# ─────────────────────────────────────────────

@dataclass
class Segmento:
    """Segmento de vídeo com metadados das três camadas de IA."""
    inicio: float
    fim: float
    duracao: float = 0.0
    score_movimento: float = 0.0   # OpenCV
    score_visao: float = 0.0       # Claude Vision
    score_audio: float = 0.0       # Whisper
    score_final: float = 0.0       # Ponderado
    transcricao: str = ""
    descricao_visual: str = ""
    razao_relevancia: str = ""
    frames_chave: list = field(default_factory=list)

    def __post_init__(self):
        self.duracao = round(self.fim - self.inicio, 2)


# ─────────────────────────────────────────────
# 1. DOWNLOAD DO VÍDEO
# ─────────────────────────────────────────────

def baixar_video(url: str, pasta_saida: str = "downloads") -> str:
    """Descarrega o vídeo da URL (YouTube ou TikTok) sem marca de água."""
    os.makedirs(pasta_saida, exist_ok=True)

    # Se for ficheiro local, devolve directamente
    if os.path.exists(url):
        print(f"✅ Ficheiro local detectado: {url}")
        return url

    opcoes = {
        "outtmpl": f"{pasta_saida}/%(title)s.%(ext)s",
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "http_headers": {
            "User-Agent": "TikTok 26.2.0 rv:262018 (iPhone; iOS 14.4.2; en_US) Cronet"
        },
        "quiet": True,
        "no_warnings": True,
    }
    if not YT_DLP_OK:
        print("⚠️  yt-dlp não instalado. Execute: pip install yt-dlp")
        return ""
    try:
        with yt_dlp.YoutubeDL(opcoes) as ydl:
            info = ydl.extract_info(url, download=True)
            ficheiro = ydl.prepare_filename(info).replace(".webm", ".mp4")
            print(f"✅ Descarregado: {ficheiro}")
            return ficheiro
    except Exception as e:
        print(f"❌ Erro no download: {e}")
        return ""


# ─────────────────────────────────────────────
# 2. EXTRAÇÃO DE ÁUDIO
# ─────────────────────────────────────────────

def extrair_audio(caminho_video: str, pasta_temp: str = "temp") -> str:
    """Extrai áudio para WAV 16kHz mono — formato ideal para Whisper."""
    os.makedirs(pasta_temp, exist_ok=True)
    saida = os.path.join(pasta_temp, Path(caminho_video).stem + "_audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", caminho_video,
        "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        saida, "-loglevel", "error"
    ]
    resultado = subprocess.run(cmd, capture_output=True)
    if resultado.returncode == 0:
        print(f"✅ Áudio extraído: {saida}")
        return saida
    print("❌ Erro ao extrair áudio — verifica se o ffmpeg está instalado")
    return ""


# ─────────────────────────────────────────────
# 3. TRANSCRIÇÃO COM WHISPER
# ─────────────────────────────────────────────

def transcrever_audio(caminho_audio: str, modelo: str = "base") -> list[dict]:
    """
    Transcreve com Whisper e devolve segmentos com timestamps e score de relevância.
    - Deteta idioma automaticamente (PT, EN, ES, ...)
    - Score de áudio: velocidade de fala + bónus por palavras-chave virais
    """
    if not WHISPER_OK:
        print("⚠️  Whisper não instalado. Execute: pip install openai-whisper")
        return []
    if not caminho_audio or not os.path.exists(caminho_audio):
        print("⚠️  Ficheiro de áudio não encontrado")
        return []

    print(f"🎙️  A transcrever com Whisper (modelo: {modelo})...")
    try:
        modelo_wh = whisper.load_model(modelo)
        resultado = modelo_wh.transcribe(
            caminho_audio, word_timestamps=True, verbose=False, language=None
        )
    except Exception as e:
        print(f"❌ Erro na transcrição Whisper: {e}")
        return []

    KEYWORDS = {
        "pt": ["incrível", "impressionante", "top", "melhor", "pior", "nunca", "sempre",
               "segredo", "revelar", "primeiro", "exclusivo", "urgente"],
        "en": ["amazing", "incredible", "best", "worst", "secret", "never", "always",
               "first", "breaking", "viral", "shocking", "exclusive"],
        "es": ["increíble", "mejor", "peor", "nunca", "siempre", "secreto",
               "primero", "exclusivo", "urgente"],
    }
    idioma = resultado.get("language", "en")
    keywords = KEYWORDS.get(idioma, KEYWORDS["en"])

    segmentos = []
    for seg in resultado.get("segments", []):
        palavras = seg.get("words", [])
        velocidade = len(palavras) / max(seg["end"] - seg["start"], 0.1)
        bonus = sum(1 for kw in keywords if kw in seg["text"].lower())
        score = min(100.0, velocidade * 10 + bonus * 15)
        segmentos.append({
            "inicio": round(seg["start"], 2),
            "fim": round(seg["end"], 2),
            "texto": seg["text"].strip(),
            "score_audio": round(score, 2),
            "idioma": idioma
        })

    print(f"✅ {len(segmentos)} segmentos | Idioma: {idioma}")
    return segmentos


def agrupar_segmentos_whisper(
    segmentos: list[dict], duracao_min: float = 10.0, duracao_max: float = 60.0
) -> list[dict]:
    """Agrupa segmentos Whisper em janelas com duração ideal para clips."""
    if not segmentos:
        return []
    grupos, buffer = [], []
    inicio_grupo = segmentos[0]["inicio"]

    for seg in segmentos:
        buffer.append(seg)
        duracao_atual = seg["fim"] - inicio_grupo
        if duracao_atual >= duracao_min:
            if duracao_atual >= duracao_max or seg is segmentos[-1]:
                grupos.append({
                    "inicio": inicio_grupo,
                    "fim": seg["fim"],
                    "texto": " ".join(s["texto"] for s in buffer),
                    "score_audio": round(
                        sum(s["score_audio"] for s in buffer) / len(buffer), 2
                    )
                })
                buffer = []
                if seg is not segmentos[-1]:
                    inicio_grupo = seg["fim"]
    return grupos


# ─────────────────────────────────────────────
# 4. EXTRAÇÃO DE FRAMES CHAVE
# ─────────────────────────────────────────────

def extrair_frames_chave(
    caminho_video: str, inicio: float, fim: float,
    num_frames: int = 3, pasta_temp: str = "temp/frames"
) -> list[str]:
    """Extrai frames uniformemente distribuídos num segmento via ffmpeg."""
    os.makedirs(pasta_temp, exist_ok=True)
    frames = []
    duracao = fim - inicio
    timestamps = [
        inicio + (duracao / (num_frames + 1)) * i
        for i in range(1, num_frames + 1)
    ]
    for i, ts in enumerate(timestamps):
        nome = os.path.join(pasta_temp, f"frame_{inicio:.0f}_{i}.jpg")
        cmd = [
            "ffmpeg", "-y", "-ss", str(ts), "-i", caminho_video,
            "-vframes", "1", "-q:v", "2", "-vf", "scale=640:-1",
            nome, "-loglevel", "error"
        ]
        if subprocess.run(cmd, capture_output=True).returncode == 0 and os.path.exists(nome):
            frames.append(nome)
    return frames


def _frame_para_base64(caminho: str) -> str:
    with open(caminho, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


# ─────────────────────────────────────────────
# 5. ANÁLISE DE VISÃO COM CLAUDE
# ─────────────────────────────────────────────

def analisar_frames_com_ia(
    frames: list[str], transcricao: str = "", api_key: str = ""
) -> dict:
    """
    Envia até 3 frames para Claude analisar visualmente o segmento.
    Avalia: interesse visual, emoção, dinamismo, potencial viral, adequação TikTok.
    Devolve score (0-100), descrição e razão de relevância.
    """
    if not ANTHROPIC_OK:
        print("⚠️  anthropic não instalado. Execute: pip install anthropic")
        return {"score_visao": 50.0, "descricao": "Análise indisponível", "razao": ""}
    if not frames:
        return {"score_visao": 0.0, "descricao": "Sem frames", "razao": ""}

    chave = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not chave:
        print("⚠️  ANTHROPIC_API_KEY não definida. A saltar análise de visão.")
        return {"score_visao": 50.0, "descricao": "Chave API ausente", "razao": ""}

    cliente = anthropic.Anthropic(api_key=chave)
    conteudo = []

    for i, frame in enumerate(frames[:3]):
        conteudo.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _frame_para_base64(frame)
            }
        })
        conteudo.append({"type": "text", "text": f"Frame {i+1} do segmento"})

    contexto = f'\n\nTranscrição do áudio: "{transcricao}"' if transcricao else ""
    conteudo.append({"type": "text", "text": f"""Analisa estes frames de um segmento de vídeo para redes sociais.{contexto}

Critérios de avaliação:
1. Interesse visual (cores, composição, ação em curso)
2. Expressões faciais e emoção transmitida
3. Dinamismo e movimento no plano
4. Relevância e contexto para redes sociais
5. Potencial viral do momento captado

Responde APENAS em JSON válido, sem texto adicional:
{{
  "score": <número inteiro de 0 a 100>,
  "descricao": "<descrição curta do que acontece visualmente>",
  "razao": "<por que este segmento é ou não adequado para publicar>",
  "emocao_dominante": "<alegria|surpresa|tensão|neutral|tristeza|humor>",
  "adequado_tiktok": <true|false>
}}"""})

    try:
        resposta = cliente.messages.create(
            model="claude-opus-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": conteudo}]
        )
        texto = resposta.content[0].text.strip()
        if "```" in texto:
            partes = texto.split("```")
            texto = partes[1] if len(partes) > 1 else partes[0]
            if texto.startswith("json"):
                texto = texto[4:]
        dados = json.loads(texto.strip())
        return {
            "score_visao": float(dados.get("score", 50)),
            "descricao": dados.get("descricao", ""),
            "razao": dados.get("razao", ""),
            "emocao": dados.get("emocao_dominante", "neutral"),
            "adequado_tiktok": dados.get("adequado_tiktok", True)
        }
    except Exception as e:
        print(f"⚠️  Erro na análise de visão: {e}")
        return {"score_visao": 50.0, "descricao": "Erro na análise", "razao": str(e)}


# ─────────────────────────────────────────────
# 6. DETEÇÃO DE MOVIMENTO (OpenCV)
# ─────────────────────────────────────────────

def analisar_movimento_segmento(
    caminho_video: str, inicio: float, fim: float, amostras: int = 10
) -> float:
    """
    Mede intensidade de movimento num segmento via diferença de frames (OpenCV).
    Devolve score normalizado de 0 a 100.
    """
    if not CV2_OK:
        return 50.0
    try:
        video = cv2.VideoCapture(caminho_video)
        fps = video.get(cv2.CAP_PROP_FPS)
        f_inicio, f_fim = int(inicio * fps), int(fim * fps)
        total = f_fim - f_inicio
        if total <= 0:
            video.release()
            return 0.0

        passo = max(1, total // amostras)
        diffs, frame_ant = [], None

        for i in range(f_inicio, f_fim, passo):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = video.read()
            if not ok:
                break
            cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_ant is not None:
                diffs.append(float(np.mean(cv2.absdiff(cinza, frame_ant))))
            frame_ant = cinza

        video.release()
        return round(min(100.0, float(np.mean(diffs)) * 3), 2) if diffs else 0.0
    except Exception as e:
        print(f"⚠️  Erro na análise de movimento: {e}")
        return 50.0


# ─────────────────────────────────────────────
# 7. ANÁLISE COMPLETA DOS SEGMENTOS COM IA
# ─────────────────────────────────────────────

def analisar_segmentos_com_ia(
    caminho_video: str,
    segmentos_candidatos: list[dict],
    api_key: str = "",
    peso_visao: float = 0.5,
    peso_audio: float = 0.3,
    peso_movimento: float = 0.2,
) -> list[Segmento]:
    """
    Analisa cada segmento candidato com três camadas de IA:
      👁️  Claude Vision → relevância visual, emoção, potencial viral
      🎙️  Whisper       → densidade de fala, keywords virais
      🏃  OpenCV        → intensidade de movimento entre frames

    Score final = soma ponderada dos três scores.
    Devolve lista de Segmentos ordenados por score_final (maior primeiro).
    """
    resultados = []
    total = len(segmentos_candidatos)
    print(f"\n🤖 A analisar {total} segmentos com IA...")
    print(f"   Pesos → Visão: {peso_visao*100:.0f}% | Áudio: {peso_audio*100:.0f}% | Movimento: {peso_movimento*100:.0f}%\n")

    for idx, cand in enumerate(segmentos_candidatos, 1):
        inicio, fim = cand["inicio"], cand["fim"]
        print(f"📍 [{idx}/{total}] {inicio:.1f}s → {fim:.1f}s", end="  ")

        seg = Segmento(inicio=inicio, fim=fim)
        seg.transcricao = cand.get("texto", "")
        seg.score_audio = cand.get("score_audio", 0.0)

        # OpenCV — movimento
        seg.score_movimento = analisar_movimento_segmento(caminho_video, inicio, fim)

        # Frames para análise de visão
        seg.frames_chave = extrair_frames_chave(caminho_video, inicio, fim, num_frames=3)

        # Claude Vision — análise visual
        analise = analisar_frames_com_ia(
            frames=seg.frames_chave,
            transcricao=seg.transcricao,
            api_key=api_key
        )
        seg.score_visao = analise.get("score_visao", 50.0)
        seg.descricao_visual = analise.get("descricao", "")
        seg.razao_relevancia = analise.get("razao", "")

        # Score final ponderado
        seg.score_final = round(
            seg.score_visao * peso_visao +
            seg.score_audio * peso_audio +
            seg.score_movimento * peso_movimento, 2
        )
        print(f"👁️ {seg.score_visao:.0f}  🎙️ {seg.score_audio:.0f}  🏃 {seg.score_movimento:.0f}  →  ⭐ {seg.score_final:.0f}")
        resultados.append(seg)

        if api_key and idx < total:
            time.sleep(0.5)  # respeita rate limit da API

    resultados.sort(key=lambda s: s.score_final, reverse=True)
    return resultados


# ─────────────────────────────────────────────
# 8. CORTAR O VÍDEO COM FFMPEG
# ─────────────────────────────────────────────

def cortar_cena(
    caminho_video: str, inicio: float, fim: float, saida: str,
    formato: str = "vertical", remover_watermark: bool = True
) -> bool:
    """Corta e exporta um segmento em formato vertical (9:16) ou horizontal (16:9)."""
    os.makedirs(os.path.dirname(saida) or ".", exist_ok=True)
    filtros = []
    if formato == "vertical":
        filtros += ["crop=ih*9/16:ih:(iw-ih*9/16)/2:0", "scale=1080:1920"]
    else:
        filtros += [
            "scale=1920:1080:force_original_aspect_ratio=decrease",
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        ]

    # delogo é opcional — nem todos os builds de ffmpeg o têm
    if remover_watermark:
        result_test = subprocess.run(
            ["ffmpeg", "-filters"], capture_output=True, text=True
        )
        if "delogo" in result_test.stdout + result_test.stderr:
            filtros.append("delogo=x=W-200:y=H-100:w=200:h=100:show=0")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(inicio), "-to", str(fim),
        "-i", caminho_video,
        "-vf", ",".join(filtros),
        "-c:v", "libx264", "-c:a", "aac",
        "-preset", "fast", "-crf", "23",
        saida, "-loglevel", "error"
    ]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode == 0:
        print(f"✅ Corte guardado: {saida}")
        return True
    print(f"❌ Erro ao cortar: {resultado.stderr[-300:]}")
    return False


# ─────────────────────────────────────────────
# 9. RELATÓRIO DE ANÁLISE
# ─────────────────────────────────────────────

def gerar_relatorio(segmentos: list[Segmento], pasta_saida: str) -> str:
    """Exporta relatório JSON detalhado com scores e metadados de todos os segmentos."""
    relatorio = {
        "total_segmentos_analisados": len(segmentos),
        "segmentos": [{
            "rank": i,
            "inicio_s": s.inicio,
            "fim_s": s.fim,
            "duracao_s": s.duracao,
            "scores": {
                "visao_ia": s.score_visao,
                "audio_whisper": s.score_audio,
                "movimento_opencv": s.score_movimento,
                "final_ponderado": s.score_final
            },
            "transcricao": s.transcricao,
            "descricao_visual": s.descricao_visual,
            "razao_relevancia": s.razao_relevancia
        } for i, s in enumerate(segmentos, 1)]
    }
    caminho = os.path.join(pasta_saida, "relatorio_ia.json")
    os.makedirs(pasta_saida, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    print(f"📄 Relatório guardado: {caminho}")
    return caminho


# ─────────────────────────────────────────────
# 10. PIPELINE COMPLETO COM IA
# ─────────────────────────────────────────────

def processar_video_com_ia(
    url: str,
    num_cortes: int = 5,
    formato: str = "vertical",
    pasta_saida: str = "cortes",
    modelo_whisper: str = "base",
    api_key: str = "",
    duracao_min_clip: float = 10.0,
    duracao_max_clip: float = 60.0,
    peso_visao: float = 0.5,
    peso_audio: float = 0.3,
    peso_movimento: float = 0.2,
) -> list[str]:
    """
    Pipeline completo com IA em 5 passos:

      1. Download       — yt-dlp (YouTube + TikTok sem watermark) ou ficheiro local
      2. Whisper        — extração de áudio + transcrição + scoring de áudio
      3. Análise IA     — Claude Vision (visual) + OpenCV (movimento)
      4. Cortes         — ffmpeg com crop/resize para o formato certo
      5. Relatório      — JSON com scores e metadados de todos os segmentos
    """
    print("\n" + "=" * 55)
    print("  🤖 CORTADOR DE VÍDEOS COM IA")
    print("     Whisper + Visão Claude + OpenCV")
    print("=" * 55)
    tem_api = bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))
    print(f"\n🎬 Fonte   : {url}")
    print(f"📐 Formato : {formato}  |  ✂️  Cortes: {num_cortes}")
    print(f"🎙️  Whisper : {modelo_whisper}  |  👁️  Visão IA: {'✅' if tem_api else '❌ sem chave'}")

    pasta_temp = "temp"
    os.makedirs(pasta_temp, exist_ok=True)
    os.makedirs(pasta_saida, exist_ok=True)

    # ── PASSO 1: Download ────────────────────────────────
    print("\n── PASSO 1: Download ──────────────────────────────")
    caminho_video = baixar_video(url)
    if not caminho_video or not os.path.exists(caminho_video):
        print("❌ Falha no download ou ficheiro não encontrado")
        return []

    # ── PASSO 2: Transcrição Whisper ────────────────────
    print("\n── PASSO 2: Transcrição Whisper ───────────────────")
    caminho_audio = extrair_audio(caminho_video, pasta_temp)
    segs_whisper = transcrever_audio(caminho_audio, modelo_whisper)
    candidatos = agrupar_segmentos_whisper(
        segs_whisper, duracao_min=duracao_min_clip, duracao_max=duracao_max_clip
    )

    # Fallback: janelas fixas se Whisper não gerar segmentos
    if not candidatos:
        print("⚠️  Sem segmentos Whisper — a usar janelas fixas como fallback...")
        if CV2_OK:
            cap = cv2.VideoCapture(caminho_video)
            fps_v = cap.get(cv2.CAP_PROP_FPS)
            frames_v = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duracao_total = frames_v / max(fps_v, 1)
            cap.release()
        else:
            duracao_total = 300
        candidatos = [
            {"inicio": t, "fim": min(t + duracao_max_clip, duracao_total),
             "texto": "", "score_audio": 0.0}
            for t in range(0, int(duracao_total), int(duracao_min_clip))
        ]

    print(f"📋 {len(candidatos)} segmentos candidatos")

    # ── PASSO 3: Análise com IA ─────────────────────────
    print("\n── PASSO 3: Análise com IA ────────────────────────")
    analisados = analisar_segmentos_com_ia(
        caminho_video=caminho_video,
        segmentos_candidatos=candidatos,
        api_key=api_key,
        peso_visao=peso_visao,
        peso_audio=peso_audio,
        peso_movimento=peso_movimento,
    )

    # ── PASSO 4: Cortes ─────────────────────────────────
    print(f"\n── PASSO 4: Criar {num_cortes} cortes ─────────────────────")
    melhores = analisados[:num_cortes]
    nome_base = Path(caminho_video).stem
    ficheiros = []

    for i, seg in enumerate(melhores, 1):
        saida = os.path.join(
            pasta_saida, f"{nome_base}_clip_{i:02d}_score{seg.score_final:.0f}.mp4"
        )
        print(f"\n✂️  Clip {i}: {seg.inicio:.1f}s → {seg.fim:.1f}s  (⭐ {seg.score_final:.1f})")
        if seg.descricao_visual:
            print(f"   👁️  {seg.descricao_visual}")
        if seg.transcricao:
            trecho = seg.transcricao[:80]
            print(f"   🎙️  \"{trecho}{'...' if len(seg.transcricao) > 80 else ''}\"")
        if cortar_cena(caminho_video, seg.inicio, seg.fim, saida, formato):
            ficheiros.append(saida)

    # ── PASSO 5: Relatório ───────────────────────────────
    print("\n── PASSO 5: Relatório ─────────────────────────────")
    gerar_relatorio(analisados, pasta_saida)

    # Limpeza temporários
    shutil.rmtree(pasta_temp, ignore_errors=True)

    print(f"\n{'='*55}")
    print(f"🎉 CONCLUÍDO! {len(ficheiros)} clips em '{pasta_saida}/'")
    print(f"{'='*55}\n")
    return ficheiros


# ─────────────────────────────────────────────
# 11. DOWNLOAD DE VÍDEO COMPLETO (sem cortes)
# ─────────────────────────────────────────────

def _ler_segundos(prompt: str, padrao: int) -> int:
    """Lê segundos ou formato hh:mm:ss / mm:ss. Ex: '1:30:00' = 5400s."""
    val = input(prompt).strip()
    if not val:
        return padrao
    partes = val.split(":")
    try:
        if len(partes) == 3:
            return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])
        elif len(partes) == 2:
            return int(partes[0]) * 60 + int(partes[1])
        else:
            return int(val)
    except ValueError:
        return padrao


def _formatar_duracao(segundos: float) -> str:
    """Converte segundos para string legível."""
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _obter_duracao_video(caminho_video: str) -> float:
    """Obtém a duração total do vídeo em segundos via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        caminho_video
    ]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(resultado.stdout.strip())
    except ValueError:
        return 0.0


def baixar_video_completo(
    url: str,
    formato: str = "horizontal",
    pasta_saida: str = "completos",
    inicio: float = 0.0,
    fim: float = 0.0,
    remover_watermark: bool = True,
) -> str:
    """
    Descarrega e exporta o vídeo completo (ou um intervalo específico) sem cortes de cena.
    Suporta URL (YouTube/TikTok) ou caminho de ficheiro local.
    """
    print("\n" + "=" * 55)
    print("  📥 DOWNLOAD DE VÍDEO COMPLETO")
    print("=" * 55)

    print("\n⬇️  A obter o vídeo...")
    caminho_video = baixar_video(url, pasta_saida="downloads")
    if not caminho_video or not os.path.exists(caminho_video):
        print("❌ Falha no download ou ficheiro não encontrado")
        return ""

    duracao_total = _obter_duracao_video(caminho_video)
    fim_real = fim if fim > 0 else duracao_total
    if inicio >= fim_real:
        print(f"⚠️  Intervalo inválido ({inicio}s → {fim_real}s). A usar vídeo completo.")
        inicio, fim_real = 0.0, duracao_total

    print(f"📏 Duração total  : {_formatar_duracao(duracao_total)}")
    print(f"✂️  Intervalo      : {_formatar_duracao(inicio)} → {_formatar_duracao(fim_real)}")
    print(f"📐 Formato        : {formato}")

    os.makedirs(pasta_saida, exist_ok=True)
    nome_base = Path(caminho_video).stem
    sufixo = (
        f"_{_formatar_duracao(inicio).replace(' ','')}-{_formatar_duracao(fim_real).replace(' ','')}"
        if (inicio > 0 or fim > 0) else "_completo"
    )
    saida = os.path.join(pasta_saida, f"{nome_base}{sufixo}.mp4")

    filtros = []
    if formato == "vertical":
        filtros += ["crop=ih*9/16:ih:(iw-ih*9/16)/2:0", "scale=1080:1920"]
    else:
        filtros += [
            "scale=1920:1080:force_original_aspect_ratio=decrease",
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        ]
    if remover_watermark:
        result_test = subprocess.run(
            ["ffmpeg", "-filters"], capture_output=True, text=True
        )
        if "delogo" in result_test.stdout + result_test.stderr:
            filtros.append("delogo=x=W-200:y=H-100:w=200:h=100:show=0")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(inicio),
        "-to", str(fim_real),
        "-i", caminho_video,
        "-vf", ",".join(filtros),
        "-c:v", "libx264", "-c:a", "aac",
        "-preset", "fast", "-crf", "23",
        saida, "-loglevel", "error"
    ]

    print(f"\n⚙️  A processar... (pode demorar para vídeos longos)")
    resultado = subprocess.run(cmd, capture_output=True, text=True)

    if resultado.returncode == 0:
        tamanho_mb = os.path.getsize(saida) / (1024 * 1024)
        print(f"\n✅ Vídeo guardado : {saida}")
        print(f"📦 Tamanho        : {tamanho_mb:.1f} MB")
        return saida
    else:
        print(f"❌ Erro ao processar: {resultado.stderr[-300:]}")
        return ""


# ─────────────────────────────────────────────
# 12. MENU DE LINHA DE COMANDOS
# ─────────────────────────────────────────────

def menu():
    print("=" * 55)
    print("  🤖 CORTADOR DE VÍDEOS COM IA")
    print("     Whisper + Visão Claude + OpenCV")
    print("=" * 55)

    print("\n📦 Dependências:")
    for nome, ok in [("yt-dlp", YT_DLP_OK), ("opencv", CV2_OK),
                     ("whisper", WHISPER_OK), ("anthropic", ANTHROPIC_OK)]:
        print(f"   {'✅' if ok else '❌'} {nome}")

    print()
    url = input("🔗 Cola o link do vídeo (ou caminho do ficheiro local): ").strip()
    if not url:
        print("❌ URL ou caminho inválido")
        return

    print("\n🎬 O que queres fazer?")
    print("  1. ✂️  Cortes automáticos com IA  (escolhe os melhores momentos)")
    print("  2. 📥 Vídeo completo              (descarrega o vídeo inteiro ou um intervalo)")
    modo = input("Escolha (1/2): ").strip()

    print("\n📐 Formato de saída:")
    print("  1. Horizontal — YouTube / MP4 normal  [padrão]")
    print("  2. Vertical   — TikTok / Reels / Shorts")
    formato = "vertical" if input("Escolha (1/2): ").strip() == "2" else "horizontal"

    # ── MODO 2: VÍDEO COMPLETO ───────────────────────────
    if modo == "2":
        print("\n⏱️  Intervalo a capturar (Enter = vídeo completo):")
        print("   (Podes escrever hh:mm:ss, mm:ss ou segundos)")
        inicio = _ler_segundos("   Início [0]: ", 0)
        fim    = _ler_segundos("   Fim    [fim do vídeo]: ", 0)

        if inicio == 0 and fim == 0:
            print("   ✅ Vídeo completo do início ao fim")
        else:
            print(f"   ✅ De {_formatar_duracao(inicio)} até {_formatar_duracao(fim) if fim > 0 else 'fim do vídeo'}")

        baixar_video_completo(
            url=url,
            formato=formato,
            pasta_saida="completos",
            inicio=float(inicio),
            fim=float(fim),
        )
        return

    # ── MODO 1: CORTES AUTOMÁTICOS COM IA ───────────────
    num = input("\n✂️  Quantos clips criar? [5]: ").strip()
    num_cortes = int(num) if num.isdigit() else 5

    print("\n🎙️  Modelo Whisper:")
    for k, v in [("1", "tiny   — rápido, menos preciso"),
                 ("2", "base   — equilíbrio [padrão]"),
                 ("3", "small  — bom equilíbrio"),
                 ("4", "medium — muito preciso"),
                 ("5", "large  — máxima precisão, lento")]:
        print(f"  {k}. {v}")
    modelos = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large"}
    modelo = modelos.get(input("Escolha (1-5): ").strip(), "base")

    print("\n⏱️  Duração de cada clip:")
    print("  1. TikTok curto   — 10s a 30s")
    print("  2. TikTok médio   — 20s a 60s  [padrão]")
    print("  3. Reels / Shorts — 15s a 90s")
    print("  4. YouTube clip   — 30s a 3 min")
    print("  5. Longo          — 5 min a 30 min")
    print("  6. Personalizado")
    preset = input("Escolha (1-6): ").strip()

    presets = {
        "1": (10, 30),
        "2": (20, 60),
        "3": (15, 90),
        "4": (30, 180),
        "5": (300, 1800),
    }

    if preset in presets:
        duracao_min, duracao_max = presets[preset]
        print(f"   ✅ {_formatar_duracao(duracao_min)} mínimo  |  {_formatar_duracao(duracao_max)} máximo")
    else:
        print("   (Podes escrever hh:mm:ss, mm:ss ou segundos)")
        duracao_min = _ler_segundos("   Duração mínima [10s]: ", 10)
        duracao_max = _ler_segundos("   Duração máxima [60s]: ", 60)
        if duracao_min >= duracao_max:
            print("   ⚠️  Mínimo >= máximo — a usar valores padrão (10s / 60s)")
            duracao_min, duracao_max = 10, 60
        print(f"   ✅ {_formatar_duracao(duracao_min)} mínimo  |  {_formatar_duracao(duracao_max)} máximo")

    api_key = input("\n🔑 Chave Anthropic API (Enter = usa ANTHROPIC_API_KEY): ").strip()

    processar_video_com_ia(
        url=url,
        num_cortes=num_cortes,
        formato=formato,
        modelo_whisper=modelo,
        api_key=api_key or "",
        duracao_min_clip=float(duracao_min),
        duracao_max_clip=float(duracao_max),
    )


if __name__ == "__main__":
    menu()
