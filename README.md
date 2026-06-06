#  Cortador Automático de Vídeos com IA

> Corta automaticamente os melhores momentos de vídeos do YouTube e TikTok usando **Whisper + Claude Vision + OpenCV**.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Anthropic](https://img.shields.io/badge/Claude-Vision-D97757?style=flat&logo=anthropic&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-required-007808?style=flat&logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

##  Descrição

Este projecto usa três camadas de inteligência artificial para identificar e cortar automaticamente os momentos mais relevantes de qualquer vídeo:

| Camada | Tecnologia | O que analisa |
|--------|-----------|---------------|
|  Visão | Claude Vision (Anthropic) | Interesse visual, emoção, potencial viral |
|  Áudio | OpenAI Whisper | Transcrição, velocidade de fala, keywords virais |
|  Movimento | OpenCV | Intensidade de movimento entre frames |

O score final de cada segmento é calculado com pesos configuráveis das três camadas, garantindo os cortes mais relevantes para redes sociais.

---

##  Funcionalidades

-  Download de vídeos do **YouTube e TikTok** (sem marca de água)
-  **Transcrição automática** com deteção de idioma (PT, EN, ES, ...)
-  **Análise visual com IA** (Claude Vision) — emoção, dinamismo, potencial viral
-  **Deteção de movimento** via OpenCV
-  Exportação em formato **vertical (9:16)** ou **horizontal (16:9)**
-  Download do **vídeo completo** ou de um intervalo específico
-  **Relatório JSON** detalhado com scores de todos os segmentos
-  Menu interativo de linha de comandos

---

##  Requisitos

### Sistema
- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) instalado e no PATH

### Instalar FFmpeg

**Windows:**
```bash
winget install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

---

##  Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/daltonmavungo/cortador-video-ia.git
cd cortador-video-ia

# 2. Criar ambiente virtual (recomendado)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt
```

---

##  Configurar a API Key

Para activar a análise de visão com Claude, precisas de uma chave da [Anthropic](https://console.anthropic.com/).

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# Linux/macOS
export ANTHROPIC_API_KEY=sk-ant-...
```

> **Nota:** O projecto funciona sem a chave API, mas a análise visual será substituída por um score padrão de 50.

---

##  Como usar

```bash
python main.py
```

O menu interativo vai guiar-te pelas opções:

```
=======================================================
   CORTADOR DE VÍDEOS COM IA
     Whisper + Visão Claude + OpenCV
=======================================================

 Dependências:
    yt-dlp
    opencv
    whisper
    anthropic

 Cola o link do vídeo: https://youtube.com/watch?v=...

 O que queres fazer?
  1.   Cortes automáticos com IA
  2.  Vídeo completo
```

---

##  Estrutura do Projecto

```
cortador-video-ia/
│
├── main.py              # Código principal + menu CLI
├── requirements.txt     # Dependências Python
├── .gitignore           # Ficheiros ignorados pelo Git
├── README.md            # Esta documentação
│
├── downloads/           # Vídeos descarregados (gerado automaticamente)
├── cortes/              # Clips exportados + relatório JSON
├── completos/           # Vídeos completos exportados
└── temp/                # Ficheiros temporários (apagados após uso)
```

---

##  Como funciona o Score

Cada segmento recebe um **score final ponderado** (0–100):

```
Score Final = (Score Visual × 50%) + (Score Áudio × 30%) + (Score Movimento × 20%)
```

| Score | Classificação |
|-------|--------------|
| 80–100 |  Excelente — publicar imediatamente |
| 60–79  |  Bom — vale a pena publicar |
| 40–59  |  Médio — considerar edição adicional |
| 0–39   |  Fraco — evitar |

---

##  Relatório de Análise

Após o processamento é gerado um ficheiro `cortes/relatorio_ia.json`:

```json
{
  "total_segmentos_analisados": 12,
  "segmentos": [
    {
      "rank": 1,
      "inicio_s": 45.2,
      "fim_s": 98.7,
      "duracao_s": 53.5,
      "scores": {
        "visao_ia": 87.0,
        "audio_whisper": 72.0,
        "movimento_opencv": 65.0,
        "final_ponderado": 79.1
      },
      "transcricao": "Este é o momento mais incrível...",
      "descricao_visual": "Pessoa com expressão de surpresa, cores vibrantes",
      "razao_relevancia": "Alto potencial viral — emoção forte e boa composição visual"
    }
  ]
}
```

---

##  Parâmetros Avançados

Podes chamar o pipeline directamente no código:

```python
from main import processar_video_com_ia

processar_video_com_ia(
    url="https://youtube.com/watch?v=...",
    num_cortes=5,
    formato="vertical",        # 'vertical' | 'horizontal'
    modelo_whisper="base",     # 'tiny' | 'base' | 'small' | 'medium' | 'large'
    api_key="sk-ant-...",
    duracao_min_clip=15.0,     # segundos
    duracao_max_clip=60.0,     # segundos
    peso_visao=0.5,
    peso_audio=0.3,
    peso_movimento=0.2,
)
```

---

##  Contribuições

Contribuições são bem-vindas! Para contribuir:

1. Faz fork do repositório
2. Cria uma branch: `git checkout -b feature/nova-funcionalidade`
3. Faz commit: `git commit -m "feat: adicionar nova funcionalidade"`
4. Push: `git push origin feature/nova-funcionalidade`
5. Abre um Pull Request

---

##  Contacto

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/josedaltonmavungo/)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=flat&logo=gmail&logoColor=white)](mailto:josedalton258@gmail.com)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-25D366?style=flat&logo=whatsapp&logoColor=white)](https://wa.me/244946822633)

---

##  Licença

Este projecto está licenciado sob a licença MIT — vê o ficheiro [LICENSE](LICENSE) para mais detalhes.

---

<p align="center">Feito com amor por <a href="https://github.com/daltonmavungo">Dalton Mavungo</a> 🇦🇴</p>
