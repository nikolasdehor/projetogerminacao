<div align="center">
  <img src="./static/banner.svg" alt="GerminaVision Banner" width="100%">

  #

  **O futuro do monitoramento agrícola chegou.** <br>
  *Sistema inteligente de visão computacional para detecção de germinação, contagem de folhas e análise de viabilidade comercial.*

  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org)
  [![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
  [![YOLO11](https://img.shields.io/badge/YOLO11-Computer%20Vision-yellow.svg)](https://ultralytics.com/)
  [![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM%20Chatbot-purple.svg)](https://openrouter.ai/)
</div>

<br>

## 🎯 O Desafio
Na agricultura e viveiros comerciais, estimar a taxa de germinação e o vigor das mudas (seedlings) é um processo exaustivo, manual e suscetível a erros. O **GerminaVision** automatiza essa inspeção com altíssima precisão usando IA.

## 🚀 Funcionalidades Principais

- **👁️ Visão Computacional de Ponta:** Detecta individualmente sementes germinadas, classifica a saúde da muda (`seedling`, `weak`, `noseedling`) e estima o número de folhas.
- **📊 Dashboard Temporal:** Registre lotes usando a etiqueta "Rótulo do dia" (ex: D0, D1, D3) e visualize a evolução do crescimento e a taxa de sucesso da bandeja em gráficos dinâmicos.
- **🤖 GerminaBot (IA Generativa):** Um chatbot nativo integrado ao modelo **LLama-3.1 120B** (via OpenRouter). Ele lê automaticamente os seus relatórios e gera insights complexos, estatísticas e dicas de cultivo.
- **🔌 Model Context Protocol (MCP):** Arquitetura pronta para plugar o banco de dados do GerminaVision diretamente no seu Cursor IDE ou Claude Desktop.

---

## 🛠️ Stack Tecnológica

| Camada | Tecnologia | Propósito |
|---|---|---|
| **Frontend** | HTML5, CSS3, Vanilla JS, Chart.js | Dashboard Dark-mode responsivo e renderização de dados. |
| **Backend** | Python, Flask, SQLite | API REST robusta, processamento paralelo e armazenamento. |
| **IA Visual** | Ultralytics YOLO11s | Treinado localmente para detectar até 6 classes de sementes/mudas. |
| **IA Texto** | OpenRouter API (LLMs) | Motor de raciocínio inteligente para respostas do GerminaBot. |

---

## ⚙️ Como Executar o Projeto

1. **Clone o repositório e instale as dependências:**
   ```bash
   git clone https://github.com/nikolasdehor/projeto-germinacao-visao-computacional.git
   cd projeto-germinacao-visao-computacional
   pip install -r requirements.txt
   ```

2. **Configure a IA (Opcional, mas recomendado):**
   Crie um arquivo `.env` na raiz do projeto contendo a sua chave do OpenRouter:
   ```env
   OPENROUTER_API_KEY=sk-or-v1-sua_chave_aqui
   ```

3. **Inicie o Servidor Local:**
   ```bash
   python run.py
   ```
   > 🌐 Acesse a aplicação em: [http://localhost:5001](http://localhost:5001)

---

## 🧠 Treinando o Seu Próprio Modelo

O projeto já acompanha a estrutura completa para fine-tuning local usando a GPU do Mac (MPS) ou CUDA.
Para treinar um modelo novo com o seu dataset:

```bash
python train.py
```
*O script foi otimizado para lidar com objetos minúsculos (`imgsz=640`) garantindo que as folhas mais finas não se percam na redução de resolução.*

---

## 🤝 Protocolo MCP
Se você utiliza assistentes como o Claude Desktop, siga nosso guia interno no painel (`/mcp`) para configurar o `claude_desktop_config.json`. Com isso, a sua IA de mesa terá acesso direto e em tempo real a todas as métricas colhidas pelo YOLO11 no banco de dados.

<br>
<div align="center">
  Feito com 💚 para a feira de tecnologia.
</div>
