# Sistema de Visão Computacional para Monitoramento de Germinação e Crescimento de Mudas
## 1. Visão geral e motivação
O monitoramento de germinação e crescimento inicial de mudas ainda é frequentemente realizado de forma manual, por inspeção visual de bandejas ou vasos, o que é trabalhoso, subjetivo e pouco escalável. Com o avanço de visão computacional e IA, já existem sistemas capazes de medir altura, área foliar e estágio de crescimento automaticamente, inclusive em estufas, fazendas verticais e cultivos hidropônicos.[^1][^2][^3][^4][^5]

Este projeto propõe um sistema de visão computacional focado em **mudas de plantas (hortaliças, ervas, ornamentais etc.)**, capaz de:

- Determinar automaticamente se uma cavidade/vaso efetivamente germinou.
- Estimar o número de folhas/galhos (ramificações visíveis) de cada muda.
- Monitorar a evolução desses indicadores ao longo do tempo.

A ideia é combinar modelos de detecção de mudas (seedlings) com técnicas de **contagem de folhas (leaf counting)** inspiradas em trabalhos de fenotipagem vegetal, criando uma solução completa que pode ser usada em viveiros, pesquisas agronômicas ou experimentos acadêmicos.[^6][^7][^1]
## 2. Objetivos
### 2.1 Objetivo geral
Desenvolver e avaliar um sistema de visão computacional para **monitorar a germinação e o crescimento inicial de mudas**, identificando automaticamente quais cavidades/vasos germinaram e estimando o número de folhas/galhos por planta a partir de imagens.
### 2.2 Objetivos específicos
- Detectar e contar mudas em imagens de bandejas/vasos, distinguindo cavidades germinadas de não germinadas.[^8][^9][^10]
- Projetar e implementar um módulo de contagem de folhas/galhos por muda, utilizando abordagens de leaf counting baseadas em deep learning.[^6][^7][^11]
- Construir um pipeline temporal para acompanhar a evolução da taxa de germinação e do número médio de folhas/galhos ao longo de vários dias.[^1][^3]
- Avaliar quantitativamente o desempenho do sistema em termos de métricas de visão (precision, recall, mAP, erro de contagem) e métricas de aplicação (diferença entre contagem automática e manual).[^4][^12]
## 3. Contexto e trabalhos relacionados
Diversos trabalhos mostram o uso de visão computacional para **monitorar crescimento de plantas**, medindo altura, área de copa e biomassa em cultivos especiais, fazendas verticais e estufas. Esses sistemas utilizam imagens RGB simples ou conjuntos de câmeras e sensores para extrair características estruturais das plantas ao longo do tempo.[^1][^13][^2][^3][^5]

Na área de **leaf counting**, foram propostos modelos especializados (por exemplo, LC-Net e frameworks guiados por segmentação) para contar o número de folhas em plantas roseta e plântulas de cereais, tratando a contagem como um problema de regressão ou classificação sobre a imagem. Esses trabalhos mostram que é possível obter erros de contagem relativamente baixos mesmo em condições de oclusão parcial, desde que haja dataset anotado adequadamente.[^6][^12][^7][^11][^14]

Há também pipelines de **contagem de sementes e plantas** baseados em deep learning, que servem de referência para a etapa de detecção e contagem em imagens de alta densidade. Em paralelo, plataformas como o **Roboflow Universe** disponibilizam diversos datasets e modelos pré-treinados de seedlings, plantas e árvores, que permitem acelerar a prototipação de soluções de detecção de mudas.[^15][^8][^9][^10][^16]

O projeto proposto se apoia nesses avanços, combinando: (a) detecção de mudas em bandejas/vasos, (b) leaf counting em nível de planta, e (c) análise temporal para monitorar germinação e crescimento.
## 4. Dados e cenário experimental
### 4.1 Tipos de plantas e ambiente
O sistema é pensado para funcionar em **viveiros, estufas ou cultivos indoor**, monitorando mudas de hortaliças (alface, rúcula, tomate) ou ervas (manjericão, salsa, etc.), mas pode ser adaptado para outras espécies. As plantas são típicas do estágio pós-germinação, quando já emergiram do substrato e apresentam algumas folhas visíveis.[^1][^2][^5]

O ambiente ideal inclui:

- Iluminação razoavelmente uniforme e estável.
- Fundo relativamente consistente (bandejas e bancadas sem muita poluição visual).
- Distância de câmera e enquadramento relativamente fixos durante o experimento.[^3][^1]
### 4.2 Fontes de dados
Duas fontes principais podem ser usadas:

1. **Datasets públicos no Roboflow**:
   - Datasets de **seedlings** (Seedlings, seedling-detection, etc.) com bounding boxes para mudas em bandejas.[^8][^9][^10]
   - Datasets mais genéricos de plantas (por exemplo, modelos de plantas/ervas) que ajudam a detectar plantas em vasos e canteiros.[^17][^18][^5]

2. **Imagens próprias coletadas para o experimento**:
   - Sequências de fotos diárias das mesmas bandejas/vasos, para montar séries temporais de crescimento.[^1][^3]
   - Anotações manuais de:
     - Quais cavidades germinaram.
     - Número de folhas/galhos por planta (em algumas imagens selecionadas para validação).[^12][^7]

Uma estratégia prática é combinar datasets públicos do Roboflow para pré-treinar/ajustar o detector de mudas e usar imagens próprias para calibrar o módulo de contagem de folhas/galhos, reduzindo o esforço de anotação total.[^15][^9][^8]
## 5. Metodologia proposta
### 5.1 Aquisição e pré-processamento de imagens
- Fotografar bandejas ou vasos em visão de topo ou ligeiramente inclinada, mantendo distância e ângulo consistentes ao longo dos dias.[^1][^3]
- Padronizar resolução (por exemplo, 1080p) e aplicar correções básicas de cor/iluminação se necessário.
- Organizar as imagens em pastas por dia (D0, D1, D2, ...), com metadados indicando a data/hora e o lote de plantas.[^3][^1]
### 5.2 Anotação e preparação do dataset
- Usar o Roboflow ou ferramenta similar para anotar manualmente:
  - Bounding boxes das mudas em um subconjunto de imagens (para treinar o detector).
  - Número de folhas/galhos em cada muda para um conjunto menor de imagens de validação (target para leaf counting).[^15][^8][^9][^7]
- Dividir o dataset em treino/validação/teste, garantindo que haja imagens de diferentes dias e condições em cada partição.
- Aplicar data augmentation moderado (rotações leves, pequenas mudanças de brilho/contraste) para tornar o modelo mais robusto sem distorcer demais a morfologia das plantas.[^1][^11]
### 5.3 Modelo 1: detecção de mudas (germinação) com RF-DETR Small
Para detectar quais cavidades/vasos germinaram, será treinado um modelo de **detecção de objetos** usando a arquitetura **Roboflow RF-DETR**, na variante **Small**, que é a opção recomendada na plataforma para equilíbrio entre velocidade e acurácia.[^19][^20][^21]

RF-DETR é um modelo de detecção em tempo real baseado em transformers, com alta acurácia em benchmark COCO e bom desempenho com pouca quantidade de dados, convergindo mais rápido em domínios específicos. A variante Small reduz o custo computacional, permitindo inferência rápida em hardware modesto sem perder muito desempenho, o que é adequado a projetos acadêmicos e protótipos com webcam.[^20][^21][^19]

- Entrada: imagem da bandeja/vasos.
- Saída: bounding boxes para cada muda detectada, com classe única ("seedling").
- Critério “germinou": se a região de uma cavidade/vaso contém ao menos uma detecção de seedling, considera-se que germinou; caso contrário, não germinou.[^1][^4]

O modelo pode ser inicialmente treinado e versionado no Roboflow, mas o foco do projeto é permitir também o treino externo, sem depender de plano pago ou créditos de computação da plataforma.[^22][^23][^24]
### 5.4 Modelo 2: contagem de folhas/galhos por muda
A partir das bounding boxes do RF-DETR, o sistema recorta imagens de cada muda individual e passa esses recortes para um segundo modelo responsável por **estimar o número de folhas/galhos**.

Inspiração em leaf counting:

- Trabalhos atuais utilizam CNNs específicas (como LC-Net) ou frameworks guiados por segmentação para prever o número de folhas de uma planta a partir de uma imagem, com bons resultados em plantas jovens.[^6][^7][^11][^14]
- A tarefa é tratada como um problema de **regressão de contagem** (modelo prevê um número real que é arredondado) ou **classificação em faixas de número de folhas** (0–2, 3–5, etc.).[^12][^11]

No projeto, pode-se começar com um modelo relativamente simples:

- Base convolucional (por exemplo, ResNet pequena ou MobileNet) pré-treinada em ImageNet.
- Camadas finais adaptadas para regressão (saída escalar) ou classificação em poucas faixas de contagem.
- Treinamento supervisionado com o número de folhas/galhos anotado manualmente para cada recorte de muda.[^7][^25]
### 5.5 Pipeline de inferência
Durante a inferência (uso prático), o pipeline funciona em etapas:

1. Receber uma imagem de bandeja/vasos em determinado dia.
2. Rodar o **Modelo 1 (RF-DETR Small)** para obter bounding boxes das plantas germinadas.[^19][^20]
3. Para cada bounding box:
   - Recortar a região correspondente à muda.
   - Enviar o recorte para o **Modelo 2 (contador de folhas/galhos)**.[^6][^7]
4. Agregar resultados por cavidade/vaso:
   - Marcando se germinou (presença de muda).
   - Registrando o número estimado de folhas/galhos.[^1][^12]
5. Repetir o processo para imagens de diferentes dias, construindo séries temporais de germinação e crescimento.[^3][^1]
### 5.6 Estratégia de treinamento: Roboflow + Google Colab
Seguindo recomendações docentes, a estratégia de treinamento busca evitar que o modelo fique limitado às restrições de créditos e planos do Roboflow, utilizando a plataforma principalmente como ferramenta de **gestão de dados** e não como único ambiente de treino.[^22][^23][^26]

O fluxo proposto é:

- **Roboflow para dataset**: usar Roboflow para anotação, versionamento e augmentation do dataset, exportando os dados em formato YOLOv8/YOLO11 ou COCO para treino externo.[^27][^28][^26]
- **Treino principal no Google Colab**: treinar o modelo de detecção (RF-DETR open-source ou YOLO da Ultralytics) em notebooks do Colab, controlando hiperparâmetros, número de épocas e checkpoints, sem depender de créditos da nuvem do Roboflow.[^28][^29][^30][^31]
- **Uso opcional do trial do Roboflow**: o período de trial (por exemplo, 14 dias) pode ser aproveitado para experimentos rápidos de treino gerenciado e para baixar pesos de modelos RF-DETR treinados lá, mas o projeto não fica dependente desse prazo, pois o treino pode ser replicado totalmente em Colab.[^32][^33][^24]

Esse arranjo garante reprodutibilidade e independência de plano pago, ao mesmo tempo em que aproveita o ecossistema do Roboflow para facilitar anotação e experimentação rápida.[^23][^26][^22]
## 6. Métricas e avaliação
A avaliação cobre tanto o desempenho de visão computacional quanto a utilidade prática do sistema.
### 6.1 Métricas de detecção de mudas
- **Precision, recall e mAP** para detecção de mudas, usando IoU adequado (por exemplo, 0,5) para determinar acertos em bounding boxes.[^15][^8][^9][^11]
- Taxa de acerto na classificação “germinou/não germinou” por cavidade, comparando com rótulos manuais.[^4]
### 6.2 Métricas de contagem de folhas/galhos
- **Erro absoluto médio (MAE)** e erro quadrático médio (MSE) entre o número estimado de folhas/galhos e a contagem manual, seguindo a prática de trabalhos de leaf counting.[^6][^7][^11][^14]
- Eventualmente, métricas de acurácia por faixa (se o problema for tratado como classificação em categorias de contagem).[^12]
### 6.3 Métricas de aplicação
- Diferença entre a **taxa de germinação** calculada automaticamente e a contagem manual em alguns dias de referência.[^4]
- Capacidade de detectar tendências de crescimento (por exemplo, aumento médio no número de folhas/galhos por dia) coerentes com a observação humana.[^1][^3]

Essas métricas permitem demonstrar se o sistema é confiável o bastante para apoiar decisões em viveiros ou experimentos científicos.
## 7. Arquitetura de sistema e implementação
### 7.1 Componentes principais
- **Módulo de captura de imagens**: scripts ou rotina de câmera para obter fotos das bandejas/vasos diariamente.[^1][^3]
- **Módulo de detecção de mudas (Modelo 1 - RF-DETR Small)**: modelo treinado no Roboflow e/ou em Colab, executado via Hosted API ou Inference Server local, que retorna bounding boxes de seedlings.[^34][^35][^36]
- **Módulo de contagem de folhas/galhos (Modelo 2)**: rede neural em Python (PyTorch/TensorFlow) que recebe recortes de mudas e retorna a contagem estimada.[^6][^7][^11]
- **Módulo de análise temporal**: scripts em Python que armazenam resultados em banco de dados ou CSV e produzem curvas de germinação e crescimento.[^3][^1]
- **Interface de usuário** (opcional, mas desejável): aplicação web simples (por exemplo, Streamlit ou FastAPI + frontend) permitindo fazer upload de imagens, visualizar detecções e ver gráficos de evolução.[^5][^1]
### 7.2 Stack tecnológica sugerida
- Linguagem: Python.
- Framework de visão/detetores: Roboflow + arquitetura RF-DETR Small para detecção de mudas, com possibilidade de treino local em Colab usando implementações open-source.[^19][^29][^20][^21]
- Framework de deep learning: PyTorch ou TensorFlow/Keras para o modelo de leaf counting.[^6][^7]
- Interface: Streamlit para protótipo rápido, ou FastAPI para backend com frontend separado.[^1]
- Armazenamento: arquivos CSV/Parquet + diretório estruturado de imagens, ou um banco simples (SQLite/PostgreSQL) se necessário.[^3]
## 8. Extensões possíveis
O projeto pode ser estendido em várias direções, dependendo do tempo e do nível da disciplina:

- **Segmentação de planta**: substituir bounding boxes por máscaras de segmentação para medir diretamente área foliar e melhorar a contagem de folhas/galhos.[^37][^11]
- **Classificação de vigor**: além de contar folhas, treinar um modelo para classificar o vigor de cada muda (fraca, média, forte) com base na estrutura e cor da planta.[^1][^3]
- **Integração com IoT**: combinar o sistema de visão com sensores ambientais (luminosidade, temperatura, umidade) para correlacionar condições de cultivo com desempenho de germinação.[^13][^5][^38]
- **Suporte a múltiplas espécies**: adaptar o pipeline para diferentes culturas e estruturas de planta, eventualmente com modelos especializados por espécie.[^2][^5][^13]
## 9. Limitações e desafios
Alguns desafios esperados:

- **Qualidade e quantidade de dados anotados**: a contagem de folhas/galhos exige anotações relativamente detalhadas, o que pode ser trabalhoso.[^6][^7][^11]
- **Variação de iluminação e oclusões**: sombras, sobreposição de folhas e reflexos podem prejudicar tanto a detecção quanto a contagem, exigindo cuidados na coleta e data augmentation apropriado.[^1][^3][^11]
- **Generalização para outras condições**: modelos treinados em um tipo de bandeja/ambiente podem não generalizar bem para outros, sendo necessário retreino ou fine-tuning.[^13][^2][^5]

Mesmo com essas limitações, o projeto é viável em contexto acadêmico e permite explorar conceitos importantes de visão computacional, deep learning, fenotipagem de plantas e aplicação prática de IA em agricultura e horticultura.

---

## References

1. [Monitoring Plant Growth using Computer Vision - Roboflow Blog](https://blog.roboflow.com/monitor-plant-growth/) - In this blog post we will show how computer vision can be used to monitor plant growth. We will focu...

2. [New computer vision system can guide specialty crops monitoring](https://www.psu.edu/news/research/story/new-computer-vision-system-can-guide-specialty-crops-monitoring) - The team developed an automated crop-monitoring system capable of providing continuous and frequent ...

3. ["Monitoring Plants Growth in indoor Vertical Farms Using Computer ...](https://huskiecommons.lib.niu.edu/allgraduate-thesesdissertations/7552/) - This study focuses on designing AI and vision enabled system to track the growth of sweet basil plan...

4. [Automated plant growth monitoring system using machine vision](https://experts.arizona.edu/en/publications/automated-plant-growth-monitoring-system-using-machine-vision/) - `Ostinata') was developed using machine vision. It makes automatic hourly measurements of the plants...

5. [Vision AI-powered hydroponic farming enhances plant monitoring](https://www.ultralytics.com/blog/vision-ai-powered-hydroponic-farming-enhances-plant-monitoring) - Computer vision gives farmers a better way to monitor their crops. Cameras can be installed above th...

6. [A CNN-based model to count the leaves of rosette plants (LC-Net)](https://www.nature.com/articles/s41598-024-51983-y) - The area segmentation and counting of the leaf is a major component of plant phenotyping, which can ...

7. [Leaf Counting: Fusing Network Components for Improved Accuracy](https://pmc.ncbi.nlm.nih.gov/articles/PMC8224400/) - Two novel deep learning approaches for visual leaf counting tasks are proposed, evaluated, and compa...

8. [seedling Object Detection Dataset by EShu broccoli](https://universe.roboflow.com/eshu-broccoli/seedling-f9rmf) - A description for this project has not been published yet. Use Free Askew, B and Noseedling Detectio...

9. [seedling-detection Object Detection Dataset by IDEASLJJ](https://universe.roboflow.com/ideasljj/seedling-detection-vtapl) - 168 open source seedling images. seedling-detection dataset by IDEASLJJ.

10. [Seedlings Object Detection Dataset by jtyjyt - Roboflow Universe](https://universe.roboflow.com/jtyjyt/seedlings-yatyo) - 739 open source Seedlings images. Seedlings dataset by jtyjyt.

11. [A Segmentation-Guided Deep Learning Framework for Leaf Counting](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.844522/full) - A two-steam deep learning framework for segmenting plants and counting leaves with various size and ...

12. [Application to Wheat Leaf Counting at Seedling Stage | Plant ...](https://spj.science.org/doi/10.34133/plantphenomics.0041) - The digital plant phenotyping platform was used to simulate a large and diverse dataset of RGB image...

13. [Computer vision and IoT based plant phenotyping and growth ...](https://www.sciencedirect.com/science/article/pii/S1537511026000164) - By integrating advanced technologies such as computer vision (CV) and the Internet of Things (IoT), ...

14. [Leveraging multiple datasets for deep leaf counting](https://www.research.ed.ac.uk/en/publications/leveraging-multiple-datasets-for-deep-leaf-counting/) - We evaluate our method on the CVPPP 2017 Leaf Counting Challenge dataset, which contains images of A...

15. [Computer Vision for Agriculture with Roboflow](https://roboflow.com/industries/agriculture) - Roboflow hosts dozens of public datasets including PlantDoc, an object detection dataset containing ...

16. [A general Seeds-Counting pipeline using deep-learning model](https://dl.acm.org/doi/abs/10.1007/s10044-024-01304-w) - This study presents a novel Seeds-Counting pipeline harnessing deep learning algorithms to facilitat...

17. [Herbs_Pland_Model Object Detection Model by Raymond Periabras](https://universe.roboflow.com/raymond-periabras-wwar1/herbs_pland_model) - 39 open source plants-herbs images plus a pre-trained Herbs_Pland_Model model and API. Created by Ra...

18. [Plants Classification Model by Plants - Roboflow Universe](https://universe.roboflow.com/plants-mcgil/plants-9wp6a) - 172 open source Plants images plus a pre-trained Plants model and API. Created by Plants.

19. [RF-DETR Object Detection Model: What is, How to Use - Roboflow](https://roboflow.com/model/rf-detr) - RF-DETR is a real-time object detection transformer-based architecture designed to transfer well to ...

20. [RF-DETR: A SOTA Real-Time Object Detection Model - Roboflow Blog](https://blog.roboflow.com/rf-detr/) - RF-DETR is a real-time object detection transformer-based architecture designed to transfer well to ...

21. [RF-DETR by Roboflow: Fast Real-time Object Detection](https://learnopencv.com/rf-detr-object-detection/) - RF-DETR is a real-time, transformer-based object detection model architecture developed by Roboflow ...

22. [Training time + credits - 🛠️ Feature Reqs - Roboflow](https://discuss.roboflow.com/t/training-time-credits/10845) - Hi, I started training an rf-detr nano model. It said that the estimated cost was around 5 credits a...

23. [Roboflow Training Time Long - Community Help](https://discuss.roboflow.com/t/roboflow-training-time-long/4228) - The duration of the training process can vary depending on the size of your dataset and the images i...

24. [Train a Model | Roboflow Docs](https://docs.roboflow.com/train/train) - When you approve a batch of image annotations, Instant models are automatically trained. These model...

25. [Rjt5412/Leaf-Count-Prediction - GitHub](https://github.com/Rjt5412/Leaf-Count-Prediction) - Given a plant image, it can predict the leaf count which can help monitor the health and growth of t...

26. [Getting Started with Roboflow](https://blog.roboflow.com/getting-started-with-roboflow/) - Roboflow provides everything you need to label, train, and deploy computer vision solutions. You can...

27. [Roboflow - Ultralytics YOLO Docs](https://docs.ultralytics.com/integrations/roboflow/) - Once your dataset version is generated, you can export it in various formats suitable for model trai...

28. [Launch: Version, Export, and Train Models in the Roboflow Python ...](https://blog.roboflow.com/launch-version-export-and-train-models-in-the-roboflow-python-package/) - The new Python export() method lets you export your dataset from the Roboflow platform in one line o...

29. [[ICLR 2026] RF-DETR is a real-time object detection and ... - GitHub](https://github.com/roboflow/rf-detr) - RF-DETR is a real-time transformer architecture for object detection and instance segmentation devel...

30. [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) - Discover YOLO11, an advancement in real-time object detection, offering excellent accuracy and effic...

31. [Object Detection with YOLO11: Ultralytics Tutorial](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-object-detection) - Explore how the new Ultralytics YOLO11 model can be used for object detection to achieve higher prec...

32. [Can I export the model I trained to my local computer?](https://discuss.roboflow.com/t/can-i-export-the-model-i-trained-to-my-local-computer/8033) - You probably can. Go here to start the process: Deploy Your Computer Vision Model - Roboflow When it...

33. [Download Model Weights | Roboflow Docs](https://docs.roboflow.com/deploy/download-roboflow-model-weights) - Roboflow Inference is our open-source, scalable system for running models locally on CPU and GPU dev...

34. [Run a model - Roboflow Inference](https://inference.roboflow.com/quickstart/run_a_model/) - There are two ways to do this: the inference Python package which loads and runs models directly in ...

35. [Serverless Hosted API | Roboflow Docs](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) - Run Workflows and Model Inference ... Models deployed to Roboflow have a REST API available through ...

36. [Roboflow Inference: Index](https://inference.roboflow.com) - Inference handles the core tasks of computer vision applications. Integrate cutting-edge models, eff...

37. [A Segmentation-Guided Deep Learning Framework for Leaf Counting](https://digitalcommons.unl.edu/biosysengfacpub/802/) - A two-steam deep learning framework for segmenting plants and counting leaves with various size and ...

38. [AI based Plant Growth Monitoring System using Computer Vision](https://ui.adsabs.harvard.edu/abs/2023tems.conf...25B/abstract) - This research work aims to develop an AI-based plant growth monitoring system using computer vision....

