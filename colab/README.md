# Notebooks Colab

## train_v4.ipynb

Retreino do modelo YOLO com dataset v4 (inclui 11+ fotos WhatsApp auto-labeled via SAM para melhor cobertura de bandejas em ambiente real).

Use o badge no README principal para abrir direto no Colab. Saida esperada: `best.pt` salvo no Google Drive em `MyDrive/projetogerminacao/runs/train/v4_retrain_YYYYMMDD_HHMMSS/weights/`.

Apos o treino, baixe o `best.pt` e commite em `models/best_v4.pt` no repositorio.

## colab_sam_label.ipynb

Auto-labelagem via Segment Anything (SAM) para fotos magenta extremo que o modelo atual rotula errado. Precursor do dataset v4.

## colab_train.ipynb

Pipeline de treino generico YOLO11 no Colab. Versao sem customizacoes v4 - util como referencia ou para retreinos futuros.

## colab_monitoramento_germinacao.ipynb

Notebook de analise e monitoramento de germinacao - exploracao de dados e visualizacao.
