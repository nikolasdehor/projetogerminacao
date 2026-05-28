# Spec: WhatsApp Bot Guard (defesa em camadas)

**Data**: 2026-05-28
**Autor**: Nikolas de Hor
**Status**: Aprovado, aguardando plano de execução

## Problema

O bot WhatsApp do projetogerminação executa inferência YOLO em toda imagem que chega em grupos onde está presente (rate-limited 5/hora por grupo), sem filtro de conteúdo. Resultado: em um grupo onde foi enviada uma foto aleatória fora do contexto de germinação, o bot respondeu com análise da imagem, gerando ruído e exposição indesejada.

Hoje (`app/whatsapp_routes.py:_handle_image_message`, linhas 492-694) o pipeline é:
1. Filtra broadcast (skip) e separa DM vs grupo (linha 387-401)
2. Rate-limit por grupo (5 imgs/hora) e dedup por message_id
3. Baixa mídia e roda `run_inference(conf_threshold=0.5)`
4. Responde com imagem anotada e métricas

Não há nenhuma verificação de "essa foto é uma bandeja de germinação ou foto aleatória".

## Estratégia aprovada

Defesa em camadas: whitelist de grupos + keyword no caption + threshold pós-inferência. Combinação rejeitada explicitamente: classificador semântico CLIP (custo de latência sem ganho proporcional vs heurísticas).

## Pipeline de decisão

```
Imagem chega ->
  é broadcast? -> SKIP (comportamento atual)
  é DM?        -> PROCESSA (sem mudança)
  é grupo?     ->
    grupo na whitelist? -> PROCESSA
    caption tem keyword? -> PROCESSA
    senão                -> SILÊNCIO + log estruturado

PROCESSA -> run_inference() ->
  detections < GUARD_MIN_DETECTIONS  -> SILÊNCIO + log
  conf média < GUARD_MIN_MEAN_CONF   -> SILÊNCIO + log
  OK                                  -> responde normalmente
```

## Componentes

### Whitelist de grupos

- Arquivo: `config/group_whitelist.json` (gitignored)
- Template versionado: `config/group_whitelist.example.json`
- Schema:
```json
{
  "groups": [
    {"jid": "5562xxxxxxxxx@g.us", "label": "Estufa Norte", "added_at": "2026-05-28"}
  ]
}
```
- Default em produção: lista vazia. Adiciona-se grupo após validação em campo.
- Carregamento: leitura no startup; recarregamento exige restart (YAGNI hot-reload por enquanto).

### Keyword filter

- Arquivo: `config/germina_keywords.json` (versionado, não é secret)
- Defaults:
```json
{
  "keywords": ["germina", "germinação", "bandeja", "plaqueta", "semente", "sementes", "contar"]
}
```
- Match: case-insensitive, accent-insensitive (normalizar com unicodedata NFD), substring no caption.

### Threshold pós-inferência

- Env vars com defaults:
  - `GUARD_MIN_DETECTIONS=3` (mínimo de bounding boxes válidas)
  - `GUARD_MIN_MEAN_CONF=0.55` (confiança média das detecções)
- Valores conservadores baseados no `conf_threshold=0.5` já usado pelo YOLO. Pode precisar de ajuste após observar dados reais.

### Logging do silêncio

Linha estruturada por skip:
```
guard_skip group=<jid_truncado> reason=<no_keyword|low_count|low_conf|no_whitelist> caption_snippet=<primeiros_40_chars>
```
Permite auditar falsos negativos sem expor caption completo.

## Mudanças por arquivo

| Arquivo | Tipo | Mudança |
|---|---|---|
| `app/guards.py` | novo (~120 linhas) | `should_process_image(message) -> tuple[bool, str]` + `passes_post_inference_guard(detections) -> tuple[bool, str]` + loaders de config |
| `app/whatsapp_routes.py:_handle_image_message` | edit | Chama `should_process_image` antes do download; chama `passes_post_inference_guard` após `run_inference`; logs estruturados em ambos os skips |
| `.gitignore` | edit | Adicionar `config/group_whitelist.json` |
| `config/group_whitelist.example.json` | novo | Template documentado |
| `config/germina_keywords.json` | novo | Lista versionada de keywords |
| `tests/test_guards.py` | novo | 9 casos: broadcast/DM/whitelist hit/keyword hit/grupo sem nada/reply ao bot/detecções OK/poucas detecções/conf baixa |
| `.env.example` | edit | Documentar `GUARD_MIN_DETECTIONS` e `GUARD_MIN_MEAN_CONF` |

## Testes

Unit (pytest):
- `test_should_process_image_broadcast_skip`
- `test_should_process_image_dm_passes`
- `test_should_process_image_group_in_whitelist_passes`
- `test_should_process_image_group_keyword_passes`
- `test_should_process_image_group_keyword_accent_insensitive`
- `test_should_process_image_group_no_keyword_silent`
- `test_passes_post_inference_low_count_silent`
- `test_passes_post_inference_low_conf_silent`
- `test_passes_post_inference_ok_processes`

Smoke manual (após deploy):
- Enviar foto de gato em grupo de teste -> silêncio. Verificar log `guard_skip ... reason=no_keyword`.
- Enviar bandeja com caption "germina aí" -> resposta normal.
- Enviar bandeja em grupo whitelisted sem caption -> resposta normal.

## Rollout

1. Deploy com whitelist vazia e keywords default. Bot só responde em grupos quando caption contém keyword OU em DM.
2. Validar em campo: foto aleatória no grupo do trabalho deve silenciar; foto de bandeja com legenda esperada deve responder.
3. Adicionar JIDs reais de grupos confiáveis na whitelist conforme aparecerem.
4. Ajustar `GUARD_MIN_DETECTIONS` e `GUARD_MIN_MEAN_CONF` se observar falsos negativos em bandejas válidas.

## Out of scope (YAGNI)

- Comando admin de runtime `/whitelist add`. Por enquanto, edita JSON e reinicia.
- Notificação ao usuário de que a foto foi ignorada. Silêncio total é o pedido.
- Classificador semântico (CLIP zero-shot). Keyword + threshold cobre 90% dos casos sem custo de latência.

## Relação com retreino v4

Serial, conforme decidido. Guard primeiro (deploy rápido para parar o ruído), retreino v4 entra em sprint separada. O retreino não é bloqueador deste spec.

## Critérios de aceite

- Foto aleatória em grupo não-whitelisted, sem keyword, gera linha `guard_skip` no log e zero resposta.
- Foto de bandeja com caption "germina" em grupo qualquer gera resposta normal.
- Foto de bandeja em grupo whitelisted gera resposta sem precisar de caption.
- DM continua aberto sem caption.
- Cobertura de testes do `app/guards.py` >= 80%.
- Nenhuma regressão no rate-limit existente nem no fluxo de DM.
