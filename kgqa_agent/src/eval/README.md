# eval 目录功能与用法（全量说明）

本目录提供**统一的 KGQA 评测框架**，当前主打场景是：

- CWQ 等 Freebase 风格问答数据
- 两类模型后端：远程 API、本地 vLLM
- 可选的 **交互式 KG 查询**（模型自己发 `<kg-query>` 调 Virtuoso SPARQL）

推荐入口：根目录下的 `run_eval_vllm.sh` / `run_eval_api.sh`，这两个脚本都会调用 `kgqa_agent/scripts/run_kg_eval.py`。

---

## 1. 目录结构总览

核心文件：

- `evaluator.py`：统一评测流程（加载数据 → 构造 Prompt → 调模型 → 算 EM/F1 → 写结果）。
- `model_client.py`：模型后端包装（API / vLLM），暴露统一接口 `generate` / `generate_batch`。
- `kg_augmented_client.py`：**KG 增强客户端**，包装任意 `BaseModelClient`，在生成时解析 `<kg-query>` 调 KG。
- `metrics.py`：评测指标（Relaxed EM / Token F1）。
- `datasets/`：数据加载器与小样本 JSON。
  - `cwq_loader.py` / `webqsp_loader.py` / `grailqa_loader.py`
  - 若干 `cwq_extracted_*.json` 小样本（由 `extract_cwq_subset.py` 生成）。

评测入口脚本：

- `kgqa_agent/scripts/run_kg_eval.py`：统一的 CLI 评测入口，由 `run_eval_with_env.sh` 调用。

提示词依赖：

- `kgqa_agent/prompts/prompts.py`：
  - `build_search_prompt(...)`：构造首轮带 Topic Entities 的 KG 搜索 Prompt。
  - `build_continuation_prompt(...)`：每次 `<kg-query>` 返回信息后，拼接续写 Prompt。
  - `FORCE_ANSWER_PROMPT`：达到最大 KG 次数后强制收敛出 `<answer>`。

SPARQL 与工具：

- `kgqa_agent/src/tools/`：
  - `direct_sparql_client.py`：直连 Virtuoso 的 SPARQL 客户端（带 BM25 排序）。
  - `relation_normalizer.py`：谓词字符串归一化，用于校验模型选择的 relation 是否在上一轮列表中。
  - `entity_resolver.py`：实体 ID/名称解析（仅在部分路径使用）。

---

## 2. 评测数据流（从数据到指标）

当前实现的整体流程可以概括为：

```text
dataset file (JSON/JSONL) ──► loader (datasets/*.py)
  └─► 统一样本格式: {id, question, answers, topic_entity, ...}
       └─ run_evaluation(...)
           ├─ 构造模型输入 prompt
           ├─ 调用模型 (可选 KG 增强)
           ├─ 解析答案 / 计算 EM & F1
           └─ 写入结果 JSON + 汇总指标
```

### 2.1 数据加载（`datasets/*.py`）

统一入口：`evaluator.load_dataset(dataset_type, dataset_path)`：

- `dataset_type == "cwq"` → `load_cwq(path)`：
  - 输出字段示例：
    - `id`: 题目 ID（`ID` 字段或顺序号）
    - `question`: 文本问题；多源字段优先级：`question` / `webqsp_question` / `machine_question`
    - `answers`: 标准答案列表（从 `answers` / `answer` / `composition_answer` 解析）
    - `topic_entity`: 可选 dict `{mid: name}`，供 KG 模式使用
    - `sparql_query`, `raw`: 其他信息

- `dataset_type == "webqsp"` / `"grailqa"`：对应 loader 做类似标准化，保证 `id` / `question` / `answers` 至少存在（目前主用 CWQ，但接口仍兼容这两种）。

> 评测时，`run_evaluation(..., limit=N)` 会在 loader 输出后再裁掉前 N 条，用于抽样测试。

### 2.2 模型封装与选择（`model_client.py`）

统一抽象：

```python
class BaseModelClient:
    def generate(self, prompt: str, *, system: Optional[str] = None, **gen_kwargs) -> str: ...
    def generate_batch(self, prompts: List[str], *, system: Optional[str] = None, **gen_kwargs) -> List[str]: ...
```

当前支持两类实现：

- `ApiModelClient`（`--model api`）：
  - 使用 `openai` 官方 SDK；
  - 通过 `ApiModelConfig` 控制 `model`、`temperature`、`max_tokens` 等；
  - 从环境变量读取：`OPENAI_API_KEY`，可选 `OPENAI_BASE_URL`。

- `VLLMModelClient`（`--model local` 或 `--model vllm`）：
  - 使用 `vllm.LLM` 本地加载大模型；
  - 支持 `tensor_parallel_size`、`pipeline_parallel_size`、`gpu_memory_utilization` 等；
  - 使用 tokenizer 的 `apply_chat_template` 尽量适配聊天格式；
  - `generate_batch` 一次前向多条样本。

工厂函数：

```python
def build_model_client(kind: str, cfg: dict) -> BaseModelClient:
    if kind == "api":
        return ApiModelClient(ApiModelConfig(**cfg))
    if kind in ("local", "vllm"):
        return VLLMModelClient(VLLMModelConfig(**cfg))
```

在 `run_evaluation` 中：

- 对本地/vLLM 后端，会先构造一个 `shared_base_model`，在多 batch 间复用；
- 每个 batch 再调用 `_wrap_with_kg` 决定是否加上 KG 增强包装。

### 2.3 KG 增强模型客户端（`kg_augmented_client.py`）

`KGAugmentedModelClient` 的作用是：

- 拦截模型输出里的 `<kg-query>...</kg-query>`；
- 解析其中的函数调用（如 `get_tail_relations("Hinduism")`）；
- 调用 `DirectSPARQLKGClient` 执行实际的 SPARQL 查询；
- 把查询结果整理成可读文本 `information` 再反馈给模型；
- 用 `<think>` / `<answer>` / `<kg-query>` 等标签构建详细的 `trace`。

#### 支持的 KG 调用原语

模型在 `<kg-query>` 中可以发出如下调用（字符串匹配）：

- `get_tail_relations("<entity>")`
- `get_head_relations("<entity>")`
- `get_tail_entities("<entity>", "<relation>")`
- `get_head_entities("<entity>", "<relation>")`

所有调用都会：

- 先用 `_resolve_entity_for_query` 将**名字**解析成 ID（依赖内部 entity registry 和 topic_entity 映射）；
- 对关系调用：
  - 使用 `kg_client.get_*_relations(..., question=augmented_question, top_k=relation_top_k)`；
  - 用 `format_relations_for_prompt(...)` 生成带编号的关系列表，记录到 `self._last_relations_set` 以约束后续选择；
- 对实体调用：
  - 校验本次 relation 是否出现在刚才列表中（通过 `normalize_relation` + `self._last_relations_set`）；
  - 再调用 `kg_client.get_*_entities(..., top_k=10)` 获得实体候选，格式化为名字列表。

#### 交互流程（简化示意）

```text
1. evaluator 构建首轮 prompt（包含问题和 Topic Entities 名称）
2. KGAugmentedModelClient._interactive_generate(initial_prompt)
   ├─ 循环 <= max_calls 次：
   │   ├─ 调 base_client.generate(current_prompt)
   │   ├─ 解析 <think> 并记录
   │   ├─ 如找到 <kg-query>：
   │   │   ├─ 解析函数 + 参数
   │   │   ├─ 调 SPARQL 客户端执行
   │   │   ├─ 记录 tool_call + information
   │   │   └─ 用 build_continuation_prompt(...) 生成下一轮 prompt
   │   └─ 否则：如含 <answer>，直接返回；否则返回完整响应
   └─ 达到 max_calls：
       └─ 用 FORCE_ANSWER_PROMPT 强制模型给出最终 <answer>
```

每个样本结束后，`get_batch_traces_and_reset()` 会取出这一批样本的所有 `trace`（包含 prompts、tool_call、information 等），并清空缓存。

### 2.4 评测核心：`run_evaluation`（`evaluator.py`）

统一评测入口如下：

```python
results = run_evaluation(
  dataset_path=".../cwq_extracted_100.json",
  dataset_type="cwq",              # cwq / webqsp / grailqa
  model_kind="api",                # api / local / vllm
  model_cfg={"model": "gemini-2.5-flash", "temperature": 0.7, "max_tokens": 4096},
  limit=100,                        # 可选：仅评测前 N 条
  num_workers=2,
  kg_server_url="http://localhost:18890",
  max_calls=3,
  kg_top_k=10,
  output_dir="eval_results"        # 结果目录
)
```

内部主要步骤：

1. **加载数据**：`data = load_dataset(dataset_type, dataset_path)`；可选 `limit` 截断。
2. **构造模型工厂**：
  - 若为 `local`/`vllm`，本地初始化一次 `shared_base_model`；
  - `create_model()` 每次返回 `shared_base_model` 或新建的 API client，并总是包装为 `KGAugmentedModelClient`（KG 功能始终启用）。
3. **构造 batch**：按 `batch_size` 切分；使用 `ThreadPoolExecutor(max_workers=num_workers)` 并行处理 batch。
4. **process_batch(batch)**：
   - 为每个样本准备：
     - `questions_raw`: 原始 `question`；
     - `topic_entities_list`: 每个样本的 topic_entity 映射（供 KG 模式使用）；
     - `qs`: 实际发送给模型的 user prompt：
       - KG 模式：`build_search_prompt(question, max_calls, topic_entities=te_names)`；
       - 非 KG 模式：`prompt_user_template.format(question=...)` 或原始问题。
   - 调用 `model.generate_batch(...)` 得到一批 `preds_text`。
   - KG 模式下，如果 model 有 `get_batch_traces_and_reset`，再取回对应 trace。
   - 对每条样本：
     - 如模型支持 `extract_answer`（KG 模式），先从完整输出中抽取答案字符串；
     - 计算 `em = exact_match(pred, golds)` 与 `f1 = f1_score(pred, golds)`；
     - 组装一条记录：`{id, raw_question, user_prompt, question, answers, prediction, raw_prediction, em, f1, trace?}`。
5. **汇总预测与指标**：
   - `preds`：拼接所有 batch 的记录；
   - 使用统一的文本指标：
     - `exact_match`：带文本归一化 + 双向 substring 的宽松 EM；
     - `token_f1_score`：基于 token 的 F1（取对所有 gold 中的 best）。
6. **写出结果**：
   - 默认写入 `output_dir/<model>_<dataset_type>_<timestamp>.json`；包含：
     - `dataset_type`, `dataset_path`, `model_kind`, `model_cfg`, `created_at` 等元信息；
     - `metrics`：聚合统计（EM/F1）；
     - `predictions`：逐样本详细结果（含 trace）。

---

## 3. KG 交互协议与 trace 结构

### 3.1 Prompt 与标签约定

在 KG 模式下，模型需要遵守以下标签协议（提示词已在 `prompts.py` 中说明）：

- `<kg-query>...</kg-query>`：包裹一条 KG 函数调用，内容必须是单行 Python 风格伪代码，如：

```text
<kg-query>get_tail_relations("Hinduism")</kg-query>
<kg-query>get_tail_entities("Hinduism", "religion.religion.holidays")</kg-query>
```

- `<think>...</think>`：模型的中间思考过程，仅记录在 trace 中，不影响执行。
- `<answer>...</answer>`：最终答案，`KGAugmentedModelClient.extract_answer` 会从完整输出中抽取这一段作为 `prediction`。

### 3.2 trace 字段（`predictions[*].trace`）

单个样本的 trace 是一个 list，按时间顺序记录：

- `{"type": "prompt", "tag": "prompt", "content": ...}`：每轮完整 prompt。
- `{"type": "think", "tag": "think", "content": ...}`：对应 `<think>` 内容。
- `{"type": "kg_query", "tag": "kg-query", "content": query_str, "call_number": k}`：模型发出的 KG 调用。
- `{"type": "tool_call", "tool": "get_tail_relations", "args": {...}, "result_count": n, ...}`：实际执行的工具调用。
- `{"type": "information", "tag": "information", "content": formatted_results, ...}`：返回给模型的文字信息。
- `{"type": "answer", "tag": "answer", "content": ...}`：最终 `<answer>` 内容。
- 以及错误 / `max_calls_reached` 等事件。

这些 trace 同时也为前面你构建的 `synthesize_information.py` 提供了风格模板。

---

## 4. 指标定义（`metrics.py`）

- `qa_normalize_answer`：统一小写、去标点、去冠词、压缩空格。
- `exact_match(pred, golds)`：
  - 先做严格 EM（normalize 之后完全相等）；
  - 然后检查“gold ⊆ pred”与“pred ⊆ gold”的子串关系；
  - 任意条件满足则记为 1，否则 0。
- `token_f1_score(pred, golds)` / `f1_score`：
  - 将 pred 与每个 gold 正则化 → 分词 → 取并集与交集计算 precision / recall；
  - 返回所有 gold 中的最大 F1。
- `calculate_prf1(gold_answers, pred_answers)`：
  - WebQSP 官方 PRF1：以列表形式计算 TP/FP/FN；
  - 返回 `(precision, recall, f1, hit)`。

官方评测封装：

- `evaluate_webqsp_official(predictions, gold_data, original_data_path)`：
  - 如提供 `original_data_path`，会加载原始 `WebQSP.test.json` 中所有 parses，
    对每个 parse 的答案列表计算 PRF1，然后取 **F1 最好的 parse** 作为该题最终分数。

- `evaluate_grailqa_official(predictions, gold_data, fb_roles_file, fb_types_file, reverse_properties_file)`：
  - 如提供 ontology 文件，使用 `SemanticMatcher` 做逻辑式比较（same_logical_form）。
  - 按 level（i.i.d. / compositional / zero-shot 等）拆分统计 EM/F1。

---

## 5. CLI 使用方式

推荐通过根目录的 `run_eval_with_env.sh`，内部会：

```bash
python kgqa_agent/scripts/run_kg_eval.py \
  --task cwq_dev_qwen3 \
  --dataset kgqa_agent/src/eval/datasets/cwq_extracted_100.json \
  --dataset-type cwq \
  --model api \
  --model-config '{"provider": "openai", "model": "gpt-4o-mini"}' \
  --eval-mode standard \
  --limit 10 \
  --batch-size 4 \
  --num-workers 2 \
  --enable-kg-query \
  --kg-server-url http://localhost:18890 \
  --max-calls 3 \
  --kg-rel-top-k 10
```

关键参数说明：

- `--task`：任务名，结果会写到 `eval_results/<TASK>/` 目录下。
- `--dataset` / `--dataset-type`：数据路径与类型（`cwq`/`webqsp`/`grailqa`）。
- `--model` / `--model-config`：模型后端与配置：
  - `--model api`：`model-config` 结构需匹配 `ApiModelConfig`；
  - `--model vllm`：`model-config` 结构需匹配 `VLLMModelConfig`。
- `--eval-mode`：`standard` 使用统一 EM/F1；`official` 时走 WebQSP/GrailQA 各自的官方指标。
- `--limit`：仅评测前 N 条，用于快速测试。
- `--batch-size` / `--num-workers`：前者是模型 batch size，后者是 Python 线程数。
- `--enable-kg-query`：打开后才会使用 `KGAugmentedModelClient`，允许模型发 `<kg-query>`；否则就是普通静态 QA。
- `--kg-server-url`：Virtuoso 基础地址，脚本内会自动补 `/sparql`。
- `--max-calls`：每个样本最多 KG 调用次数。
- `--kg-rel-top-k`：每次列关系时最多展示多少个 predicate。

脚本还提供一个可选的 KG 自检：

```bash
python kgqa_agent/scripts/run_kg_eval.py \
  ... \
  --enable-kg-query \
  --self-check-kg
```

会用固定实体 `m.02mjmr`（Albert Einstein）去调用 `get_tail_entities` / `get_tail_relations`，检查 Virtuoso 是否可用。

---

## 6. 输出文件与人工审阅

`run_evaluation` 返回的结果中包含：

- `out_path`：主结果 JSON 文件路径，结构大致为：

```json
{
  "task": "cwq_dev_qwen3",
  "dataset_type": "cwq",
  "model_kind": "api",
  "metrics": {
    "em": 0.42,
    "f1": 0.55,
    "num_examples": 100
  },
  "predictions": [
    {
      "id": "cwq_0",
      "raw_question": "...",
      "user_prompt": "...",
      "question": "...",
      "answers": ["..."],
      "prediction": "...",
      "raw_prediction": "...",
      "em": 1.0,
      "f1": 1.0,
      "trace": [ ... ]
    },
    ...
  ],
  "kg_query_stats": {
    "total_calls": 123,
    "avg_calls_per_question": 1.23
  }
}
```

`run_kg_eval.py` 额外会调用 `save_human_summary(...)` 生成一份人类可读的摘要：

- 路径：`eval_results/<TASK>/summary_YYYYMMDD_HHMMSS.txt`
- 内容包括：
  - 主要指标（EM/F1 等）；
  - KG Query 统计；
  - 若干条样本的问答与得分，方便快速 eyeballing。

---

## 7. 与上游组件的关系

- 上游数据生成：
  - `kgqa_agent/src/data_gen/synthesize_information.py` / pipeline 会为同一批 CWQ 问题生成 KG trace（信息合成）；
  - 本目录的评测使用的则是**面向 LLM 的问答数据**（`question` + `answers`），互相独立，但 trace 结构风格保持一致。
- 模型提示词：
  - 所有 KG 指令（如何写 `<kg-query>`、如何选择 relation/entity）都集中在 `kgqa_agent/prompts/prompts.py` 中配置，评测端只负责调用 `build_search_prompt` / `build_continuation_prompt`。

这意味着你可以在不改评测框架代码的前提下，单独调参：

- 更换/微调模型；
- 修改 Prompt 模板；
- 替换/扩展数据集；
- 或禁用 KG 直接跑纯文本 QA，对比有无 KG 时的性能差异。
