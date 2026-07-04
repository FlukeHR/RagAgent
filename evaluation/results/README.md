# evaluation/results/

评估历史记录（append-only）。每次带 `--record` 跑评估会往 `history.jsonl` 追加一行，
记录「时间 + git 提交/分支/是否 dirty + 配置快照 + 指标」，用于回溯「哪次改动带来哪点提升」。

- 写入：`python3 evaluation/eval_qasper.py --record`、`... eval_generation.py --yes --record`
- 查看：`python3 evaluation/results_log.py`（`--kind` 过滤、`--last N`、`--compare` 看最新两次增减）

本目录**纳入 git**（评估历史是项目资产）；`history.jsonl` 一行一条 JSON，便于 diff 与画趋势。
