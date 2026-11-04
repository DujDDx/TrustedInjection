# multi-writer training data: each writer writes whole events (a shard of the 280 train events); Llama writes the 60 val events
S=$(cd "$(dirname "$0")" && pwd); L=/mnt/hdd4/dujddx/mw; M=/mnt/hdd4/models; PY=/mnt/hdd4/venvs/hipporag/bin/python
D=/mnt/hdd4/TrustedInjection/experiments/tagged_propagation_20261031
stop() { for p in $(pgrep -x llama-server); do kill $p; done; while pgrep -x llama-server >/dev/null; do sleep 2; done; sleep 5; }
gen() { # label port kwargs minmax args...
  w=$1; port=$2; kw=$3; mt=$4; shift 4; s=$(date +%s); cd $D
  TI_BASE=http://127.0.0.1:$port TI_TEMPLATE_KWARGS="$kw" TI_MIN_MAX_TOKENS=$mt $PY gen_heldout.py --writer $w --pool train "$@" --out mw_train/docs_$w.jsonl > $L/gen_$w.log 2>&1
  echo "DONE $w rc=$? $(( $(date +%s) - s ))s docs $(wc -l < mw_train/docs_$w.jsonl) dropped $(grep -c 'A failed' $L/gen_$w.log) $(date +%H:%M)"
}
swap() { stop; bash $S/serve.sh "$1" > $L/serve.txt 2>&1; tail -1 $L/serve.txt; grep -q READY $L/serve.txt || { echo "SERVERFAIL $1"; exit 1; }; }
mkdir -p $D/mw_train
gen qwen3.8-27b 8818 '{}' 0 --splits train --shard 0/4
swap $M/gpt-oss-20b-F16.gguf;  gen gpt-oss-20b 8820 '{"reasoning_effort":"low"}' 1500 --splits train --shard 1/4
swap $M/GLM-4.7-Flash-UD-Q6_K_XL.gguf; gen glm-4.7-flash 8820 '{}' 0 --splits train --shard 2/4
swap $M/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf; gen gemma-4-26b-it 8820 '{}' 1500 --splits train --shard 3/4
swap /mnt/hdd4/dujddx/gguf/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf; gen llama-3.1-8b 8820 '{}' 0 --splits val
stop
cd $D; cat mw_train/docs_*.jsonl > mw_train/docs.jsonl; mkdir -p mw_train/pairs
$PY build_pairs.py --events events_p4v2.jsonl --docs mw_train/docs.jsonl --out-dir mw_train/pairs --tok /mnt/hdd4/models/research/NEWS-COPY-wire-clustering-na/tokenizer.json > $L/build.log 2>&1; echo "BUILD rc=$?"; tail -5 $L/build.log
echo GENDONE
