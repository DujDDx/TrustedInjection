# serve.sh <gguf> : (re)start a writer server on GPUs 1-2, port 8820, wait for health
for p in $(pgrep -x llama-server); do tr "\0" " " < /proc/$p/cmdline | grep -q "port 8820" && kill $p; done; sleep 3
cd /home/dujddx/llama.cpp && CUDA_VISIBLE_DEVICES=1,2 nohup ./build/bin/llama-server -m "$1" -ngl 999 --tensor-split 1,1 \
  --flash-attn on --ctx-size 8192 --parallel 1 --host 127.0.0.1 --port 8820 > /mnt/hdd4/dujddx/mw/server_$(basename $1 .gguf).log 2>&1 &
for i in $(seq 480); do curl -s -m 2 http://127.0.0.1:8820/health | grep -q ok && { echo READY; exit 0; }; pgrep -x llama-server >/dev/null || break; sleep 5; done
echo SERVERFAIL; tail -5 /mnt/hdd4/dujddx/mw/server_$(basename $1 .gguf).log
