# ctx=4096, decode 64
echo "=================Qwen3-0.6B, ctx=4096, decode=64==================="
python eval.py --model Qwen/Qwen3-0.6B         --samples 50 --ctx-len 4096 --decode-len 64  --out outputs/qwen3_0.6B_4k                                               
echo "=================Llama-3.2-1B, ctx=4096, decode=64==================="
python eval.py --model meta-llama/Llama-3.2-1B --samples 50 --ctx-len 4096 --decode-len 64  --out outputs/llama_1B_4k                                                 
echo "=================OLMo-2-0425-1B, ctx=4096, decode=64==================="
python eval.py --model allenai/OLMo-2-0425-1B  --samples 50 --ctx-len 4096 --decode-len 64  --out outputs/olmo2_1B_4k                                                 
                                                                                                                                                                    
# ctx=8192, decode 128                                                                                                                                                
echo "=================Qwen3-0.6B, ctx=8192, decode=128==================="
python eval.py --model Qwen/Qwen3-0.6B         --samples 50 --ctx-len 8192 --decode-len 128 --out outputs/qwen3_0.6B_8k                                               
echo "=================Llama-3.2-1B, ctx=8192, decode=128==================="
python eval.py --model meta-llama/Llama-3.2-1B --samples 50 --ctx-len 8192 --decode-len 128 --out outputs/llama_1B_8k                                                 
echo "=================OLMo-2-0425-1B, ctx=8192, decode=128==================="
python eval.py --model allenai/OLMo-2-0425-1B  --samples 50 --ctx-len 8192 --decode-len 128 --out outputs/olmo2_1B_8k