```sh
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly
python compile.py
python infer.py
```

note：
add `Array` in `/site-packages/tvm/relax/utils.py:L120`.