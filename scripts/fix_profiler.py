import pathlib
import re

p = pathlib.Path('src/litecascade/eval/profiler.py')
text = p.read_text()

new_code = '''    # 4. FP32 and INT8 file size
    import os
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(model.state_dict(), path)
    size_fp32_kb = os.path.getsize(path) / 1024.0
    os.remove(path)

    # Dynamic quantize linear layers as an estimate for INT8 size
    try:
        quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8)
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(quantized.state_dict(), path)
        size_int8_kb = os.path.getsize(path) / 1024.0
        os.remove(path)
    except Exception:
        size_int8_kb = size_fp32_kb / 4.0'''

old_regex = r'# 4\. FP32 and INT8 file size.*?except Exception:\s*size_int8_kb = size_fp32_kb / 4\.0'
text = re.sub(old_regex, new_code, text, flags=re.DOTALL)
p.write_text(text)
