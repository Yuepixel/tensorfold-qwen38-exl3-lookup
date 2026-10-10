import json, glob, os
root = "/models/prefix-spill"
for d in glob.glob(root + "/*"):
    if not os.path.isdir(d):
        continue
    print("DIR", os.path.basename(d))
    for jf in sorted(glob.glob(d + "/*.json")):
        m = json.load(open(jf))
        st = jf[:-5] + ".safetensors"
        sz = os.path.getsize(st) if os.path.exists(st) else -1
        ids = m.get("ids", [])
        print("  %s pos=%s len(ids)=%s mtp_len=%s kv=%s layers=%s q=%s mtp=%s saved=%s file=%dMB head=%s" % (
            os.path.basename(jf)[:12], m.get("pos"), len(ids), m.get("mtp_len"),
            m.get("kv_dtype"), m.get("layers"), m.get("quantized"), m.get("mtp"),
            m.get("saved"), sz // 1048576, ids[:6]))
