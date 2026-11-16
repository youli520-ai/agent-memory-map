#!/usr/bin/env python3
"""会话检索语义层试点与生产引擎：bge-small-zh 嵌入（threads 可钉）、20 题基准评估、多种融合模式对打。
现役生产语料=build3 双面（user 全量 + assistant>80 字 cap300），索引 /opt/data/state/semantic_recall_v3_dual.npz。
生产召回走 session_recall.py（复用本模块 bm25_scores）；⚠️ 盲审 F11：score() 与 hybrid_eval2 同款基础打分
（双面 dense+双面词法 RRF、父子同根），但生产附加时间词过滤与通道强提名保位——评估分数不完全外推生产。
  build  # v1 摘要级索引（历史基准复现用）
  build2 # v2 用户消息级索引（历史基准复现用）
  build3 # v3 双面语料索引（生产现役）
  eval / eval2 [k] [tag]   # 20 问评估（v1 索引 / v2 索引）
  hybrid [k] [tag]         # v2 索引 + BM25 RRF
  hybrid2 [k] [tag] [mode] # 双面词法 RRF（rrf）或两段式（two_stage），父子同根计分
  q [query] / q2 [query]   # 单查询演示
基准题库: /opt/data/memory/benchmarks/session-recall-20q.md
"""
import json, sqlite3, sys, datetime, os, re

DB = os.environ.get('SEM_DB', '/home/agent/.hermes/state.db')  # 多 profile 分库：SEM_DB 指向 profiles/<名>/state.db
OUT = '/opt/data/state/semantic_recall.npz'
BENCH = '/opt/data/memory/benchmarks/session-recall-20q.md'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

def load_sessions():
    db = sqlite3.connect(DB)
    tz = datetime.timezone(datetime.timedelta(hours=8))
    rows = db.execute("""SELECT id, title, source, started_at FROM sessions
                         WHERE source IN ('feishu','tui') ORDER BY started_at""").fetchall()
    corpus = []
    for sid, title, src, ts in rows:
        date = datetime.datetime.fromtimestamp(ts, tz).strftime('%Y-%m-%d')
        first = db.execute("""SELECT substr(content,1,200) FROM messages
                              WHERE session_id=? AND role='user' AND content NOT LIKE '[System:%'
                              ORDER BY id LIMIT 1""", (sid,)).fetchone()
        last = db.execute("""SELECT substr(content,1,600) FROM messages
                             WHERE session_id=? AND role='assistant' AND length(content)>50
                             ORDER BY id DESC LIMIT 1""", (sid,)).fetchone()
        text = f"会话（{date}，{src}）：{title}。用户首问：{(first[0] if first else '')}。结论摘要：{(last[0] if last else '')}"
        corpus.append({'id': sid, 'title': title, 'date': date, 'text': text})
    return corpus

def get_model():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=MODEL, threads=int(os.environ.get('EMB_THREADS', '0')) or None,
                         cache_dir=os.environ.get('EMB_CACHE', None) or None)

MUNDANE = {"天气","时间","时候","东西","问题","地方","感觉","事情","工作","学习","吃饭","睡觉",
           "视频","照片","今天","明天","昨天","手机","电脑","大家","哪个","什么","怎么","最后",
           "之前","上次","前面","说过","回来","知道","找不","多少","能不能","是不是"}

def terms(text):
    """xuyi 同款词法：英文 token + 中文单字 + 二元组。"""
    result = re.findall(r"[a-z0-9]+", text.lower())
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        result.extend(run)
        result.extend(run[i:i+2] for i in range(len(run)-1))
    return result

def bm25_scores(texts, query):
    from collections import Counter
    import math
    docs = [Counter(t for t in terms(x) if t not in MUNDANE and len(t) > 1) for x in texts]
    n = len(docs)
    avg = sum(sum(d.values()) for d in docs)/max(n, 1) or 1
    scores = [0.0]*n
    for term in sorted({t for t in terms(query) if t not in MUNDANE and len(t) > 1}):
        df = sum(term in d for d in docs)
        if not df: continue
        idf = math.log1p((n-df+0.5)/(df+0.5))
        for i, doc in enumerate(docs):
            tf = doc.get(term, 0)
            if tf: scores[i] += idf*2.2*tf/(tf+1.2*(0.25+0.75*sum(doc.values())/avg))
    return scores

def load_texts():
    """与 build2 完全同序重建语料文本（消息 + 标题）。"""
    db = sqlite3.connect(DB)
    tz = datetime.timezone(datetime.timedelta(hours=8))
    sess = db.execute("""SELECT id, title, started_at FROM sessions
                         WHERE source IN ('feishu','tui') ORDER BY started_at""").fetchall()
    sid_idx = {s[0]: i for i, s in enumerate(sess)}
    ttexts = [f"会话（{datetime.datetime.fromtimestamp(s[2], tz).strftime('%Y-%m-%d')}）：{s[1]}" for s in sess]
    msg_rows = db.execute("""SELECT m.session_id, m.content FROM messages m
                             JOIN sessions s ON m.session_id=s.id
                             WHERE s.source IN ('feishu','tui') AND m.role='user'
                             AND m.content NOT LIKE '[System:%'
                             ORDER BY m.id""").fetchall()
    mtexts = ['问：' + r[1][:500] for r in msg_rows]
    msg_sess = [sid_idx.get(r[0], -1) for r in msg_rows]
    return ttexts, mtexts, msg_sess, len(sess)

def load_parents():
    db = sqlite3.connect(DB)
    return dict(db.execute("SELECT id, parent_session_id FROM sessions WHERE parent_session_id IS NOT NULL"))

def root_of(sid, parents):
    seen = set()
    cur = sid
    while cur in parents and cur not in seen:
        seen.add(cur)
        cur = parents[cur]
    return cur

def hybrid_eval2(k=3, tag='v4', mode='rrf'):
    """双面语料词法 + 双面 dense，两种融合模式：
    rrf = 对称 RRF；two_stage = xuyi 原版（词法主窗口→dense 窗口内重排→RRF）"""
    import numpy as np
    tvecs, mvecs, titles, msg_sess_v = load_index2()
    db = sqlite3.connect(DB)
    tz = datetime.timezone(datetime.timedelta(hours=8))
    sess = db.execute("""SELECT id, title, started_at FROM sessions
                         WHERE source IN ('feishu','tui') ORDER BY started_at""").fetchall()
    sid_idx = {s[0]: i for i, s in enumerate(sess)}
    ttexts = [f"会话（{datetime.datetime.fromtimestamp(s[2], tz).strftime('%Y-%m-%d')}）：{s[1]}" for s in sess]
    rows = db.execute("""SELECT m.session_id, m.role, m.content FROM messages m
                         JOIN sessions s ON m.session_id=s.id
                         WHERE s.source IN ('feishu','tui')
                           AND (m.role='user' OR (m.role='assistant' AND length(m.content)>80))
                           AND m.content NOT LIKE '[System:%' ORDER BY m.id""").fetchall()
    docs = list(ttexts) + [('问：' if r[1]=='user' else '答：') + r[2][:300] for r in rows]
    doc_sess = list(range(len(sess))) + [sid_idx.get(r[0], -1) for r in rows]
    n_sess = len(sess)
    model = get_model()
    qs = []
    for line in open(BENCH, encoding='utf-8'):
        m = re.match(r'\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(\S+)\s*\|', line)
        if m and m.group(3).startswith('2026'):
            qs.append((int(m.group(1)), m.group(2), m.group(3)))
    print('词法文档数:', len(docs))
    parents = load_parents()
    gt_roots = {n: root_of(gt, parents) for n, _, gt in qs}
    hit = 0
    for n, q, gt in qs:
        qv = np.array(list(model.embed([INSTRUCT.format(q=q)])), dtype=np.float32)
        qv /= (np.linalg.norm(qv) + 1e-9)
        dsess = np.full(n_sess, -1.0, dtype=np.float32)
        np.maximum.at(dsess, np.array(msg_sess_v), mvecs @ qv[0])
        dsess = np.maximum(dsess, tvecs @ qv[0])
        ls = bm25_scores(docs, q)
        lsess = np.full(n_sess, -1.0, dtype=np.float32)
        np.maximum.at(lsess, np.array(doc_sess), np.array(ls, dtype=np.float32))
        if mode == 'rrf':
            fused = {}
            for ranking in (dsess.argsort()[::-1], lsess.argsort()[::-1]):
                for rank, si in enumerate(ranking):
                    fused[si] = fused.get(si, 0.0) + 1.0/(60+rank)
            top = sorted(fused, key=lambda s: -fused[s])[:k]
        else:  # two_stage: 词法排序取前 30 会话为窗口，dense 在窗口内重排，RRF 融合两序
            lr = lsess.argsort()[::-1][:30]
            dr = dsess.argsort()[::-1]
            dr_pos = {int(s): p for p, s in enumerate(dr)}
            fused = {}
            for p, s in enumerate(lr):
                fused[int(s)] = fused.get(int(s), 0.0) + 1.0/(60+p+1)
            for s in lr:
                fused[int(s)] = fused.get(int(s), 0.0) + 1.0/(60+dr_pos.get(int(s), 200)+1)
            top = sorted(fused, key=lambda s: -fused[s])[:k]
        res = [(titles[i]['id'], titles[i]['title'], fused[i]) for i in top]
        ok = any(root_of(i, parents) == gt_roots[n] for i, _, _ in res)
        hit += ok
        if not ok:
            print(f"Q{n:02d} ❌ | gt={gt[:15]}(root={gt_roots[n][:15]}) | {' / '.join(f'{i[:15]}({s:.2f})' for i,_,s in res)}")
    print(f"\n{tag}（k={k}，mode={mode}，父子同根计分）成绩：{hit}/{len(qs)}（strict 计分 dense-only 10/20）")

def hybrid_eval(k=3, tag='v3-hybrid'):
    """小模型 dense（消息级 max-pool）+ BM25 词法，RRF 融合。"""
    import numpy as np
    tvecs, mvecs, titles, msg_sess_v = load_index2()
    ttexts, mtexts, msg_sess, n_sess = load_texts()
    model = get_model()
    qs = []
    for line in open(BENCH, encoding='utf-8'):
        m = re.match(r'\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(\S+)\s*\|', line)
        if m and m.group(3).startswith('2026'):
            qs.append((int(m.group(1)), m.group(2), m.group(3)))
    docs = ttexts + mtexts
    doc_sess = list(range(n_sess)) + msg_sess
    hit = 0
    for n, q, gt in qs:
        qv = np.array(list(model.embed([INSTRUCT.format(q=q)])), dtype=np.float32)
        qv /= (np.linalg.norm(qv) + 1e-9)
        # dense 会话分
        dsess = np.full(n_sess, -1.0, dtype=np.float32)
        np.maximum.at(dsess, np.array(msg_sess_v), mvecs @ qv[0])
        dsess = np.maximum(dsess, tvecs @ qv[0])
        # 词法会话分
        ls = bm25_scores(docs, q)
        lsess = np.full(n_sess, -1.0, dtype=np.float32)
        np.maximum.at(lsess, np.array(doc_sess), np.array(ls, dtype=np.float32))
        # RRF 融合
        fused = {}
        for ranking in (dsess.argsort()[::-1], lsess.argsort()[::-1]):
            for rank, si in enumerate(ranking):
                fused[si] = fused.get(si, 0.0) + 1.0/(60+rank)
        top = sorted(fused, key=lambda s: -fused[s])[:k]
        res = [(titles[i]['id'], titles[i]['title'], fused[i]) for i in top]
        ok = any(i.startswith(gt[:15]) for i, _, _ in res)
        hit += ok
        if not ok:
            print(f"Q{n:02d} ❌ | gt={gt[:15]} | {' / '.join(f'{i[:15]}({s:.2f})' for i,_,s in res)}")
    print(f"\n语义层 {tag}（k={k}）成绩：{hit}/{len(qs)}（dense-only 8/20，FTS 0/20）")

def build():
    import numpy as np
    corpus = load_sessions()
    print('语料数:', len(corpus))
    model = get_model()
    texts = [c['text'] for c in corpus]
    vecs = np.array(list(model.embed(texts, batch_size=32)), dtype=np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    np.savez_compressed(OUT, vecs=vecs,
                        meta=json.dumps([{k: c[k] for k in ('id','title','date')} for c in corpus], ensure_ascii=False))
    print('嵌入完成:', vecs.shape, '→', OUT)

def load_index():
    import numpy as np
    d = np.load(OUT, allow_pickle=False)
    return d['vecs'], json.loads(str(d['meta']))

def query(q, k=3):
    import numpy as np
    vecs, meta = load_index()
    model = get_model()
    qv = np.array(list(model.embed([INSTRUCT.format(q=q)])), dtype=np.float32)
    qv /= (np.linalg.norm(qv) + 1e-9)
    scores = vecs @ qv[0]
    top = scores.argsort()[::-1][:k]
    return [(meta[i]['id'], meta[i]['title'], float(scores[i])) for i in top]

def eval20():
    import re
    qs = []
    for line in open(BENCH, encoding='utf-8'):
        m = re.match(r'\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(\S+)\s*\|', line)
        if m and m.group(3).startswith('2026'):
            qs.append((int(m.group(1)), m.group(2), m.group(3)))
    print('题目数:', len(qs))
    hit = 0
    for n, q, gt in qs:
        res = query(q)
        ids = [r[0] for r in res]
        ok = any(i.startswith(gt[:15]) for i in ids)
        hit += ok
        print(f"Q{n:02d} {'✅' if ok else '❌'} | {' / '.join(f'{i[:15]}({s:.2f})' for i,_,s in res)}")
    print(f"\n语义层成绩：{hit}/{len(qs)}（FTS 基线 0/20）")

OUT2 = os.environ.get('EMB_OUT2', '/opt/data/state/semantic_recall_v2.npz')
MODEL = os.environ.get('EMB_MODEL', 'BAAI/bge-small-zh-v1.5')
INSTRUCT = ('Instruct: 给定一个用户回忆过往对话的搜索请求，检索最相关的历史会话\nQuery: {q}'
            if 'Qwen3' in MODEL else
            '为这个句子生成表示以用于检索相关文章：{q}')

def build3():
    """v3：用户消息 + 助手长消息（>80字，cap 300字）双面语料，消息级 max-pool + 标题向量。"""
    import numpy as np
    db = sqlite3.connect(DB)
    tz = datetime.timezone(datetime.timedelta(hours=8))
    sess = db.execute("""SELECT id, title, started_at FROM sessions
                         WHERE source IN ('feishu','tui') ORDER BY started_at""").fetchall()
    sid_idx = {s[0]: i for i, s in enumerate(sess)}
    tmeta = [{'id': s[0], 'title': s[1],
              'date': datetime.datetime.fromtimestamp(s[2], tz).strftime('%Y-%m-%d')} for s in sess]
    rows = db.execute("""SELECT m.session_id, m.role, m.content FROM messages m
                         JOIN sessions s ON m.session_id=s.id
                         WHERE s.source IN ('feishu','tui')
                           AND (m.role='user' OR (m.role='assistant' AND length(m.content)>80))
                           AND m.content NOT LIKE '[System:%'
                         ORDER BY m.id""").fetchall()
    meta, texts = [], []
    for sid, role, content in rows:
        i = sid_idx.get(sid, -1)
        if i < 0: continue
        tag = '问：' if role == 'user' else '答：'
        texts.append(tag + content[:300])
        meta.append(i)
    aliases = {}
    apath = '/opt/data/state/semantic_aliases.json'
    if os.path.exists(apath):
        aliases = json.load(open(apath, encoding='utf-8'))
    ttexts = []
    for m in tmeta:
        extra = ' '.join(' '.join(v) if isinstance(v, list) else str(v)
                         for k, v in aliases.items() if m['id'].startswith(k))
        ttexts.append(f"会话（{m['date']}）：{m['title']}。{extra}")
    model = get_model()
    print(f'嵌入 {len(ttexts)} 标题 + {len(texts)} 双面消息 ...')
    tvecs = np.array(list(model.embed(ttexts, batch_size=64)), dtype=np.float32)
    mvecs = np.array(list(model.embed(texts, batch_size=64)), dtype=np.float32)
    tvecs /= (np.linalg.norm(tvecs, axis=1, keepdims=True) + 1e-9)
    mvecs /= (np.linalg.norm(mvecs, axis=1, keepdims=True) + 1e-9)
    np.savez_compressed(OUT2, tvecs=tvecs, mvecs=mvecs,
                        meta=json.dumps({'titles': tmeta, 'msg_sess': meta}, ensure_ascii=False))
    print('完成:', tvecs.shape, mvecs.shape, '→', OUT2)

def build2():
    """消息级嵌入：每条 user 消息一个向量（cap 500 字）+ 每会话标题向量；会话分=max。"""
    import numpy as np
    db = sqlite3.connect(DB)
    tz = datetime.timezone(datetime.timedelta(hours=8))
    sess = db.execute("""SELECT id, title, started_at FROM sessions
                         WHERE source IN ('feishu','tui') ORDER BY started_at""").fetchall()
    sid_idx = {s[0]: i for i, s in enumerate(sess)}
    tmeta = [{'id': s[0], 'title': s[1],
              'date': datetime.datetime.fromtimestamp(s[2], tz).strftime('%Y-%m-%d')} for s in sess]
    msg_rows = db.execute("""SELECT m.session_id, m.content FROM messages m
                             JOIN sessions s ON m.session_id=s.id
                             WHERE s.source IN ('feishu','tui') AND m.role='user'
                             AND m.content NOT LIKE '[System:%'
                             ORDER BY m.id""").fetchall()
    mmeta = [sid_idx.get(r[0], -1) for r in msg_rows]
    mtexts = [('问：' + r[1][:500]) for r in msg_rows]
    ttexts = [f"会话（{m['date']}）：{m['title']}" for m in tmeta]
    model = get_model()
    print(f'嵌入 {len(ttexts)} 标题 + {len(mtexts)} 消息 ...')
    tvecs = np.array(list(model.embed(ttexts, batch_size=64)), dtype=np.float32)
    mvecs = np.array(list(model.embed(mtexts, batch_size=64)), dtype=np.float32)
    tvecs /= (np.linalg.norm(tvecs, axis=1, keepdims=True) + 1e-9)
    mvecs /= (np.linalg.norm(mvecs, axis=1, keepdims=True) + 1e-9)
    np.savez_compressed(OUT2, tvecs=tvecs, mvecs=mvecs,
                        meta=json.dumps({'titles': tmeta, 'msg_sess': mmeta}, ensure_ascii=False))
    print('完成:', tvecs.shape, mvecs.shape, '→', OUT2)

def load_index2():
    import numpy as np
    d = np.load(OUT2, allow_pickle=False)
    meta = json.loads(str(d['meta']))
    return d['tvecs'], d['mvecs'], meta['titles'], meta['msg_sess']

def query2(q, k=3):
    import numpy as np
    tvecs, mvecs, titles, msg_sess = load_index2()
    model = get_model()
    qv = np.array(list(model.embed([INSTRUCT.format(q=q)])), dtype=np.float32)
    qv /= (np.linalg.norm(qv) + 1e-9)
    n_sess = len(titles)
    sess_score = np.full(n_sess, -1.0, dtype=np.float32)
    np.maximum.at(sess_score, np.array(msg_sess), mvecs @ qv[0])
    sess_score = np.maximum(sess_score, tvecs @ qv[0])
    top = sess_score.argsort()[::-1][:k]
    return [(titles[i]['id'], titles[i]['title'], float(sess_score[i])) for i in top]

def eval20_v2(k=3, tag='v2'):
    import re
    qs = []
    for line in open(BENCH, encoding='utf-8'):
        m = re.match(r'\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(\S+)\s*\|', line)
        if m and m.group(3).startswith('2026'):
            qs.append((int(m.group(1)), m.group(2), m.group(3)))
    hit = 0
    for n, q, gt in qs:
        res = query2(q, k=k)
        ok = any(i.startswith(gt[:15]) for i, _, _ in res)
        hit += ok
        if not ok:
            print(f"Q{n:02d} ❌ | gt={gt[:15]} | {' / '.join(f'{i[:15]}({s:.2f})' for i,_,s in res)}")
    print(f"\n语义层 {tag}（k={k}）成绩：{hit}/{len(qs)}（FTS 0/20）")

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'eval'
    if cmd == 'build':
        build()
    elif cmd == 'eval':
        eval20()
    elif cmd == 'build2':
        build2()
    elif cmd == 'build3':
        build3()
    elif cmd == 'eval2':
        eval20_v2(k=int(sys.argv[2]) if len(sys.argv) > 2 else 3,
                  tag=sys.argv[3] if len(sys.argv) > 3 else 'v2')
    elif cmd == 'q':
        for r in query(sys.argv[2]):
            print(r)
    elif cmd == 'hybrid':
        hybrid_eval(k=int(sys.argv[2]) if len(sys.argv) > 2 else 3,
                    tag=sys.argv[3] if len(sys.argv) > 3 else 'v3-hybrid')
    elif cmd == 'hybrid2':
        hybrid_eval2(k=int(sys.argv[2]) if len(sys.argv) > 2 else 3,
                     tag=sys.argv[3] if len(sys.argv) > 3 else 'v4',
                     mode=sys.argv[4] if len(sys.argv) > 4 else 'rrf')
    elif cmd == 'q2':
        for r in query2(sys.argv[2]):
            print(r)
    else:
        print('未知子命令:', cmd, '\n可用: build build2 build3 eval eval2 [k] [tag] hybrid [k] [tag] hybrid2 [k] [tag] [mode] q q2', file=sys.stderr)
        sys.exit(2)
