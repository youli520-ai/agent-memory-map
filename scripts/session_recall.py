#!/usr/bin/env python3
"""会话语义召回（生产兜底）：bge-small-zh 双面语料 dense max-pool + 双面 BM25 RRF（v4 档位）。
与评估路径 semantic_recall_pilot.hybrid_eval2 同款打分（父子同根归并）。
用法（须用 emb venv 解释器，裸 python3 会 ModuleNotFoundError）:
  /opt/data/venvs/emb/bin/python /opt/data/scripts/session_recall.py "自然语言查询" [k]
输出: top-k 会话（父子链归并到每链最高分代表），每行 = 会话id | 日期 | 标题 | RRF分
索引: /opt/data/state/semantic_recall_v3_dual.npz（夜间 cron 全量重建——快、幂等，非增量）
"""
import json, sqlite3, sys, os, re, datetime
import numpy as np

sys.path.insert(0, '/opt/data/scripts')
from semantic_recall_pilot import bm25_scores  # 词法与评估路径同源，防分叉

DB = os.environ.get('SEM_DB', '/home/agent/.hermes/state.db')  # 多 profile 分库：须与索引生成侧同库
IDX = os.environ.get('SEM_IDX', '/opt/data/state/semantic_recall_v3_dual.npz')
MODEL = 'BAAI/bge-small-zh-v1.5'  # 生产档位锁死；重建侧亦钉此模型（见薄壳）
INSTRUCT = '为这个句子生成表示以用于检索相关文章：{q}'

DAY_WORDS = {"昨天": 1, "昨日": 1, "前天": 2, "大前天": 3}
# 🔴 长词优先：上上周/大上周 必须排在 上周 之前，否则子串命中短词（盲审 F8①）
SPAN_WORDS = re.compile(r"大上周|上上周|大前天|前天|昨天|昨日|前几天|前些天|这几天|前两天"
                        r"|明天|明日|下周|下个月"
                        r"|上个?月|上月|这个月|本月"
                        r"|上个?星期|上个?礼拜|上周"
                        r"|这(?:一)?个?星期|这一周|这周(?!的?[末六日天一二三四五六])|本周(?!的?[末六日天一二三四五六])")

def _span_for_word(w, today):
    if w in ("明天", "明日", "下周", "下个月"):  # 盲审 F8③ 补：未来词也入解析，与过去词矛盾时交集为空→保守不过滤
        if "月" in w:
            first = today.replace(day=1)
            nxt = (first + datetime.timedelta(days=32)).replace(day=1)
            return nxt, (nxt + datetime.timedelta(days=32)).replace(day=1) - datetime.timedelta(days=1)
        if w in ("明天", "明日"):
            d = today + datetime.timedelta(days=1); return d, d
        monday = today - datetime.timedelta(days=today.weekday())
        return monday + datetime.timedelta(days=7), monday + datetime.timedelta(days=13)
    if w in ("大上周", "上上周"):
        monday = today - datetime.timedelta(days=today.weekday())
        return monday - datetime.timedelta(days=14), monday - datetime.timedelta(days=8)
    if w in DAY_WORDS:
        d = today - datetime.timedelta(days=DAY_WORDS[w]); return d, d
    if w in ("前几天", "前些天"): return today - datetime.timedelta(days=7), today - datetime.timedelta(days=2)
    if w == "这几天": return today - datetime.timedelta(days=7), today - datetime.timedelta(days=1)
    if w == "前两天": return today - datetime.timedelta(days=3), today - datetime.timedelta(days=1)
    monday = today - datetime.timedelta(days=today.weekday())
    if w.startswith("上") and ("月" in w):
        first = today.replace(day=1)
        end = first - datetime.timedelta(days=1); return end.replace(day=1), end
    if w in ("这个月", "本月"):
        first = today.replace(day=1); return first, today
    if w.startswith("上") and ("周" in w or "星期" in w or "礼拜" in w):
        return monday - datetime.timedelta(days=7), monday - datetime.timedelta(days=1)
    if "周" in w or "星期" in w: return monday, today
    return None

def _query_span(query, today=None):
    """问句时间词 → (起,止) 日期（含）；无时间词返回 None。
    多时间词取区间交集（盲审 F8③：双时间词只认第一个会过宽）；
    已知边界（盲审 F8④，如实标注）：「这周周六」类周内具体日仍按整周处理。"""
    tz = datetime.timezone(datetime.timedelta(hours=8))
    today = today or datetime.datetime.now(tz).date()
    spans = [_span_for_word(m.group(0), today) for m in SPAN_WORDS.finditer(query)]
    spans = [s for s in spans if s]
    if not spans: return None
    lo = max(s[0] for s in spans)   # 交集
    hi = min(s[1] for s in spans)
    if lo > hi: return None         # 区间无交集 = 时间词互相矛盾，保守不过滤
    return lo, hi

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
        os.environ.setdefault('OMP_NUM_THREADS', '2')
        from fastembed import TextEmbedding
        _MODEL = TextEmbedding(model_name=MODEL, threads=2, cache_dir='/opt/data/.cache/fastembed')
    return _MODEL

def _load():
    d = np.load(IDX, allow_pickle=False)
    meta = json.loads(str(d['meta']))
    db = sqlite3.connect(DB)
    parents = dict(db.execute("SELECT id, parent_session_id FROM sessions WHERE parent_session_id IS NOT NULL"))
    return d['tvecs'], d['mvecs'], meta['titles'], meta['msg_sess'], parents, db

def _dual_docs(db):
    """与 build3 同构的双面词法语料：标题（含别名） + user 全量 + assistant>80字（cap 300）。"""
    sess = db.execute("""SELECT id, title FROM sessions WHERE source IN ('feishu','tui')
                         ORDER BY started_at""").fetchall()
    sid_idx = {s[0]: i for i, s in enumerate(sess)}
    docs, doc_sess = [], []
    apath = '/opt/data/state/semantic_aliases.json'
    aliases = json.load(open(apath, encoding='utf-8')) if os.path.exists(apath) else {}
    for i, s in enumerate(sess):
        extra = ' '.join(' '.join(v) if isinstance(v, list) else str(v)
                         for k, v in aliases.items() if s[0].startswith(k))
        docs.append(s[1] + (('。' + extra) if extra else ''))
        doc_sess.append(i)
    rows = db.execute("""SELECT m.session_id, m.role, m.content FROM messages m
                         JOIN sessions s ON m.session_id=s.id
                         WHERE s.source IN ('feishu','tui')
                           AND (m.role='user' OR (m.role='assistant' AND length(m.content)>80))
                           AND m.content NOT LIKE '[System:%' ORDER BY m.id""").fetchall()
    for sid, role, content in rows:
        i = sid_idx.get(sid, -1)
        if i < 0: continue
        docs.append(('问：' if role == 'user' else '答：') + content[:300])
        doc_sess.append(i)
    return docs, doc_sess

def _root_of(sid, parents):
    seen, cur = set(), sid
    while cur in parents and cur not in seen:
        seen.add(cur); cur = parents[cur]
    return cur

def score(q, k=5):
    """返回 top-k 会话 [(id, date, title, rrf_score)]，父子同根归并。"""
    tvecs, mvecs, titles, msg_sess, parents, db = _load()
    model = _get_model()
    qv = np.array(list(model.embed([INSTRUCT.format(q=q[:2000])])), dtype=np.float32)  # 🔴 盲审 F19：查询显式限长
    if qv.shape[1] != tvecs.shape[1]:
        raise RuntimeError(f'查询向量维度 {qv.shape[1]} 与索引 {tvecs.shape[1]} 不一致——索引/模型错配，请重跑夜间重建薄壳')
    qv /= (np.linalg.norm(qv) + 1e-9)
    n_sess = len(titles)
    dsess = np.full(n_sess, -1.0, dtype=np.float32)
    np.maximum.at(dsess, np.array(msg_sess), mvecs @ qv[0])
    dsess = np.maximum(dsess, tvecs @ qv[0])
    docs, doc_sess = _dual_docs(db)
    ls = bm25_scores(docs, q)
    lsess = np.full(n_sess, -1.0, dtype=np.float32)
    np.maximum.at(lsess, np.array(doc_sess), np.array(ls, dtype=np.float32))
    # 时间词过滤：问句含昨天/上周/上个月等时仅保留日期区间内的会话；候选不足 k 自动放宽
    span = _query_span(q)
    cand = None
    if span:
        lo, hi = span
        cand = {i for i, t in enumerate(titles) if lo.isoformat() <= t.get('date', '') <= hi.isoformat()}
        if len(cand) < k: cand = None
    fused = {}
    for ranking in (dsess.argsort()[::-1], lsess.argsort()[::-1]):
        for rank, si in enumerate(ranking):
            if cand is not None and si not in cand: continue
            fused[si] = fused.get(si, 0.0) + 1.0/(60+rank)
    top = sorted(fused, key=lambda s: -fused[s])[:k*2]  # 取 2k 再归并根，保证根层仍够 k 条
    # 通道强提名保位：单通道断崖领先的会话（如词法第一名）不该被融合平均淹没
    # 🔴 契约（盲审 F5 教训）：返回 = top-k + 显式标记的强提名（5 元组末位 nominated=True，最多 2 条），不再静默超额
    started = {t['id']: t['date'] for t in titles}
    dr = dsess.argsort()[::-1][:1]
    lr2 = lsess.argsort()[::-1][:1]
    out, roots_in = [], set()
    for i in top:
        sid = titles[i]['id']
        r = _root_of(sid, parents)
        if r in roots_in: continue
        roots_in.add(r)
        out.append((sid, started.get(sid, '?'), titles[i]['title'], fused[i], False))
        if len(out) >= k: break
    for si in (*lr2, *dr):
        if cand is not None and si not in cand: continue
        r = _root_of(titles[si]['id'], parents)
        if r in roots_in: continue
        roots_in.add(r)
        out.append((titles[si]['id'], started.get(titles[si]['id'], '?'),
                    titles[si]['title'], fused.get(si, 0.0), True))
    return out

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    q = sys.argv[1]
    try:
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    except ValueError:
        print('k 必须是整数'); sys.exit(2)
    if k < 1:
        print('k 必须 >=1'); sys.exit(2)
    try:
        rows = score(q, k)
    except FileNotFoundError:
        print(f'索引不存在：{IDX}——先跑夜间重建薄壳 semantic_recall_build.py'); sys.exit(3)
    except RuntimeError as e:
        print(str(e)); sys.exit(3)
    for sid, date, title, s, nom in rows:
        mark = ' 🔧强提名' if nom else ''
        print(f"{sid} | {date} | {title} | {s:.3f}{mark}")

if __name__ == '__main__':
    main()
