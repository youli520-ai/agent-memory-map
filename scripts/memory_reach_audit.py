#!/usr/bin/env python3
"""记忆可达性体检（常驻哨兵）：xuyi consolidate 式——确定性重推导、0 模型调用、幂等。
两层：
  A 结构体检（自动全覆盖）：L0 条目承载/L1 文件/INDEX.md 三方一致性——悬空引用、孤儿文件
  B 探针抽检（题库 /opt/data/memory/probes.json，人工维护真实问法）：
      走真实检索路径（INDEX → grep L0/L1）验证「存了 ≠ 找得回」
三态（xuyi 同款）：当前候选 / 存而不达 / 已被更替（条目带 [OBSOLETE]/wrong 标记）
退出码：0=全部可达 1=存在存而不达（喂 cron_watchdog 分诊）
用法：无参=全量体检（cron 契约）；--json=输出 JSON
"""
import json, os, re, sys, datetime

MEM_DIR = '/opt/data/memory'
L0 = '/home/agent/.hermes/memories/MEMORY.md'  # 真实 L0 位置（≠ /opt/data/memory/，那里只放 L1 详情）
L1_DIRS = [os.path.join(MEM_DIR, 'default'), os.path.join(MEM_DIR, 'maojiu'), os.path.join(MEM_DIR, 'lingyuan')]
INDEX = '/opt/data/INDEX.md'
PROBES = '/opt/data/memory/probes.json'
STATE = '/opt/data/state/memory_reach_state.json'

def _read(p):
    try:
        return open(p, encoding='utf-8').read()
    except Exception:
        return None

def structural_audit():
    """A 层：L0 条目 ↔ 承载 ↔ INDEX 一致性。返回问题清单。"""
    issues = []
    l0 = _read(L0) or ''
    index = _read(INDEX) or ''
    l1_files = {}
    for d in L1_DIRS:
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            if f.endswith('.md'):
                l1_files[f] = os.path.join(d, f)
    # ① INDEX.md 引用的文件必须存在
    for ref in set(re.findall(r'(/opt/data/memory/[^\s)`]+\.md|memory/(?:default|maojiu|lingyuan)/[^\s)`]+\.md)', index)):
        rp = ref if ref.startswith('/') else os.path.join('/opt/data', ref)
        if not os.path.exists(rp):
            issues.append(f'INDEX 悬空引用: {ref}')
    # ② L1 文件应被 INDEX 收录（孤儿文件=存了但索引找不回）
    for f in l1_files:
        if f'/{f}' not in index and f not in index:
            issues.append(f'L1 孤儿文件（INDEX 未收录）: {f}')
    # ③ L0 指针条目指向的详情文件必须存在（→memory/default/xxx.md 形态）
    for ref in set(re.findall(r'→\s*memory/(default|maojiu|lingyuan)/([^\s）)、,，]+\.md)', l0)):
        rp = os.path.join(MEM_DIR, ref[0], ref[1])
        if not os.path.exists(rp):
            issues.append(f'L0 指针悬空: memory/{ref[0]}/{ref[1]}')
    return issues, l1_files, index

def probe_audit():
    """B 层：探针走真实检索路径（INDEX 关键词 → grep L0/L1 全文）。"""
    probes = json.loads(_read(PROBES) or '[]')
    corpus = {}
    l0 = _read(L0) or ''
    if l0:
        corpus[L0] = l0  # 🔴 L0 显式入语料（不在 MEM_DIR 下）
    if os.path.isdir(MEM_DIR):
        for root, _dirs, files in os.walk(MEM_DIR):
            if 'benchmarks' in root or 'candidates' in root or 'sessions' in root: continue
            for f in files:
                if f.endswith('.md'):
                    p = os.path.join(root, f)
                    corpus[p] = (_read(p) or '')
    results = []
    for pr in probes:
        q = pr.get('probe', '')
        expect = pr.get('expect_file', '')
        aliases = [a.lower() for a in pr.get('aliases', []) if a]
        # 真实检索路径①：INDEX 关键词命中
        index = (_read(INDEX) or '').lower()
        hit_index = any(a in index for a in aliases) if aliases else False
        # ②全文 grep（L0+L1 语料）
        hit_files = [os.path.relpath(p, MEM_DIR) for p, t in corpus.items()
                     if any(a in t.lower() for a in aliases)]
        expect_ok = any(h.endswith(expect.split('/')[-1]) for h in hit_files) if expect else bool(hit_files)
        # 已被更替：预期承载文件内容里带 OBSOLETE/wrong 标记
        carrier = next((p for p, t in corpus.items() if expect.split('/')[-1] in os.path.relpath(p, MEM_DIR)), None)
        superseded = bool(carrier and re.search(r'\[OBSOLETE|已更替|wrong', corpus[carrier] or ''))
        if superseded:
            state = '已被更替'
        elif expect_ok or hit_index:
            state = '当前候选'
        else:
            state = '存而不达'
        results.append({'probe': q, 'expect': expect, 'state': state,
                        'hit_files': hit_files[:3], 'hit_total': len(hit_files), 'hit_index': hit_index})
    return results

def map_audit():
    """C 层（新增，盲审修订 5/9）：地图一致性——地图引用的承载件必须存在。"""
    issues = []
    l0 = _read(L0) or ''
    if not l0:
        return ['L0 地图文件读取失败'] 
    # ①地图行引用的 memory/*.md 必须存在（L0 已地图化，引用即导航边）
    for ref in set(re.findall(r'memory/(default|maojiu|lingyuan)/([^\s）)、,，]+\.md)', l0)):
        rp = os.path.join(MEM_DIR, ref[0], ref[1])
        if not os.path.exists(rp):
            issues.append(f'地图悬空引用: memory/{ref[0]}/{ref[1]}')
    # ②枢纽文件必须存在且被 INDEX 收录
    for hub in ('channels.md',):
        rp = os.path.join(MEM_DIR, 'default', hub)
        if not os.path.exists(rp):
            issues.append(f'地图枢纽缺失: {hub}')
        elif hub not in (_read(INDEX) or ''):
            issues.append(f'地图枢纽未进 INDEX: {hub}')
    # ③探针题库必须存在
    if not os.path.exists(PROBES):
        issues.append('探针题库缺失: probes.json')
    return issues

def capacity_audit():
    """D 层（新增，盲审三 F「涨满维护点」）：L0 容量监测——连续分数+分级。
    阈值：<75% 正常静默 / ≥75% warning / ≥90% critical 计入 issues（推卡）。"""
    l0 = _read(L0) or ''
    chars = len(l0)
    budget = 8000
    ratio = chars / budget
    level = 'normal' if ratio < 0.75 else ('warning' if ratio < 0.9 else 'critical')
    cap_issue = None
    if level == 'critical':
        cap_issue = f'L0 容量 critical：{chars}/{budget} 字符（{ratio:.0%}）——触发治理：合并归组/降级休眠/淘汰，禁砍类报用户'
    elif level == 'warning':
        cap_issue = f'L0 容量 warning：{chars}/{budget} 字符（{ratio:.0%}）——下次写入优先合并归组，勿新增独立行'
    return {'chars': chars, 'budget': budget, 'ratio': round(ratio, 3), 'level': level,
            'advice': cap_issue}

def main():
    issues, l1_files, _ = structural_audit()
    issues += map_audit()
    capacity = capacity_audit()
    if capacity['level'] == 'critical' and capacity['advice']:
        issues.append(capacity['advice'])
    probes = probe_audit()
    unreachable = [r for r in probes if r['state'] == '存而不达']
    superseded = [r for r in probes if r['state'] == '已被更替']
    report = {
        'ts': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'structural_issues': issues,
        'probe_total': len(probes),
        'unreachable': unreachable,
        'superseded': superseded,
        'reachable': len(probes) - len(unreachable) - len(superseded),
        'l1_files': len(l1_files),
        'capacity': capacity,
    }
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    if '--json' in sys.argv:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        sys.exit(1 if (unreachable or issues) else 0)
    # 人读摘要：无问题静默（cron watchdog 形态）
    if not unreachable and not issues:
        print(f"✅ 可达性体检通过：探针 {len(probes)}/{len(probes)} 可达，结构 0 问题，L1 {len(l1_files)} 文件；L0 容量 {capacity['chars']}/{capacity['budget']}（{capacity['ratio']:.0%}，{capacity['level']}）")
        sys.exit(0)
    print(f"⚠️ 可达性体检异常：结构问题 {len(issues)} · 存而不达 {len(unreachable)} · L0 容量 {capacity['ratio']:.0%}（{capacity['level']}）")
    for i in issues: print(f'  [结构] {i}')
    for r in unreachable: print(f"  [不达] {r['probe']} → 预期 {r['expect']}，命中 {r['hit_files'] or '无'}")
    sys.exit(1)

if __name__ == '__main__':
    main()
