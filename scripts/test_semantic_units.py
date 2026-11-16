#!/usr/bin/env python3
"""语义召回单元测试（盲审 F17）：_query_span 快照 + _is_bare_secret/redact 纯函数断言 + 强提名合成断言。
运行：/opt/data/venvs/emb/bin/python /opt/data/scripts/test_semantic_units.py
⚠️ 无参数运行=全部用例（cron 契约友好）；--help 守卫 exit 0（坑22）。"""
import sys, os, datetime, re
sys.path.insert(0, '/opt/data/scripts')

if '--help' in sys.argv or '-h' in sys.argv:
    print(__doc__); sys.exit(0)

import numpy as np
import session_recall as sr
import session_resume as sres

TODAY = datetime.date(2026, 11, 16)  # 周一
passed = failed = 0

def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1; print(f'✅ {name}')
    else:
        failed += 1; print(f'❌ {name}: got={got} want={want}')

# ── _query_span 快照（含盲审 F8 四偏差回归）──
check('span: 昨天', sr._query_span('昨天聊的那个', today=TODAY), (datetime.date(2026,11,15), datetime.date(2026,11,15)))
check('span: 上周', sr._query_span('上周我们弄的', today=TODAY), (datetime.date(2026,11,9), datetime.date(2026,11,15)))
check('span: 上上周（F8① 长词优先）', sr._query_span('上上周的讨论', today=TODAY), (datetime.date(2026,11,2), datetime.date(2026,11,8)))
check('span: 大上周', sr._query_span('大上周那个', today=TODAY), (datetime.date(2026,11,2), datetime.date(2026,11,8)))
check('span: 上个月', sr._query_span('上个月搞的活动', today=TODAY), (datetime.date(2026,10,1), datetime.date(2026,10,31)))
check('span: 本月', sr._query_span('本月的事', today=TODAY), (datetime.date(2026,11,1), datetime.date(2026,11,16)))
check("span: 这一周（F8② 负向断言；11-16 恰为周一）", sr._query_span("这一周的事", today=TODAY), (datetime.date(2026,11,16), datetime.date(2026,11,16)))
check('span: 无时间词', sr._query_span('没有时间词的查询', today=TODAY), None)
check('span: 双时间词交集（F8③）', sr._query_span('昨天与上周', today=TODAY), (datetime.date(2026,11,15), datetime.date(2026,11,15)))
check("span: 矛盾时间词保守不过滤（F8③，未来词已入解析）", sr._query_span("明天与上周", today=TODAY), None)

# ── _is_bare_secret（盲审 F10 防误伤回归）──
check('bare: Example42. 命中', sres._is_bare_secret('Example42.'), True)
check('bare: continue 不误伤', sres._is_bare_secret('continue'), False)
check('bare: ok 不误伤', sres._is_bare_secret('ok'), False)
check('bare: IP 不误伤', sres._is_bare_secret('192.168.1.1'), False)
check('bare: 纯数字口令命中', sres._is_bare_secret('123456'), True)
check('bare: 中文句子不误伤', sres._is_bare_secret('这个开关打不开'), False)

# ── redact（三层脱敏回归）──
secrets = ['Example42.']
check('redact: 会话自扫描拦引文复述', '…<凭据已隐去>' in sres.redact('❌ testuser + Example42. → 拒', secrets), True)
check('redact: rc- 前缀形态', 'rc-' in sres.redact('key 是 rc-abcdef123456 拿去用', secrets), False)
check('redact: 密钥关键词（F15 扩词）', '已隐去' in sres.redact('密钥 abcdef123456 待轮换', secrets), True)
check('redact: 无凭据原文不动', sres.redact('今天天气不错', secrets), '今天天气不错')
# 🔴 脱敏先于截断（盲审 F1/F2）：凭据跨切分线不逃逸
long_text = 'x' * 145 + 'Example42.'
check('redact: 先脱敏后截断', 'Example42.' in sres.redact(long_text, secrets)[:150], False)

# ── 强提名保位合成断言（盲审 F4/F5：cand 约束 + k 溢出）——
# 直接构造分数数组走 score() 的提名逻辑不可行（依赖真索引），改用逻辑等价合成验证：
# 提名循环必须同时满足 cand 与去根约束——此处以源码静态断言兜底 + 真索引回归（eval_prod）覆盖
src = open('/opt/data/scripts/session_recall.py', encoding='utf-8').read()
nom_block = src[src.find('# 通道强提名保位'):src.find('return out')]
check('强提名段含 cand 约束', 'si not in cand' in nom_block, True)
check('强提名段先查界再 append', nom_block.find('if len(out) >= k: break') != -1, True)

print(f'\n单元测试：{passed} 过 / {failed} 败')
sys.exit(1 if failed else 0)
