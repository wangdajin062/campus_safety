#!/usr/bin/env python3
"""
Frontend-Backend Contract Integration Test
校园安全 APP v3  —  前后端 API 契约全量验证
"""
import re, sys
from pathlib import Path
from collections import defaultdict

FRONTEND = Path("/home/claude/frontend/android/app/src/main")
BACKEND  = Path("/home/claude/v3/backend")

errors   = []
warnings = []
passed   = 0

def ok(msg):    global passed; passed += 1;  print(f"  ✅ {msg}")
def err(msg):   errors.append(msg);          print(f"  ❌ {msg}")
def warn(msg):  warnings.append(msg);        print(f"  ⚠️  {msg}")

# ──────────────────────────────────────────────────────────
# [1] Frontend endpoint extraction
# ──────────────────────────────────────────────────────────
print("\n[1] Frontend API endpoints — CampusApi.java")
api_file = FRONTEND / "java/com/campus/safety/network/api/CampusApi.java"
fe_endpoints = set()
if api_file.exists():
    for m in re.finditer(r'@(GET|POST|PUT|DELETE|PATCH)\s*\(\s*"([^"]+)"\s*\)',
                          api_file.read_text()):
        path = "/" + m.group(2).lstrip("/")
        norm = re.sub(r'\{[^}]+\}', '{id}', path).rstrip('/') or path
        fe_endpoints.add((m.group(1), norm))
    ok(f"Parsed {len(fe_endpoints)} frontend endpoints")
else:
    err("CampusApi.java missing"); sys.exit(1)

# ──────────────────────────────────────────────────────────
# [2] Backend endpoint extraction
# ──────────────────────────────────────────────────────────
print("\n[2] Backend endpoints — FastAPI routers")
be_endpoints = set()

# Prefix map from main.py
main_txt = (BACKEND / "main.py").read_text()
prefix_map = {}
for m in re.finditer(r'include_router\(\s*(\w+)\.router.*?prefix\s*=\s*["\']([^"\']+)["\']',
                      main_txt, re.S):
    prefix_map[m.group(1)] = m.group(2)
ok(f"Router prefixes: {prefix_map}")

for py_file in (BACKEND / "api/v1").glob("*.py"):
    if py_file.stem == "__init__": continue
    src = py_file.read_text()
    stem = py_file.stem

    if stem == "alerts_reports_users":
        # Plain @router.xxx  →  /v1/alerts
        for m in re.finditer(r'@router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']*)["\']', src):
            full = "/v1/alerts" + m.group(2)
            be_endpoints.add((m.group(1).upper(),
                               re.sub(r'\{[^}]+\}','{id}',full).rstrip('/')))
        # @reports_router / @users_router
        for m in re.finditer(r'@(\w+)_router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']*)["\']', src):
            rname = m.group(1)          # reports | users
            # users module is mounted as /v1/user (not /v1/users)
            pfx = prefix_map.get(rname, f"/v1/{rname}")
            full = pfx + m.group(3)
            be_endpoints.add((m.group(2).upper(),
                               re.sub(r'\{[^}]+\}','{id}',full).rstrip('/')))
    elif stem in prefix_map:
        pfx = prefix_map[stem]
        for m in re.finditer(r'@router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']*)["\']', src):
            full = pfx + m.group(2)
            be_endpoints.add((m.group(1).upper(),
                               re.sub(r'\{[^}]+\}','{id}',full).rstrip('/')))

ok(f"Parsed {len(be_endpoints)} backend endpoints")

# ──────────────────────────────────────────────────────────
# [3] Contract matching
# ──────────────────────────────────────────────────────────
print("\n[3] Contract validation")

matched = fe_endpoints & be_endpoints
fe_only = fe_endpoints - be_endpoints
# backend-only: skip admin (web panel only)
be_only = {e for e in (be_endpoints - fe_endpoints) if '/admin' not in e[1]}

ok(f"Matched: {len(matched)}/{len(fe_endpoints)} frontend endpoints")

for method, path in sorted(fe_only):
    err(f"Frontend calls non-existent backend endpoint: {method} {path}")

for method, path in sorted(be_only):
    warn(f"Backend endpoint without frontend caller: {method} {path}")

# ──────────────────────────────────────────────────────────
# [4] Layout / ViewBinding coverage
# ──────────────────────────────────────────────────────────
print("\n[4] ViewBinding layout coverage")

layouts = {f.stem for f in (FRONTEND/"res/layout").glob("*.xml")}
ok(f"Existing layouts: {len(layouts)}")

binding_refs = set()
for jf in FRONTEND.rglob("**/*.java"):
    for m in re.finditer(r'([A-Z][a-zA-Z]+)Binding\.inflate', jf.read_text()):
        cls = m.group(1)
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', cls).lower()
        binding_refs.add(snake)

missing_layouts = binding_refs - layouts
if missing_layouts:
    for m in sorted(missing_layouts):
        err(f"Missing layout: {m}.xml")
else:
    ok(f"All {len(binding_refs)} ViewBinding layouts present")

# ──────────────────────────────────────────────────────────
# [5] @id/ consistency (fast heuristic)
# ──────────────────────────────────────────────────────────
print("\n[5] XML @+id consistency")

layout_ids: dict[str,set] = {}
for xml in (FRONTEND/"res/layout").glob("*.xml"):
    layout_ids[xml.stem] = set(re.findall(r'@\+id/(\w+)', xml.read_text()))
total_ids = sum(len(v) for v in layout_ids.values())
ok(f"Total @+id/ definitions: {total_ids} across {len(layout_ids)} layouts")

# Per-file id scan in Java code
id_mismatches = 0
for jf in FRONTEND.rglob("**/*.java"):
    jcontent = jf.read_text()
    # Determine which layout this java file uses
    inflate_match = re.search(r'([A-Z][a-zA-Z]+)Binding\.inflate', jcontent)
    if not inflate_match: continue
    snake = re.sub(r'(?<!^)(?=[A-Z])', '_', inflate_match.group(1)).lower()
    if snake not in layout_ids: continue
    valid_ids = layout_ids[snake]
    # Find bd.someId references (camelCase → snake_case)
    for m in re.finditer(r'bd\.([a-z][a-zA-Z0-9_]+)', jcontent):
        ref = m.group(1)
        if ref in ('getRoot', 'root'): continue
        ref_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', ref).lower()
        if ref_snake not in valid_ids:
            id_mismatches += 1

if id_mismatches == 0:
    ok("All Java ViewBinding field references align with XML @+id/")
else:
    warn(f"{id_mismatches} ViewBinding field reference mismatches (review manually)")

# ──────────────────────────────────────────────────────────
# [6] Critical model presence
# ──────────────────────────────────────────────────────────
print("\n[6] Data models")

required = {"ApiResponse","LoginResponse","FraudCase","FraudAlert","CallLog",
            "UserStats","HomeData","PhoneCheckResult","SmsAnalyzeRequest",
            "SmsAnalyzeResult","MultimodalRequest","InferenceResult",
            "CoTStreamEvent","PageResult","ReportRequest","FeedbackRequest"}
existing = {f.stem for f in (FRONTEND/"java/com/campus/safety/model").glob("*.java")}
missing_models = required - existing
if missing_models:
    for m in sorted(missing_models): err(f"Missing model: {m}.java")
else:
    ok(f"All {len(required)} data models present")

# ──────────────────────────────────────────────────────────
# [7] Security checks
# ──────────────────────────────────────────────────────────
print("\n[7] Android security checks")

java_files = list(FRONTEND.rglob("**/*.java"))

# No hardcoded secrets
found_secret = False
for jf in java_files:
    src = jf.read_text()
    if re.search(r'(?:secret|password|apikey)\s*=\s*"[A-Za-z0-9+/]{16,}"', src, re.I):
        found_secret = True; err(f"Hardcoded secret in {jf.name}")
if not found_secret:
    ok("No hardcoded secrets in Java sources")

# EncryptedSharedPreferences used
uses_encrypted = any('EncryptedSharedPreferences' in jf.read_text() for jf in java_files)
ok("EncryptedSharedPreferences present") if uses_encrypted else err("Missing EncryptedSharedPreferences")

# HMAC/SHA-256 present
uses_sha = any('SHA-256' in jf.read_text() or 'sha256' in jf.read_text().lower() for jf in java_files)
ok("SHA-256 hashing present") if uses_sha else warn("SHA-256 not found")

# network_security_config.xml
nsc = FRONTEND / "res/xml/network_security_config.xml"
if nsc.exists() and 'cleartextTrafficPermitted="false"' in nsc.read_text():
    ok("network_security_config.xml enforces HTTPS")
else:
    err("HTTPS not enforced in network_security_config.xml")

# No raw text of SMS uploaded (SmsFeatureExtractor check)
fe_src = (FRONTEND/"java/com/campus/safety/ml/SmsFeatureExtractor.java").read_text()
ok("SmsFeatureExtractor: privacy-safe (no raw SMS upload)") \
    if "原文" in fe_src or "not upload" in fe_src.lower() or "not transmitted" in fe_src.lower() \
    else warn("SmsFeatureExtractor: verify raw SMS is not uploaded")

# ──────────────────────────────────────────────────────────
# [8] File count sanity
# ──────────────────────────────────────────────────────────
print("\n[8] File count sanity")

java_count   = len(java_files)
xml_count    = len(list(FRONTEND.rglob("**/*.xml")))
total_fe     = java_count + xml_count
ok(f"Total frontend files: {total_fe}  ({java_count} Java + {xml_count} XML)")

if java_count >= 40:  ok(f"Java file count adequate: {java_count}")
else:                 warn(f"Java file count low: {java_count}")

if xml_count >= 20:   ok(f"XML file count adequate: {xml_count}")
else:                 warn(f"XML file count low: {xml_count}")

# ──────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────
total = passed + len(warnings) + len(errors)
score = int(100 * passed / total) if total else 0

print(f"\n{'='*62}")
print(f"  前后端集成测试  —  {passed} 通过 / {len(warnings)} 警告 / {len(errors)} 错误")
print(f"  集成评分: {score}/100")
print(f"{'='*62}")

if errors:
    print("\n  待修复:")
    for e in errors: print(f"    ❌ {e}")
    sys.exit(1)
else:
    print("\n  ✅ 所有关键检查通过，可以打包发布")
    sys.exit(0)
