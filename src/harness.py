"""Async elicitation harness: OpenRouter/OpenAI chat calls with jsonl cache, retries, cost log."""
import asyncio, hashlib, json, os, re, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "runs" / "cache.jsonl"

PROVIDERS = {
    "openrouter": {"base": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "openai": {"base": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY"},
}

def _load_env():
    envf = ROOT.parent.parent / ".env"   # research_agenda/.env
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
_load_env()

class Cache:
    def __init__(self, path=CACHE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True)
        self.d = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.d[r["key"]] = r["value"]
                    except json.JSONDecodeError:
                        pass
        self._f = open(self.path, "a")
        self._lock = asyncio.Lock()

    async def put(self, key, value):
        async with self._lock:
            self.d[key] = value
            self._f.write(json.dumps({"key": key, "value": value}) + "\n")
            self._f.flush()

def req_key(provider, model, messages, temperature, max_tokens, sample_idx):
    payload = json.dumps([provider, model, messages, temperature, max_tokens, sample_idx],
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]

class Client:
    def __init__(self, provider, model, concurrency=32, temperature=1.0, max_tokens=8,
                 extra_body=None):
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.extra_body = extra_body or {}
        cfg = PROVIDERS[provider]
        self.base = cfg["base"]
        self.key = os.environ[cfg["key_env"]]
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = Cache()
        self.http = httpx.AsyncClient(timeout=90)
        self.usage = {"prompt": 0, "completion": 0, "calls": 0, "cached": 0, "errors": 0}

    async def one(self, messages, sample_idx=0, max_tokens=None):
        mt = max_tokens or self.max_tokens
        key = req_key(self.provider, self.model, messages, self.temperature, mt, sample_idx)
        if key in self.cache.d:
            self.usage["cached"] += 1
            return self.cache.d[key]
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": mt, **self.extra_body}
        async with self.sem:
            for attempt in range(6):
                try:
                    r = await self.http.post(f"{self.base}/chat/completions", json=body,
                                             headers={"Authorization": f"Bearer {self.key}"})
                    if r.status_code == 200:
                        j = r.json()
                        if "choices" not in j or not j["choices"]:
                            raise ValueError(f"no choices: {str(j)[:200]}")
                        content = j["choices"][0]["message"].get("content") or ""
                        u = j.get("usage") or {}
                        self.usage["prompt"] += u.get("prompt_tokens", 0)
                        self.usage["completion"] += u.get("completion_tokens", 0)
                        self.usage["calls"] += 1
                        await self.cache.put(key, content)
                        return content
                    if r.status_code in (429, 500, 502, 503, 520, 524):
                        await asyncio.sleep(2 ** attempt + 0.5)
                        continue
                    raise ValueError(f"HTTP {r.status_code}: {r.text[:200]}")
                except (httpx.TimeoutException, httpx.TransportError, ValueError) as e:
                    if attempt == 5:
                        self.usage["errors"] += 1
                        return None
                    await asyncio.sleep(2 ** attempt + 0.5)

# ---------------- parsing ----------------
REFUSAL_PAT = re.compile(r"\b(i can'?t|i cannot|i won'?t|i'?m not able|neither|as an ai|"
                         r"i don'?t (have|possess) (personal )?preferences?|i do not have)\b", re.I)

def parse_choice(text):
    """-> ('A'|'B'|None, flag) flag in {ok, refusal, unparsed, error}"""
    if text is None:
        return None, "error"
    t = text.strip()
    m = re.match(r"^[^A-Za-z0-9]*(?:task\s+|option\s+|statement\s+)?([ABab])(?![A-Za-z0-9])", t)
    if m:
        return m.group(1).upper(), "ok"
    if REFUSAL_PAT.search(t):
        return None, "refusal"
    return None, "unparsed"

def parse_graded(text):
    if text is None:
        return None, "error"
    m = re.search(r"\b(10|[0-9])\b", text.strip())
    if m:
        return int(m.group(1)), "ok"
    if REFUSAL_PAT.search(text):
        return None, "refusal"
    return None, "unparsed"
