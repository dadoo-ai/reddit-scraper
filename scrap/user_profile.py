from __future__ import annotations
import asyncio, csv, random, re
from pathlib import Path
from typing import List, Optional, Union, Dict
from urllib.parse import urlparse, unquote
from httpx import AsyncClient, Response, TimeoutException, HTTPError, RequestError
from parsel import Selector

DEFAULT_HEADERS = {
    "Accept-Language":"en-US,en;q=0.9",
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding":"gzip, deflate, br",
    "Cache-Control":"no-cache","Pragma":"no-cache","Cookie":"intl_splash=false",
}

def normalize_profile_url(url:str)->str:
    if not url: return url
    url = url.strip()
    url = re.sub(r"^https?://www\.reddit\.com","https://old.reddit.com",url)
    url = re.sub(r"[?#].*$","",url)
    return url if url.endswith("/") else url + "/"

def join_url(base:str,suffix:str)->str:
    base = normalize_profile_url(base)
    if suffix.startswith("/"): suffix = suffix[1:]
    return base + suffix

def safe_author_from_profile(profile_url:str)->str:
    try:
        m = re.search(r"/user/([^/]+)/?", urlparse(profile_url).path)
        author = m.group(1) if m else "unknown"
    except Exception:
        author = "unknown"
    author = unquote(author)
    return re.sub(r"[^\w\-.@+]+","_",author)

class CSVProfileReader:
    def __init__(self, csv_path:Union[str,Path], delimiter:str=",", column:str="author_profile")->None:
        self.csv_path = Path(csv_path); self.delimiter = delimiter; self.column = column
    def read_profiles(self)->List[str]:
        profiles: List[str] = []
        with self.csv_path.open("r",encoding="utf-8",newline="") as f:
            r = csv.DictReader(f, delimiter=self.delimiter)
            if not r.fieldnames or self.column not in r.fieldnames:
                raise ValueError(f"Colonne '{self.column}' introuvable. En-têtes: {r.fieldnames}")
            for row in r:
                u = (row.get(self.column) or "").strip()
                if u: profiles.append(normalize_profile_url(u))
        seen=set(); uniq=[]
        for u in profiles:
            if u not in seen: seen.add(u); uniq.append(u)
        return uniq

class RedditHTTP:
    def __init__(self, timeout:float=30.0, max_retries:int=8, base_backoff:float=1.6,
                 request_gap_s:float=1.4, jitter_s:float=0.7)->None:
        self.client = AsyncClient(http2=True, headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout)
        self.max_retries = max_retries; self.base_backoff = base_backoff
        self.request_gap_s = request_gap_s; self.jitter_s = jitter_s
        self._gap_lock = asyncio.Lock()
    async def fetch(self, url:str)->Response:
        last_exc: Optional[Exception] = None
        total_429 = 0
        for attempt in range(1, self.max_retries+1):
            async with self._gap_lock:
                await asyncio.sleep(self.request_gap_s + random.random()*self.jitter_s)
            try:
                resp = await self.client.get(url)
                if resp.status_code == 429:
                    total_429 += 1
                    ra = resp.headers.get("Retry-After")
                    try: wait_s = max(1.0, float(ra)) if ra else self.base_backoff*attempt
                    except ValueError: wait_s = self.base_backoff*attempt
                    await asyncio.sleep(wait_s + random.random()*self.jitter_s)
                    if total_429 >= self.max_retries:
                        raise HTTPError(f"429 Too Many Requests after {total_429} retries for {url}")
                    continue
                resp.raise_for_status()
                return resp
            except (TimeoutException, HTTPError, RequestError) as e:
                last_exc = e
                wait_s = self.base_backoff*attempt
                await asyncio.sleep(wait_s + random.random()*self.jitter_s)
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Max retries exceeded without success for {url}")
    async def aclose(self): await self.client.aclose()

class RedditProfileParser:
    @staticmethod
    def find_next_url(html:str)->Optional[str]:
        return Selector(html).xpath("//span[@class='next-button']/a/@href").get()
    @staticmethod
    def parse_submitted_page(html:str)->List[Dict]:
        sel = Selector(html)
        items = sel.xpath("//div[contains(@class,'thing') and contains(@class,'link')]")
        out: List[Dict] = []
        for it in items:
            title = it.xpath(".//a[contains(@class,'title')]/text()").get() or ""
            permalink = it.xpath(".//a[contains(@class,'comments')]/@href").get()
            if permalink: out.append({"title": title.strip(), "permalink": permalink})
        return out

class RedditPostParser:
    @staticmethod
    def parse_post_text(html:str)->str:
        sel = Selector(html)
        nodes = sel.xpath("(//div[contains(@class,'thing') and contains(@class,'link')])[1]"
                          "//div[contains(@class,'usertext-body')]/div//text()").getall()
        if not nodes:
            nodes = sel.xpath("//div[contains(@class,'usertext-body')]/div//text()").getall()
        text = " ".join(t.strip() for t in nodes if t and t.strip())
        return re.sub(r"\s+"," ",text).strip()

class RedditUserCollector:
    def __init__(self, http:RedditHTTP, threshold:int=100, max_pages:Optional[int]=None,
                 page_gap_s:float=1.2, page_jitter_s:float=0.6, post_fetch_concurrency:int=1)->None:
        self.http=http; self.threshold=threshold; self.max_pages=max_pages
        self.page_gap_s=page_gap_s; self.page_jitter_s=page_jitter_s
        self.post_sem = asyncio.Semaphore(max(1, post_fetch_concurrency))
    async def collect_until_valid(self, profile_url:str)->Dict:
        url = join_url(profile_url,"submitted/"); total=0; pages=0; collected:List[Dict]=[]
        while url and (self.max_pages is None or pages < self.max_pages):
            resp = await self.http.fetch(url)
            posts = RedditProfileParser.parse_submitted_page(resp.text)
            total += len(posts); collected.extend(posts)
            if total > self.threshold: break
            url = RedditProfileParser.find_next_url(resp.text); pages += 1
            if url: await asyncio.sleep(self.page_gap_s + random.random()*self.page_jitter_s)
        async def _fetch_text(item:Dict)->None:
            async with self.post_sem:
                pr = await self.http.fetch(item["permalink"])
                item["text"] = RedditPostParser.parse_post_text(pr.text)
        await asyncio.gather(*[asyncio.create_task(_fetch_text(it)) for it in collected])
        return {"is_valid": total > self.threshold, "total_seen": total, "items": collected}

class ProfileBatchRunner:
    def __init__(self, csv_path:Union[str,Path], output_dir:Union[str,Path],
                 delimiter:str=",", column:str="author_profile", concurrency:int=3,
                 threshold:int=100, max_pages:Optional[int]=None, pause_between_batches:int=10,
                 request_gap_s:float=1.6, page_gap_s:float=1.2, post_fetch_concurrency:int=1,
                 stop_on_first_valid:bool=True)->None:
        self.reader = CSVProfileReader(csv_path, delimiter=delimiter, column=column)
        self.http = RedditHTTP(request_gap_s=request_gap_s)
        self.collector = RedditUserCollector(self.http, threshold=threshold, max_pages=max_pages,
                                             page_gap_s=page_gap_s, post_fetch_concurrency=post_fetch_concurrency)
        self.concurrency=max(1,concurrency); self.threshold=threshold
        self.pause_between_batches=max(0,pause_between_batches)
        self.stop_on_first_valid=stop_on_first_valid
        self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self._found_valid = asyncio.Event()
    def _write_user_csv(self, profile_url:str, items:List[Dict])->Path:
        author = safe_author_from_profile(profile_url)
        out = self.output_dir / f"{author}.csv"
        with out.open("w",encoding="utf-8",newline="") as f:
            w = csv.writer(f); w.writerow(["url_user","url_post","titre_post","texte_post"])
            for it in items:
                w.writerow([profile_url, it.get("permalink",""), it.get("title",""), it.get("text","")])
        return out
    async def _process_one(self, profile_url:str)->None:
        if self.stop_on_first_valid and self._found_valid.is_set(): return
        author = safe_author_from_profile(profile_url)
        try:
            res = await self.collector.collect_until_valid(profile_url)
            total = res["total_seen"]
            if res["is_valid"]:
                path = self._write_user_csv(profile_url, res["items"])
                print(f"[VALABLE] {author} (n_posts={total}) -> {path}")
                if self.stop_on_first_valid: self._found_valid.set()
            else:
                print(f"[NON VALABLE] {author} (n_posts={total})")
        except Exception as e:
            print(f"[ERREUR] {author} -> {type(e).__name__}: {e}")
    async def run(self)->None:
        profiles = self.reader.read_profiles()
        if not profiles:
            print("Aucun profil à analyser."); await self.http.aclose(); return
        for i in range(0, len(profiles), self.concurrency):
            if self.stop_on_first_valid and self._found_valid.is_set(): break
            batch = profiles[i:i+self.concurrency]
            await asyncio.gather(*[asyncio.create_task(self._process_one(p)) for p in batch])
            if (i+self.concurrency)<len(profiles) and not self._found_valid.is_set() and self.pause_between_batches>0:
                print(f"Pause {self.pause_between_batches}s avant le batch suivant...")
                await asyncio.sleep(self.pause_between_batches)
        await self.http.aclose()

