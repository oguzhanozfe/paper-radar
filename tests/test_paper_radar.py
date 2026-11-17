import json
import re
import unittest
from pathlib import Path

from paper_radar import arxiv, digest, evaluate, rank

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"
PROFILE = rank.load_profile((ROOT / "profile.md").read_text(encoding="utf-8"))


def P(pid, title, abstract, code=()):
    return {"id": pid, "title": title, "abstract": abstract, "comment": "", "url": f"https://arxiv.org/abs/{pid}",
            "categories": ["cs.CL"], "code": list(code), "code_hint": False}


class TestArxiv(unittest.TestCase):
    def test_parse_recorded_feed(self):
        total, papers = arxiv.parse_feed((FIX / "arxiv_sample.xml").read_text(encoding="utf-8"))
        self.assertGreater(total, 0)
        self.assertTrue(papers)
        for p in papers:
            self.assertRegex(p["id"], r"^\d{4}\.\d{4,5}$")  # version suffix stripped
            self.assertTrue(p["title"] and p["abstract"] and p["authors"])
        by_id = {p["id"]: p for p in papers}
        # recorded entries: code link in the comment field, and an entry without any code link
        self.assertIn("https://github.com/Aayush-Marasini/WeightBench", by_id["2611.18553"]["code"])
        self.assertEqual(by_id["2611.18563"]["code"], [])

    def test_code_links(self):
        got = arxiv.code_links("Code at https://github.com/foo/bar.", "see github.com/Foo/Bar and www.github.com/x/y")
        self.assertEqual(got, ["https://github.com/foo/bar", "https://github.com/x/y"])
        self.assertEqual(arxiv.code_links("no links here"), [])

    def test_fetch_pages_dedupes_and_spaces_requests(self):
        feed = (FIX / "arxiv_sample.xml").read_text(encoding="utf-8")
        total, papers = arxiv.parse_feed(feed)
        calls, sleeps = [], []

        def get(params):
            calls.append(params)
            # every page returns the same entries: they must be deduplicated, and paging must stop
            return feed.replace(f"<opensearch:totalResults>{total}", f"<opensearch:totalResults>{2 * len(papers)}")

        out = arxiv.fetch(["cs.MA"], days=1, get=get, sleep=sleeps.append, log=lambda *_: None)
        self.assertEqual(len(out), len(papers))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(s >= 3 for s in sleeps) and len(sleeps) == 1)
        self.assertIn("submittedDate:[", calls[0]["search_query"])


class TestRank(unittest.TestCase):
    def test_profile(self):
        self.assertEqual(len(PROFILE), 5)
        self.assertIn("mcp", PROFILE[1]["keywords"])

    def test_keyword_scores(self):
        papers = [P("1", "A benchmark for tool use agents", "We evaluate agents calling tools via MCP."),
                  P("2", "Protein folding", "We fold proteins.")]
        s = {x["id"]: x for x in rank.keyword_scores(papers, PROFILE)}
        self.assertGreater(s["1"]["score"], 0)
        self.assertEqual(s["2"]["score"], 0)

    def test_embed_scores_with_fake(self):
        def fake_embed(texts):  # 2-d vectors: "game" axis vs everything else
            return [[1.0, 0.0] if "game" in t.lower() else [0.0, 1.0] for t in texts]
        papers = [P("1", "Level generation for a video game", "x"), P("2", "Other", "y")]
        s = {x["id"]: x for x in rank.embed_scores(papers, PROFILE, embed=fake_embed)}
        self.assertEqual(s["1"]["theme"], "AI for games and player data")
        self.assertAlmostEqual(s["1"]["score"], 1.0)

    def test_quote_verification(self):
        ab = "We introduce “ToolBench-X”, a benchmark of 3,000 tool calls.\nResults vary."
        self.assertTrue(rank.quote_verified('introduce "ToolBench-X", a benchmark', ab))
        self.assertTrue(rank.quote_verified("a  benchmark of 3,000 tool calls.", ab))
        self.assertFalse(rank.quote_verified("a benchmark of 5,000 tool calls", ab))
        self.assertFalse(rank.quote_verified("benchmark", ab))  # too short to count as grounding

    def test_llm_batch_drops_fabricated_quote_and_maps_theme(self):
        papers = [P("1", "T", "Agents are evaluated on long tasks."), P("2", "U", "We fold proteins quickly.")]
        reply = '<thought>let me think {"x": 1}</thought>```json\n' + json.dumps({"results": [
            {"id": "1", "score": 8, "theme": "Tool use", "reason": "r", "quote": "agents beat humans easily"},
            {"id": "2", "score": 1, "theme": "RAG with evidence", "reason": "no", "quote": "We fold proteins quickly"}]}) + "\n```"
        j1, j2 = rank.llm_judge_batch(papers, PROFILE, lambda _: (reply, "gemini:m"))
        self.assertEqual((j1["score"], j1["quote"], j1["quote_ok"], j1["model"]), (8, "", False, "gemini:m"))
        self.assertEqual(j1["theme"], "Tool use and MCP")
        self.assertTrue(j2["quote_ok"])

    def test_llm_batch_accepts_bare_list(self):
        reply = json.dumps([{"id": "1", "score": 6, "theme": "RAG with evidence", "reason": "r", "quote": ""}])
        (j,) = rank.llm_judge_batch([P("1", "T", "a b c")], PROFILE, lambda _: reply)
        self.assertEqual((j["score"], j["theme"]), (6, "RAG with evidence"))

    def test_llm_batch_bad_output(self):
        js = rank.llm_judge_batch([P("1", "T", "a b c"), P("2", "T", "d e f")], PROFILE, lambda _: "not json")
        self.assertEqual([j["score"] for j in js], [0, 0])
        self.assertIn("error", js[0])

    def test_llm_rank_prefilter_batches_and_tiebreak(self):
        papers = [P(str(i), f"t{i}", "a b c d") for i in range(12)]
        emb = [{"id": str(i), "score": i / 100, "theme": PROFILE[0]["theme"]} for i in range(12)]
        prompts = []

        def chat(prompt):
            prompts.append(prompt)
            ids = re.findall(r"Paper id: (\S+)", prompt)
            return json.dumps({"results": [{"id": i, "score": 7, "theme": "x", "reason": "r", "quote": ""} for i in ids]})
        r = rank.llm_rank(papers, PROFILE, emb, k=10, chat=chat)
        self.assertEqual(len(prompts), 2)  # 10 papers in batches of 8
        self.assertEqual([x["id"] for x in r[:3]], ["11", "10", "9"])  # ties -> embedding order
        self.assertEqual([x["score"] for x in r[9:]], [7, None, None])


class TestEvalAndDigest(unittest.TestCase):
    def test_metrics(self):
        labels = {str(i): int(i < 4) for i in range(40)}
        ids = ["0", "10", "1", "11", "12", "13", "14", "15", "16", "17"] + [str(i) for i in range(18, 40)] + ["2", "3"]
        self.assertAlmostEqual(evaluate.precision_at(ids, labels, 10), 0.2)
        self.assertAlmostEqual(evaluate.recall_at(ids, labels, 20), 0.5)
        self.assertEqual(evaluate.random_expectation(labels), (0.1, 0.5))

    def test_sample_is_deterministic(self):
        pool = [{"id": str(i)} for i in range(300)]
        self.assertEqual(evaluate.sample(pool), evaluate.sample(list(reversed(pool))))

    def test_committed_eval_recomputes(self):
        """The README table can be recomputed offline from committed labels and scores."""
        scores_path = ROOT / "data/eval/scores.json"
        if not scores_path.exists():
            self.skipTest("no recorded scores yet")
        from paper_radar.__main__ import rankings_from
        labels = evaluate.load_labels(ROOT / "data/eval/labels.csv")
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        self.assertEqual(len(labels), 100)
        self.assertIn("P@10", evaluate.table(rankings_from(scores), labels))

    def test_digest_escapes_and_groups(self):
        papers = {"1": P("1", "<script>alert(1)</script> agents", "abs", code=["https://github.com/a/b"]),
                  "2": P("2", "Close one", "abs"), "3": P("3", "Far", "abs")}
        ranked = [{"id": "1", "score": 9, "theme": "Tool use and MCP", "reason": "r & <b>", "quote": "q"},
                  {"id": "2", "score": 6, "theme": "RAG with evidence", "reason": "close", "quote": ""},
                  {"id": "3", "score": 2, "theme": "RAG with evidence", "reason": "far", "quote": ""},
                  {"id": "4", "score": None, "theme": "RAG with evidence"}]
        top, close = digest.select(ranked, papers)
        self.assertEqual(([p["id"] for p in top], [p["id"] for p in close]), (["1"], ["2"]))
        themes = [t["theme"] for t in PROFILE]
        h = digest.html("T", top, close, themes, "stats")
        self.assertNotIn("<script>", h)
        self.assertIn("&lt;script&gt;", h)
        self.assertIn('href="https://github.com/a/b"', h)
        m = digest.markdown("T", top, close, themes, "stats")
        self.assertIn("## Tool use and MCP", m)
        self.assertIn("## Skipped but close", m)
        self.assertNotIn("## RAG with evidence", m)  # only themes with top papers get a section


class TestModels(unittest.TestCase):
    def test_fallback_and_cache(self):
        import tempfile
        from paper_radar import models
        calls = []

        def fake_post(provider, path, body):
            calls.append(body["model"])
            if body["model"] == "a":
                raise models.Unavailable("HTTP 429")
            return {"choices": [{"message": {"content": '{"results": []}'}}], "usage": {"prompt_tokens": 5}}
        saved = (models._post, models.LLM_MODELS, models.CACHE)
        try:
            models._post, models.LLM_MODELS = fake_post, ["a", "b"]
            models.CACHE = tempfile.mkdtemp()
            self.assertEqual(models.chat("p"), ('{"results": []}', f"{models.LLM}:b"))
            self.assertEqual(models.chat("p")[1], f"{models.LLM}:b")  # second call served from cache
            self.assertEqual(calls, ["a", "b"])
        finally:
            models._post, models.LLM_MODELS, models.CACHE = saved


class TestDemo(unittest.TestCase):
    def test_replay_reproduces_committed_digest(self):
        import contextlib
        import io
        import os
        import tempfile
        from paper_radar import demo
        out, cwd = tempfile.mkdtemp(), os.getcwd()
        try:
            os.chdir(ROOT)
            with contextlib.redirect_stdout(io.StringIO()) as said, contextlib.redirect_stderr(io.StringIO()):
                demo.run("replay", out)
        finally:
            os.chdir(cwd)
        self.assertIn("mode: replay (recorded", said.getvalue())
        for ext in (".md", ".html"):
            got = Path(out, demo.NAME + ext).read_text(encoding="utf-8")
            self.assertEqual(got, (ROOT / "digests" / (demo.NAME + ext)).read_text(encoding="utf-8"))

    def test_resolution_order(self):
        import os
        from unittest import mock
        from paper_radar import demo
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "x"}):
            self.assertEqual(demo.resolve("auto"), "gemini")
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(demo, "ollama_ready", return_value=True):
                self.assertEqual(demo.resolve("auto"), "ollama")
            with mock.patch.object(demo, "ollama_ready", return_value=False):
                self.assertEqual(demo.resolve("auto"), "replay")
        self.assertEqual(demo.resolve("replay"), "replay")


if __name__ == "__main__":
    unittest.main()
