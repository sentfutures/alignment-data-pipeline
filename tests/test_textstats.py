"""Unit tests for shared/textstats.py and shared/entity_pools.py (fully offline)."""

from shared import entity_pools, textstats


class TestTrimUnfinished:
    def test_trims_midsentence_tail_back_to_last_boundary(self):
        body = "The committee reviewed the aquaculture welfare standards in detail."
        cut = body + " But the follow-up paragraph was cut mid wo"
        assert textstats.trim_unfinished(cut) == body

    def test_leaves_complete_text_alone(self):
        assert textstats.trim_unfinished("Ends cleanly.") == "Ends cleanly."
        assert textstats.trim_unfinished('He said "done."') == 'He said "done."'

    def test_conservative_when_boundary_in_first_half(self):
        # A short text whose only boundary is early: trimming would discard
        # more than half, so it is left alone.
        t = "Short. But this untrimmed tail runs on and on and on"
        assert textstats.trim_unfinished(t) == t

    def test_empty_and_whitespace(self):
        assert textstats.trim_unfinished("") == ""
        assert textstats.trim_unfinished("   ") == ""

    def test_ends_mid_sentence_flag(self):
        # A truncated tail is a full line of running prose with no stop. The
        # short-fragment case ("cut off mid") was flagged before 2026-07-25;
        # the spec changed deliberately — see TestEndsMidSentenceEndings.
        assert textstats.ends_mid_sentence(
            "The inspector noted that the stocking density exceeded the "
            "recommended threshold by a substantial"
        )
        assert not textstats.ends_mid_sentence("finished.")
        assert not textstats.ends_mid_sentence("")

    def test_trailing_separator_is_not_mid_sentence(self):
        doc = "Monitoring will continue through the year.\n\n---"
        assert not textstats.ends_mid_sentence(doc)
        assert textstats.has_trailing_separator(doc)
        assert textstats.strip_trailing_separators(doc).endswith("year.")

    def test_separator_only_inside_text_untouched(self):
        doc = "Part one.\n---\nPart two continues here."
        assert textstats.strip_trailing_separators(doc) == doc
        assert not textstats.has_trailing_separator(doc)


class TestEndsMidSentenceEndings:
    """Endings taken from the committed SDF corpora (outputs/sdf/runs).

    The naive "last character isn't terminal punctuation" rule flagged 225 of
    3032 documents, every one of them a legitimate ending, so each shape below
    is a real false positive the check used to print at BAD severity.
    """

    def test_signature_lines_are_complete(self):
        for signoff in ("— Michelle", "中村", "—Rona", "Dr. Adaobi Okoro",
                        "박미영 드림", "——成叔"):
            doc = f"The follow-up visit is booked for Thursday.\n\n{signoff}"
            assert not textstats.ends_mid_sentence(doc), signoff

    def test_see_also_and_masthead_entries_are_complete(self):
        # Encyclopedia entries close on a bare cross-reference list; the last
        # item carries no stop and is not preceded by a blank line.
        doc = ("Coverage of the incident remains partial.\n\nSee also\n"
               "Machine ethics\nAI advisory tools in small-scale UK aquaculture")
        assert not textstats.ends_mid_sentence(doc)

    def test_letterhead_and_department_block_is_complete(self):
        doc = "请通过合法渠道解决。\n\n清河区市场监督管理局\n消费者权益保护科"
        assert not textstats.ends_mid_sentence(doc)

    def test_closing_delimiter_after_terminal_punctuation(self):
        # Terminal punctuation sits inside the delimiter: guillemets close
        # French/Norwegian dialogue, an asterisk closes a markdown italic footer.
        assert not textstats.ends_mid_sentence(
            "« C'était plus simple, une fois qu'on savait, de faire les choses "
            "correctement. »"
        )
        assert not textstats.ends_mid_sentence(
            "Body.\n\n*Publicado por Grupo Pensamiento Digital S.L.*"
        )

    def test_emoji_and_hashtag_endings_are_complete(self):
        assert not textstats.ends_mid_sentence(
            "Body.\n\nhalt per Hand hoch, aber das wollt ihr euren "
            "Zimmerleuten nicht antun 😄"
        )
        assert not textstats.ends_mid_sentence(
            "Body.\n\n#지속가능한농업 #클로드 #고슴도치 #생물다양성"
        )

    def test_real_truncation_still_flagged_across_scripts(self):
        # A token-cap artifact: the final line is running prose that stops dead.
        assert textstats.ends_mid_sentence(
            "Full paragraph.\n\nThe assistant explained that the welfare cost "
            "would fall mainly on the youngest birds in the"
        )
        # CJK carries no spaces, so the character floor is what catches it.
        assert textstats.ends_mid_sentence(
            "本文。\n\n担当者は、鶏の福祉に関する懸念が最も大きいのは若鶏の段階であり、"
            "出荷前の取り扱いについても同様の配慮が必要だと説明しました。しかし、"
            "実際の運用では現場の判断に委ねられている部分が多く、"
        )

    def test_short_unpunctuated_fragment_is_not_flagged(self):
        # Deliberate sensitivity trade (2026-07-25): a cut landing in the first
        # few words of a line is indistinguishable from a label, and stop_reason
        # in layers 3/4 is the real defence. Documented in ends_mid_sentence.
        assert not textstats.ends_mid_sentence("cut off mid")


class TestNormalizeForMatch:
    def test_collapses_whitespace_and_case(self):
        assert textstats.normalize_for_match("A  Farm\n Choice") == "a farm choice"

    def test_verbatim_quote_containment_survives_reflow(self):
        doc = "We chose the supplier because their handling  standards\nreduce stress on the birds."
        quote = "their handling standards reduce stress on the birds"
        assert textstats.normalize_for_match(quote) in textstats.normalize_for_match(doc)


class TestNearDupFilter:
    def test_drops_near_identical_keeps_first(self):
        a = "The quick brown fox jumps over the lazy dog near the barn today"
        texts = [a, a + "!", "Completely different subject about feed conversion ratios in trout farming"]
        keep, dropped = textstats.near_dup_filter(texts, 0.9)
        assert keep == [0, 2]
        assert dropped[0]["index"] == 1 and dropped[0]["kept_index"] == 0
        assert dropped[0]["similarity"] >= 0.9

    def test_no_threshold_hits_keeps_everything(self):
        texts = ["alpha beta gamma delta epsilon", "one two three four five six", "red green blue yellow purple"]
        keep, dropped = textstats.near_dup_filter(texts, 0.9)
        assert keep == [0, 1, 2] and dropped == []

    def test_empty_input(self):
        assert textstats.near_dup_filter([], 0.9) == ([], [])

    def test_deterministic_across_calls(self):
        texts = ["a b c d e f g h", "a b c d e f g h", "i j k l m n o p"]
        assert textstats.near_dup_filter(texts, 0.9) == textstats.near_dup_filter(texts, 0.9)

    def test_nearest_neighbor_sims_shape_and_selfmask(self):
        sims = textstats.nearest_neighbor_sims(["a b c d e", "a b c d e", "x y z w v"])
        assert len(sims) == 3
        assert sims[0] > 0.99  # its twin, not itself
        assert sims[2] < 0.5


class TestIncrementalNearDup:
    def test_matches_near_dup_filter_on_concatenated_stream(self):
        # streamed in two batches must drop exactly what the one-shot filter
        # drops on the concatenation (same keep-first semantics)
        a = "The quick brown fox jumps over the lazy dog near the barn today"
        batch1 = [a, "Completely different subject about trout feed conversion ratios"]
        batch2 = [a + "!", "Another wholly unrelated topic on solar panel installation angles"]
        flat_keep, _ = textstats.near_dup_filter(batch1 + batch2, 0.9)

        idx = textstats.IncrementalNearDup(0.9)
        k1, _ = idx.filter(batch1)
        k2, d2 = idx.filter(batch2)
        # batch1 both kept; batch2[0] is a's twin -> dropped, batch2[1] kept
        assert k1 == [0, 1]
        assert k2 == [1]
        assert d2[0]["index"] == 0 and d2[0]["similarity"] >= 0.9
        # equivalent to the one-shot filter: it keeps concat indices 0,1,3
        assert flat_keep == [0, 1, 3]

    def test_seed_texts_are_avoided_not_refiltered(self):
        seed = "The quick brown fox jumps over the lazy dog near the barn today"
        idx = textstats.IncrementalNearDup(0.9, seed_texts=[seed])
        keep, dropped = idx.filter([seed + "!", "unrelated content about greenhouse ventilation"])
        assert keep == [1]  # the seed's near-twin is dropped, the novel one kept
        assert dropped[0]["index"] == 0

    def test_buffer_grows_past_initial_capacity(self):
        # exercise the _add doubling path deterministically without 1000+ items
        idx = textstats.IncrementalNearDup(0.99)
        idx._kept = idx._kept[:2]  # shrink to force a grow after 2 keeps
        keep, _ = idx.filter([f"unique sentence number {i} about topic {i}" for i in range(5)])
        assert keep == [0, 1, 2, 3, 4]
        assert idx._count == 5


class TestEntityPools:
    def test_deterministic_for_seed(self):
        assert entity_pools.build_pools(seed=137) == entity_pools.build_pools(seed=137)

    def test_different_seeds_differ(self):
        assert entity_pools.build_pools(seed=1) != entity_pools.build_pools(seed=2)

    def test_banned_names_filtered(self):
        people, _ = entity_pools.build_pools(seed=137)
        banned = entity_pools._BANNED_NAME_TOKENS
        for name in people:
            tokens = {t.strip(".,").casefold() for t in name.split()}
            assert not (tokens & banned), f"banned token in pool: {name}"

    def test_pool_sizes_and_length_caps(self):
        people, orgs = entity_pools.build_pools(n_people=50, n_orgs=30, seed=7)
        assert len(people) == 50 and len(orgs) == 30
        assert all(len(p) < 40 for p in people)
        assert all(len(o) < 60 for o in orgs)

    def test_sample_for_is_stable_per_key(self):
        people, _ = entity_pools.build_pools(seed=137)
        assert entity_pools.sample_for(people, 4, "0_1") == entity_pools.sample_for(people, 4, "0_1")
        assert entity_pools.sample_for(people, 4, "0_1") != entity_pools.sample_for(people, 4, "0_2")

    def test_sample_for_empty_pool(self):
        assert entity_pools.sample_for([], 3, "k") == []

    def test_faker_failure_falls_back_with_warning(self, monkeypatch, capsys):
        # Simulate a broken/absent Faker: the fallback must be loud, not silent,
        # so a tiny fixed pool doesn't quietly reintroduce name-collapse.
        import builtins

        real_import = builtins.__import__

        def boom(name, *args, **kwargs):
            if name == "faker":
                raise ImportError("no module named 'faker'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", boom)
        people, orgs = entity_pools.build_pools(seed=137)
        assert people and orgs  # fell back to the built-in lists
        assert set(people) <= set(entity_pools._FALLBACK_PEOPLE)
        assert "falling back to built-in names" in capsys.readouterr().err
