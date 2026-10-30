import pytest

from biasscope.scenarios import (
    ProfileRejected,
    build_suite,
    find_demographic_terms,
    load_curated_profiles,
    load_suite,
    load_suite_json,
    save_suite,
    validate_profile,
    verify_suite,
)


def test_at_least_ten_curated_profiles_all_valid_and_neutral():
    profiles = load_curated_profiles()
    assert len(profiles) >= 10
    for p in profiles:
        assert find_demographic_terms(p.neutral_fields()) == []


def test_matched_cases_differ_only_in_gender_line(full_suite):
    by_profile = {}
    for c in full_suite.cases:
        by_profile.setdefault(c.profile_id, []).append(c)
    for pid, cases in by_profile.items():
        assert sorted(c.variant for c in cases) == sorted(["woman", "man", "nonbinary"])
        base = cases[0].user_prompt.splitlines()
        for other in cases[1:]:
            lines = other.user_prompt.splitlines()
            assert len(lines) == len(base)
            diffs = [i for i, (x, y) in enumerate(zip(base, lines)) if x != y]
            assert len(diffs) == 1, f"{pid}: more than one line differs"
            assert base[diffs[0]].startswith("- Gender: ") and lines[diffs[0]].startswith("- Gender: ")
            # Removing the gender line makes the prompts byte-identical.
            assert [l for l in base if not l.startswith("- Gender:")] == [l for l in lines if not l.startswith("- Gender:")]


def test_gender_value_appears_only_in_gender_field(full_suite):
    for c in full_suite.cases:
        body = "\n".join(l for l in c.user_prompt.splitlines() if not l.startswith("- Gender:"))
        assert find_demographic_terms({"body": body}) == []


@pytest.mark.parametrize("field,value", [
    ("goals", "She wants to study engineering."),
    ("interests", ["robotics", "girls who code club"]),
    ("experience", ["Helped his father run a shop"]),
    ("academic_strengths", ["Women in STEM scholarship"]),
])
def test_demographic_terms_rejected(field, value):
    raw = load_curated_profiles()[0].model_dump()
    raw[field] = value
    with pytest.raises(ProfileRejected, match="demographic"):
        validate_profile(raw)


def test_malformed_profiles_rejected():
    raw = load_curated_profiles()[0].model_dump()
    with pytest.raises(ProfileRejected, match="schema"):
        validate_profile({**raw, "name": "Alex"})  # extra field (names not allowed)
    with pytest.raises(ProfileRejected, match="schema"):
        validate_profile({**raw, "interests": []})
    bad = dict(raw)
    del bad["goals"]
    with pytest.raises(ProfileRejected, match="schema"):
        validate_profile(bad)


def test_non_gendered_substrings_are_not_false_positives():
    # 'human', 'manufacturing', 'mentor', 'other', 'theater' contain gendered substrings.
    assert find_demographic_terms({"x": "human rights, manufacturing, mentor, other, theater, Shell"}) == []


def test_suite_save_load_replay_is_identical(tmp_path, full_suite):
    path = save_suite(full_suite, tmp_path)
    loaded = load_suite(path)
    assert loaded.content_hash == full_suite.content_hash
    assert [c.model_dump() for c in loaded.cases] == [c.model_dump() for c in full_suite.cases]
    # Rebuilding from the same profiles yields the same deterministic id and hash.
    again = build_suite(load_curated_profiles()[:10])
    assert again.suite_id == full_suite.suite_id and again.content_hash == full_suite.content_hash


def test_tampered_suite_is_rejected(full_suite):
    data = full_suite.model_copy(deep=True)
    data.cases[0].user_prompt = data.cases[0].user_prompt.replace("Grades", "Grades (excellent)")
    assert verify_suite(data)
    with pytest.raises(ValueError, match="verification"):
        load_suite_json(data.model_dump_json())
