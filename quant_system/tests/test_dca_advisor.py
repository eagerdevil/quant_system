from dca_advisor import evaluate_dca_guidance, monthly_budget_cost_note


def _score(**overrides):
    score = {
        "code": "159659", "price": 1.25,
        "score": 84, "premium_info": {"premium_pct": 0.5},
        "indicators": {"rsi": 50},
        "returns": {"r5d": 1, "r60d": 4},
        "vs_ma": {"pct_ma20": 1, "pct_ma60": 1},
    }
    for key, value in overrides.items():
        score[key] = value
    return score


def test_stale_or_short_history_never_produces_entry_signal():
    stale = evaluate_dca_guidance(_score(), data_is_current=False, history_len=200)
    short = evaluate_dca_guidance(_score(), data_is_current=True, history_len=30)
    assert stale["action"].startswith("数据不足")
    assert short["action"].startswith("数据不足")
    assert "不是未来收益概率" in stale["score_note"]


def test_high_premium_skips_this_etf_for_now():
    guidance = evaluate_dca_guidance(
        _score(premium_info={"premium_pct": 6.2, "is_qdii": True,
                             "history": {"current_source": "iopv"}}),
        data_is_current=True, history_len=200
    )
    assert guidance["action"].startswith("本期先跳过")
    assert "6.2%" in guidance["reasons"][0]


def test_overheated_market_keeps_base_dca_but_discourages_chasing():
    score = _score(indicators={"rsi": 73}, returns={"r5d": 9, "r60d": 15})
    guidance = evaluate_dca_guidance(score, data_is_current=True, history_len=200)
    assert guidance["action"].startswith("基础定投可照常")
    assert any("RSI" in reason or "近5日" in reason for reason in guidance["reasons"])


def test_pullback_candidate_does_not_claim_bottom_or_size_order():
    score = _score(indicators={"rsi": 34}, returns={"r5d": -4, "r60d": 2},
                   vs_ma={"pct_ma20": -2, "pct_ma60": -4})
    guidance = evaluate_dca_guidance(score, data_is_current=True, history_len=200)
    assert guidance["action"].startswith("回撤分批候选")
    assert "不据此一次性抄底" in guidance["action"]


def test_long_term_fund_suitability_is_explicitly_unrated():
    guidance = evaluate_dca_guidance(_score(), data_is_current=True, history_len=200)
    assert guidance["long_term_fit"].startswith("未评级")
    assert "跟踪误差" in guidance["long_term_fit"]


def test_budget_note_calculates_minimum_fee_drag_from_current_config():
    note = monthly_budget_cost_note(1000)
    assert "1000元" in note
    assert "5.1元" in note
    assert "20.1元" in note
    assert "券商实际费率" in note


def test_missing_domestic_premium_is_not_reported_as_cross_border_warning():
    guidance = evaluate_dca_guidance(
        _score(premium_info={"premium_pct": None, "is_qdii": False}),
        data_is_current=True, history_len=200
    )
    assert not any("QDII溢价" in reason for reason in guidance["reasons"])


def test_qdii_missing_premium_does_not_get_overridden_by_hot_signal():
    score = _score(
        premium_info={"premium_pct": None, "is_qdii": True},
        indicators={"rsi": 76}, returns={"r5d": 9, "r60d": 15}
    )
    guidance = evaluate_dca_guidance(score, data_is_current=True, history_len=200)
    assert guidance["action"].startswith("QDII溢价数据不足")
    assert any("RSI" in reason for reason in guidance["reasons"])


def test_measured_premium_warning_has_priority_over_overbought_state():
    score = _score(
        premium_info={"premium_pct": 4.2, "is_qdii": True,
                      "data_source": "official_nav", "history": {"current_source": "official_nav"}},
        indicators={"rsi": 75}, returns={"r5d": 9, "r60d": 15}
    )
    guidance = evaluate_dca_guidance(score, data_is_current=True, history_len=200)
    assert guidance["action"].startswith("溢价口径可能滞后")


def test_stale_nav_high_premium_requires_live_iopv_check():
    guidance = evaluate_dca_guidance(
        _score(premium_info={"premium_pct": 6.2, "is_qdii": True,
                             "data_source": "official_nav", "history": {"current_source": "official_nav"}}),
        data_is_current=True, history_len=200
    )
    assert guidance["action"].startswith("溢价口径可能滞后")
    assert any("滞后净值" in reason for reason in guidance["reasons"])
