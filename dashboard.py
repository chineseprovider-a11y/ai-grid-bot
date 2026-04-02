"""
AI Grid Bot — Live Trading Dashboard
Отдельное приложение для мониторинга торговли.

Запуск: streamlit run dashboard.py --server.port=8502
"""

import os
import json
import time
from glob import glob
from datetime import datetime, timezone, timedelta

# Каирское время (UTC+2)
CAIRO_TZ = timezone(timedelta(hours=2))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ═══════════════════════════════════════════
# Настройки страницы
# ═══════════════════════════════════════════

st.set_page_config(
    page_title="AI Grid Bot — Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Веса портфеля для сортировки
PORTFOLIO_WEIGHTS = {
    "BTC/USDT": 30, "ETH/USDT": 25, "SOL/USDT": 12, "BNB/USDT": 12,
    "LINK/USDT": 10, "ADA/USDT": 6, "DOGE/USDT": 5,
}

# ═══════════════════════════════════════════
# Стили
# ═══════════════════════════════════════════

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #151922 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .metric-label { color: #8b95a5; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 28px; font-weight: 700; margin-top: 4px; }
    .metric-small { font-size: 20px; font-weight: 600; margin-top: 4px; }
    .green { color: #00e676; }
    .red { color: #ff5252; }
    .blue { color: #448aff; }
    .amber { color: #ffc107; }
    .white { color: #ffffff; }
    .header-bar {
        background: linear-gradient(90deg, #1a1f2e 0%, #0e1117 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #2d3748;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-left { flex: 1; }
    .header-right {
        display: flex;
        gap: 30px;
        align-items: center;
    }
    .header-stat {
        text-align: center;
    }
    .header-stat-label { color: #8b95a5; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
    .header-stat-value { font-size: 22px; font-weight: 700; margin-top: 2px; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.3rem; }
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, color="white", small=False):
    size_class = "metric-small" if small else "metric-value"
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="{size_class} {color}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def fmt_time(iso_str: str) -> str:
    """Форматирует ISO время в каирское время: 02.04 17:31"""
    if not iso_str:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_cairo = dt.astimezone(CAIRO_TZ)
        return dt_cairo.strftime("%d.%m %H:%M")
    except Exception:
        return iso_str[:16]


def load_all_states():
    """Загружает все state-файлы, сортирует по весу портфеля."""
    state_files = sorted(glob(os.path.join(DATA_DIR, "live_state_*.json")))
    states = []
    for sp in state_files:
        try:
            with open(sp) as f:
                states.append(json.load(f))
        except Exception:
            pass
    # Сортируем по весу портфеля (убывание)
    states.sort(key=lambda s: PORTFOLIO_WEIGHTS.get(s.get("symbol", ""), 0), reverse=True)
    return states


# ═══════════════════════════════════════════
# Автообновление
# ═══════════════════════════════════════════

auto_refresh = st.sidebar.checkbox("Автообновление (10с)", value=True)
if auto_refresh:
    st.sidebar.caption("Страница обновляется каждые 10 секунд")

# ═══════════════════════════════════════════
# Данные
# ═══════════════════════════════════════════

all_states = load_all_states()

if not all_states:
    st.warning("Нет данных о торговле. Убедитесь что бот запущен.")
    if auto_refresh:
        time.sleep(10)
        st.rerun()
    st.stop()

# ═══════════════════════════════════════════
# Общие расчёты
# ═══════════════════════════════════════════

total_equity = sum(s.get("current_equity", 0) for s in all_states)
total_investment = sum(s.get("initial_investment", 0) for s in all_states)
total_pnl = total_equity - total_investment
total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
total_trades = sum(len(s.get("trade_history", [])) for s in all_states)
total_positions = sum(len(s.get("grid", {}).get("bought_levels", {})) for s in all_states)
running_count = sum(1 for s in all_states if s.get("status") == "running")
max_dd = max((s.get("safety", {}).get("current_drawdown", 0) for s in all_states), default=0) * 100

total_realized = sum(
    sum(t.get("profit", 0) for t in s.get("trade_history", []) if t.get("side") == "sell")
    for s in all_states
)
sell_trades = sum(
    sum(1 for t in s.get("trade_history", []) if t.get("side") == "sell")
    for s in all_states
)

# Режим
mode_str = "PAPER"
sample_state = all_states[0] if all_states else None
if sample_state:
    mode_str = sample_state.get("mode", "paper").upper()

mode_colors = {"PAPER": "#ffc107", "TESTNET": "#ff9800", "LIVE": "#ff5252"}
mode_color = mode_colors.get(mode_str, "#8b95a5")

# Uptime
started_at = min((s.get("started_at", "") for s in all_states), default="")
uptime_str = ""
if started_at:
    try:
        dt_start = datetime.fromisoformat(started_at)
        delta = datetime.now(timezone.utc) - dt_start
        days = delta.days
        hours = delta.seconds // 3600
        uptime_str = f"{days}д {hours}ч" if days > 0 else f"{hours}ч {(delta.seconds % 3600) // 60}м"
    except Exception:
        pass

# BTC price
btc_price = 0
for s in all_states:
    if s.get("symbol") == "BTC/USDT":
        curve = s.get("equity_curve", [])
        if curve:
            btc_price = curve[-1].get("p", 0)
        break

# ═══════════════════════════════════════════
# Заголовок с быстрой сводкой
# ═══════════════════════════════════════════

pnl_color = "#00e676" if total_pnl >= 0 else "#ff5252"

header_right_html = f"""
<div class="header-right">
    <div class="header-stat">
        <div class="header-stat-label">BTC</div>
        <div class="header-stat-value white">${btc_price:,.0f}</div>
    </div>
    <div class="header-stat">
        <div class="header-stat-label">P&L</div>
        <div class="header-stat-value" style="color:{pnl_color};">${total_pnl:+.2f} ({total_pnl_pct:+.1f}%)</div>
    </div>
    <div class="header-stat">
        <div class="header-stat-label">Аптайм</div>
        <div class="header-stat-value white">{uptime_str}</div>
    </div>
</div>
"""

st.markdown(f"""
<div class="header-bar">
    <div class="header-left">
        <h1 style="margin:0; color:white;">📊 AI Grid Bot — Live Dashboard</h1>
        <p style="color:#8b95a5; margin:5px 0 0 0;">
            Мониторинг торговли в реальном времени |
            <span style="color:{mode_color}; font-weight:bold;">{mode_str} MODE</span>
        </p>
    </div>
    {header_right_html}
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# Общая сводка портфеля (одна строка HTML)
# ═══════════════════════════════════════════

pnl_cls = "green" if total_pnl >= 0 else "red"
real_cls = "green" if total_realized >= 0 else "red"
dd_cls = "red" if max_dd > 5 else "amber"

st.markdown(f"""
<div style="display:flex; gap:10px; margin-bottom:10px;">
    <div class="metric-box" style="flex:1; min-width:0;">
        <div class="metric-label">Портфель</div>
        <div class="metric-value blue">${total_equity:.2f}</div>
    </div>
    <div class="metric-box" style="flex:1; min-width:0;">
        <div class="metric-label">P&L</div>
        <div class="metric-value {pnl_cls}">${total_pnl:+.2f} ({total_pnl_pct:+.1f}%)</div>
    </div>
    <div class="metric-box" style="flex:1; min-width:0;">
        <div class="metric-label">Позиции</div>
        <div class="metric-value amber">{total_positions}</div>
    </div>
    <div class="metric-box" style="flex:1; min-width:0;">
        <div class="metric-label">Реализовано ({sell_trades} сд.)</div>
        <div class="metric-value {real_cls}">${total_realized:+.2f}</div>
    </div>
    <div class="metric-box" style="flex:1; min-width:0;">
        <div class="metric-label">Макс. просадка</div>
        <div class="metric-value {dd_cls}">{max_dd:.1f}%</div>
    </div>
    <div class="metric-box" style="flex:1; min-width:0;">
        <div class="metric-label">Всего сделок</div>
        <div class="metric-value blue">{total_trades}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# 3 последние сделки (под сводкой)
# ═══════════════════════════════════════════

all_recent_trades = []
for s in all_states:
    sym = s.get("symbol", "?")
    for t in s.get("trade_history", []):
        t_copy = dict(t)
        t_copy["_symbol"] = sym
        all_recent_trades.append(t_copy)

# Сортируем по времени (новые первые)
all_recent_trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
last_3 = all_recent_trades[:3]

if last_3:
    trade_cols = st.columns(3)
    for idx, t in enumerate(last_3):
        side = t.get("side", "?")
        is_buy = side == "buy"
        side_icon = "🟢 BUY" if is_buy else "🔴 SELL"
        side_color = "#00e676" if is_buy else "#ff5252"
        sym = t.get("_symbol", "?")
        price = t.get("price", 0)
        ts = fmt_time(t.get("timestamp", ""))
        profit = t.get("profit", 0)
        pnl_color = "#00e676" if profit >= 0 else "#ff5252"
        pnl_html = f'<span style="color:{pnl_color}; margin-left:8px;">P&L: ${profit:+.2f}</span>' if not is_buy else ""

        with trade_cols[idx]:
            st.markdown(f"""<div class="metric-box" style="padding:12px 16px;">
<div style="display:flex; justify-content:space-between; align-items:center;">
<span style="color:{side_color}; font-weight:700; font-size:14px;">{side_icon}</span>
<span style="color:#8b95a5; font-size:12px;">{ts}</span>
</div>
<div style="color:white; font-size:16px; font-weight:600; margin-top:4px;">{sym} @ ${price:,.2f}</div>
<div style="color:#8b95a5; font-size:12px; margin-top:2px;">{pnl_html}</div>
</div>""", unsafe_allow_html=True)

st.markdown("")

# ═══════════════════════════════════════════
# Таблица пар + Распределение
# ═══════════════════════════════════════════

col_table, col_chart = st.columns([1, 1])

with col_table:
    st.markdown("### Торговые пары")

    pairs_data = []
    for s in all_states:
        symbol = s.get("symbol", "?")
        equity = s.get("current_equity", 0)
        inv = s.get("initial_investment", 0)
        pnl = equity - inv
        pnl_pct = (pnl / inv * 100) if inv > 0 else 0
        positions = len(s.get("grid", {}).get("bought_levels", {}))
        indicators = s.get("indicators", {})
        trend = indicators.get("trend", "-")
        rsi = indicators.get("rsi", 50)
        weight = PORTFOLIO_WEIGHTS.get(symbol, 0)

        status = s.get("status", "?")
        status_emoji = {"running": "🟢", "paused": "⏸️", "stopped": "⏹️", "error": "🔴"}.get(status, "❓")
        trend_emoji = {"up": "📈", "down": "📉", "neutral": "↔️"}.get(trend, "-")

        # Текущая цена из equity_curve
        cur_price = 0
        curve = s.get("equity_curve", [])
        if curve:
            cur_price = curve[-1].get("p", 0)

        pairs_data.append({
            "": status_emoji,
            "Пара": symbol,
            "Вес": f"{weight}%",
            "Цена": f"${cur_price:,.2f}" if cur_price else "-",
            "Баланс": f"${equity:.2f}",
            "P&L": f"${pnl:+.2f} ({pnl_pct:+.1f}%)",
            "Тренд": trend_emoji,
            "RSI": f"{rsi:.0f}",
            "Поз.": positions,
        })

    df_pairs = pd.DataFrame(pairs_data)
    st.dataframe(df_pairs, use_container_width=True, hide_index=True, height=min(400, 40 + 35 * len(pairs_data)))

with col_chart:
    st.markdown("### Распределение портфеля")

    labels = [s.get("symbol", "?") for s in all_states]
    values = [s.get("current_equity", 0) for s in all_states]
    colors = ["#448aff", "#00e676", "#ffc107", "#ff5252", "#e040fb", "#00bcd4", "#ff9800"]

    fig_pie = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.5,
        marker_colors=colors[:len(labels)],
        textinfo="label+percent",
        textfont_size=12,
    )])
    fig_pie.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

st.markdown("")

# ═══════════════════════════════════════════
# Графики equity
# ═══════════════════════════════════════════

st.markdown("### Графики баланса")

fig_combined = go.Figure()
colors_list = ["#448aff", "#00e676", "#ffc107", "#ff5252", "#e040fb", "#00bcd4", "#ff9800"]

for i, s in enumerate(all_states):
    curve = s.get("equity_curve", [])
    if curve:
        symbol = s.get("symbol", "?")
        fig_combined.add_trace(go.Scatter(
            x=[p["t"] for p in curve],
            y=[p["e"] for p in curve],
            mode="lines",
            name=symbol,
            line=dict(color=colors_list[i % len(colors_list)], width=2),
        ))

if fig_combined.data:
    fig_combined.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Баланс ($)",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_combined, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Графики появятся после первых торговых циклов")

# ═══════════════════════════════════════════
# Детали выбранной пары
# ═══════════════════════════════════════════

st.markdown("---")
st.markdown("### Детали пары")

symbol_list = [s.get("symbol", "?") for s in all_states]
selected = st.selectbox("Выберите пару:", symbol_list, label_visibility="collapsed")
live_state = next((s for s in all_states if s.get("symbol") == selected), all_states[0])

detail_cols = st.columns(6)
equity = live_state.get("current_equity", 0)
inv = live_state.get("initial_investment", 0)
pnl = equity - inv
pnl_pct = (pnl / inv * 100) if inv > 0 else 0
dd = live_state.get("safety", {}).get("current_drawdown", 0) * 100
ai_state = live_state.get("ai_state", {})
ai_sig = ai_state.get("signal", 0)
trades = live_state.get("trade_history", [])
bought = live_state.get("grid", {}).get("bought_levels", {})
indicators = live_state.get("indicators", {})

with detail_cols[0]: metric_card("Баланс", f"${equity:.2f}", "blue", small=True)
with detail_cols[1]: metric_card("P&L", f"${pnl:+.2f}", "green" if pnl >= 0 else "red", small=True)
with detail_cols[2]: metric_card("ROI", f"{pnl_pct:+.2f}%", "green" if pnl_pct >= 0 else "red", small=True)
with detail_cols[3]: metric_card("Просадка", f"{dd:.1f}%", "red" if dd > 5 else "amber", small=True)
with detail_cols[4]:
    sig_text = "📈" if ai_sig > 0.2 else ("📉" if ai_sig < -0.2 else "↔️")
    metric_card("AI сигнал", f"{sig_text} {ai_sig:.2f}", "blue", small=True)
with detail_cols[5]:
    trend = indicators.get("trend", "neutral")
    rsi = indicators.get("rsi", 50)
    trend_emoji = {"up": "📈", "down": "📉", "neutral": "↔️"}.get(trend, "-")
    metric_card("Тренд / RSI", f"{trend_emoji} {rsi:.0f}", "green" if trend == "up" else ("red" if trend == "down" else "amber"), small=True)

# Текущая цена пары
cur_price = 0
pair_curve = live_state.get("equity_curve", [])
if pair_curve:
    cur_price = pair_curve[-1].get("p", 0)

# Открытые позиции
if bought:
    st.markdown("#### Открытые позиции")
    pos_data = []
    for lvl_key, pos in bought.items():
        buy_price = pos.get("buy_price", 0)
        peak = pos.get("peak_price", buy_price)
        amount = pos.get("amount", 0)
        # Текущий P&L позиции
        if cur_price > 0 and buy_price > 0:
            pos_pnl = (cur_price - buy_price) * amount
            pos_pnl_pct = (cur_price - buy_price) / buy_price * 100
            pnl_str = f"${pos_pnl:+.2f} ({pos_pnl_pct:+.1f}%)"
        else:
            pnl_str = "-"
        pos_data.append({
            "Уровень": f"${float(lvl_key):.2f}",
            "Кол-во": f"{amount:.8f}",
            "Цена покупки": f"${buy_price:.2f}",
            "Текущий P&L": pnl_str,
            "Пик цены": f"${peak:.2f}",
            "AI при покупке": f"{pos.get('ai_signal_at_buy', 0):.2f}",
            "Время": fmt_time(pos.get("buy_time", "")),
        })
    st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# AI решения (лог) — все записи
# ═══════════════════════════════════════════

ai_decisions = live_state.get("ai_decisions", [])
if ai_decisions:
    with st.expander(f"🧠 AI решения ({len(ai_decisions)})", expanded=False):
        # Показываем все решения (новые сверху)
        recent_decisions = ai_decisions[::-1]
        dec_data = []
        for d in recent_decisions:
            action = d.get("action", "?")
            action_labels = {
                "buy": "🟢 ПОКУПКА",
                "sell": "🔴 ПРОДАЖА",
                "block_buy": "🚫 БЛОК",
                "skip_upper": "⏭️ ПРОПУСК",
                "hold": "⏳ ДЕРЖИМ",
                "hold_oversold": "⏳ ПЕРЕПРОДАН",
                "stop_loss": "🔴 СТОП-ЛОСС",
                "forced_sell": "🔴 СТОП-ЛОСС",
            }
            trend_val = d.get("trend", "-")
            trend_emoji = {"up": "📈", "down": "📉", "neutral": "↔️"}.get(trend_val, "-")
            dec_data.append({
                "Время": fmt_time(d.get("t", "")),
                "Действие": action_labels.get(action, action),
                "Причина": d.get("reason", ""),
                "Цена": f"${d.get('price', 0):.2f}",
                "AI": f"{d.get('signal', 0):.2f}",
                "RSI": f"{d.get('rsi', 50):.0f}",
                "Тренд": trend_emoji,
            })
        st.dataframe(pd.DataFrame(dec_data), use_container_width=True, hide_index=True,
                     height=min(600, 40 + 35 * len(dec_data)))

# Последние сделки — все записи
if trades:
    with st.expander(f"Последние сделки ({len(trades)})", expanded=False):
        # Показываем все сделки (новые сверху)
        all_trades = trades[::-1]
        trade_data = []
        for t in all_trades:
            side_emoji = "🟢 BUY" if t["side"] == "buy" else "🔴 SELL"
            trade_data.append({
                "Время": fmt_time(t.get("timestamp", "")),
                "Тип": side_emoji,
                "Кол-во": f"{t.get('amount', 0):.8f}",
                "Цена": f"${t.get('price', 0):.2f}",
                "P&L": f"${t.get('profit', 0):.2f}" if t["side"] == "sell" else "-",
            })
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True,
                     height=min(600, 40 + 35 * len(trade_data)))

# ═══════════════════════════════════════════
# Управление
# ═══════════════════════════════════════════

st.markdown("---")
st.markdown("### Управление")

cmd_col1, cmd_col2, cmd_col3, cmd_col4 = st.columns(4)
cmd_path = os.path.join(DATA_DIR, "live_commands.json")

with cmd_col1:
    if st.button("⏸️ Пауза всех", use_container_width=True):
        with open(cmd_path, "w") as f:
            json.dump({"command": "pause"}, f)
        st.toast("Все боты на паузу")

with cmd_col2:
    if st.button("▶️ Продолжить", use_container_width=True):
        with open(cmd_path, "w") as f:
            json.dump({"command": "resume"}, f)
        st.toast("Все боты возобновлены")

with cmd_col3:
    if st.button("⏹️ Стоп всех", type="primary", use_container_width=True):
        with open(cmd_path, "w") as f:
            json.dump({"command": "stop"}, f)
        st.toast("Все боты остановлены")

with cmd_col4:
    if st.button("🔄 Обновить", use_container_width=True):
        st.rerun()

latest_update = max(
    (s.get("updated_at", "") for s in all_states),
    default=""
)
st.caption(f"Последнее обновление: {fmt_time(latest_update)} (Каир)")

# Автообновление
if auto_refresh:
    time.sleep(10)
    st.rerun()
