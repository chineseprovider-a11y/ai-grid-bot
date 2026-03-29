"""
AI Grid Bot — Live Trading Dashboard
Отдельное приложение для мониторинга торговли.

Запуск: streamlit run dashboard.py --server.port=8502
"""

import os
import json
import time
from glob import glob
from datetime import datetime, timezone

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
    }
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
# Заголовок
# ═══════════════════════════════════════════

# Определяем режим
mode_str = "PAPER"
sample_state = None
state_files = glob(os.path.join(DATA_DIR, "live_state_*.json"))
if state_files:
    try:
        with open(state_files[0]) as f:
            sample_state = json.load(f)
        mode_str = sample_state.get("mode", "paper").upper()
    except Exception:
        pass

mode_colors = {"PAPER": "#ffc107", "TESTNET": "#ff9800", "LIVE": "#ff5252"}
mode_color = mode_colors.get(mode_str, "#8b95a5")

st.markdown(f"""
<div class="header-bar">
    <h1 style="margin:0; color:white;">📊 AI Grid Bot — Live Dashboard</h1>
    <p style="color:#8b95a5; margin:5px 0 0 0;">
        Мониторинг торговли в реальном времени |
        <span style="color:{mode_color}; font-weight:bold;">{mode_str} MODE</span>
    </p>
</div>
""", unsafe_allow_html=True)

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
# Общая сводка портфеля
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

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: metric_card("Портфель", f"${total_equity:.2f}", "blue")
with c2: metric_card("P&L", f"${total_pnl:+.2f}", "green" if total_pnl >= 0 else "red")
with c3: metric_card("ROI", f"{total_pnl_pct:+.2f}%", "green" if total_pnl_pct >= 0 else "red")
with c4: metric_card("Реализовано", f"${total_realized:+.2f}", "green" if total_realized >= 0 else "red")
with c5: metric_card("Макс. просадка", f"{max_dd:.1f}%", "red" if max_dd > 5 else "amber")
with c6: metric_card("Активных", f"{running_count} / {len(all_states)}", "green" if running_count > 0 else "red")

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
        status = s.get("status", "?")
        trades = len(s.get("trade_history", []))
        positions = len(s.get("grid", {}).get("bought_levels", {}))
        ai_sig = s.get("ai_state", {}).get("signal", 0)
        dd = s.get("safety", {}).get("current_drawdown", 0) * 100
        indicators = s.get("indicators", {})
        trend = indicators.get("trend", "-")
        rsi = indicators.get("rsi", 50)

        status_emoji = {"running": "🟢", "paused": "⏸️", "stopped": "⏹️", "error": "🔴"}.get(status, "❓")
        sig_emoji = "📈" if ai_sig > 0.2 else ("📉" if ai_sig < -0.2 else "↔️")
        trend_emoji = {"up": "📈", "down": "📉", "neutral": "↔️"}.get(trend, "-")

        pairs_data.append({
            "": status_emoji,
            "Пара": symbol,
            "Баланс": f"${equity:.2f}",
            "P&L": f"${pnl:+.2f}",
            "ROI%": f"{pnl_pct:+.1f}%",
            "AI": f"{sig_emoji} {ai_sig:.2f}",
            "Тренд": f"{trend_emoji}",
            "RSI": f"{rsi:.0f}",
            "Позиц.": positions,
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
ai_acc = ai_state.get("accuracy", 0.5)
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

# Открытые позиции
if bought:
    st.markdown("#### Открытые позиции")
    pos_data = []
    for lvl_key, pos in bought.items():
        buy_price = pos.get("buy_price", 0)
        peak = pos.get("peak_price", buy_price)
        pos_data.append({
            "Уровень": f"${float(lvl_key):.2f}",
            "Кол-во": f"{pos.get('amount', 0):.8f}",
            "Цена покупки": f"${buy_price:.2f}",
            "Пик цены": f"${peak:.2f}",
            "AI при покупке": f"{pos.get('ai_signal_at_buy', 0):.2f}",
            "Время": pos.get("buy_time", "")[:16],
        })
    st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# AI решения (лог)
# ═══════════════════════════════════════════

ai_decisions = live_state.get("ai_decisions", [])
if ai_decisions:
    with st.expander(f"🧠 AI решения ({len(ai_decisions)})", expanded=False):
        recent_decisions = ai_decisions[-20:][::-1]
        dec_data = []
        for d in recent_decisions:
            action = d.get("action", "?")
            action_colors = {
                "buy": "🟢 BUY",
                "sell": "🔴 SELL",
                "block_buy": "🚫 БЛОК",
                "skip_upper": "⏭️ ПРОПУСК",
                "hold": "⏳ ДЕРЖИМ",
                "hold_oversold": "⏳ OVERSOLD",
                "stop_loss": "🔴 СТОП",
                "forced_sell": "🔴 FORCED",
            }
            dec_data.append({
                "Время": d.get("t", "")[:16],
                "Действие": action_colors.get(action, action),
                "Причина": d.get("reason", ""),
                "Цена": f"${d.get('price', 0):.2f}",
                "AI": f"{d.get('signal', 0):.2f}",
                "RSI": f"{d.get('rsi', 50):.0f}",
                "Тренд": d.get("trend", "-"),
            })
        st.dataframe(pd.DataFrame(dec_data), use_container_width=True, hide_index=True)

# Последние сделки
if trades:
    with st.expander(f"Последние сделки ({len(trades)})", expanded=False):
        recent = trades[-15:][::-1]
        trade_data = []
        for t in recent:
            side_emoji = "🟢 BUY" if t["side"] == "buy" else "🔴 SELL"
            trade_data.append({
                "Время": t.get("timestamp", "")[:16],
                "Тип": side_emoji,
                "Кол-во": f"{t.get('amount', 0):.8f}",
                "Цена": f"${t.get('price', 0):.2f}",
                "P&L": f"${t.get('profit', 0):.2f}" if t["side"] == "sell" else "-",
            })
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)

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
st.caption(f"Последнее обновление: {latest_update[:19]} UTC")

# Автообновление
if auto_refresh:
    time.sleep(10)
    st.rerun()
