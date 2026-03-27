"""AI Grid Trading Bot — Современный интерфейс."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt
import os

st.set_page_config(page_title="AI Grid Bot", page_icon="⚡", layout="wide")

# ─── Стили ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .block-container {padding-top: 1rem; max-width: 1200px;}

    /* Статус-бар */
    .status-bar {
        display: flex; gap: 12px; padding: 12px 0;
        flex-wrap: wrap;
    }
    .status-chip {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 14px; border-radius: 20px;
        font-size: 13px; font-weight: 500;
    }
    .chip-ready { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid rgba(0,230,118,0.3); }
    .chip-missing { background: rgba(255,82,82,0.12); color: #ff5252; border: 1px solid rgba(255,82,82,0.25); }

    /* Карточки */
    .card {
        background: linear-gradient(145deg, #1e1e2e, #252540);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 20px;
        margin: 8px 0;
        transition: border-color 0.2s;
    }
    .card:hover { border-color: rgba(255,255,255,0.15); }
    .card-title { font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
    .card-value { font-size: 28px; font-weight: 700; }
    .val-green { color: #00e676; }
    .val-red { color: #ff5252; }
    .val-blue { color: #60a5fa; }
    .val-amber { color: #fbbf24; }

    /* Шаги */
    .step-row {
        display: flex; align-items: center; gap: 14px;
        padding: 14px 18px; margin: 6px 0;
        background: linear-gradient(145deg, #1e1e2e, #252540);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
    }
    .step-num {
        width: 32px; height: 32px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 14px; flex-shrink: 0;
    }
    .step-active { background: #6366f1; color: white; }
    .step-done { background: #00e676; color: #111; }
    .step-pending { background: rgba(255,255,255,0.08); color: #666; }
    .step-text { flex: 1; }
    .step-label { font-weight: 600; font-size: 15px; }
    .step-desc { font-size: 13px; color: #888; margin-top: 2px; }

    /* Пресет-кнопки */
    .preset-card {
        background: linear-gradient(145deg, #1e1e2e, #252540);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .preset-card h4 { margin: 0 0 4px 0; font-size: 16px; }
    .preset-card p { margin: 0; font-size: 13px; color: #999; }

    /* Вкладки */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    .hint { color: #888; font-size: 13px; }
    .divider { border-top: 1px solid rgba(255,255,255,0.06); margin: 16px 0; }
</style>
""", unsafe_allow_html=True)

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT"]
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

PRESETS = {
    "Осторожный": {"icon": "🛡️", "grid_count": 8, "range_pct": 1.5,
        "desc": "Узкая сетка, меньше риска", "detail": "Для стабильных монет и боковика",
        "opt_gc": (5, 12, 1), "opt_rp": (0.5, 2.0, 0.5)},
    "Стандартный": {"icon": "⚖️", "grid_count": 12, "range_pct": 2.5,
        "desc": "Баланс риска и прибыли", "detail": "Хороший старт для большинства пар",
        "opt_gc": (8, 20, 4), "opt_rp": (1.0, 4.0, 0.5)},
    "Агрессивный": {"icon": "🔥", "grid_count": 20, "range_pct": 4.0,
        "desc": "Широкая сетка, больше сделок", "detail": "Для волатильных монет (SOL, DOGE)",
        "opt_gc": (15, 30, 5), "opt_rp": (2.0, 6.0, 0.5)},
}


# ─── Утилиты ───
def check_data():
    """Возвращает список пар, для которых есть данные."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return [s for s in SYMBOLS
            if os.path.exists(os.path.join(DATA_DIR, f"{s.replace('/', '_')}_1h.csv"))]


def check_models():
    """Возвращает список пар, для которых есть обученные модели."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    return [s for s in SYMBOLS
            if os.path.exists(os.path.join(MODEL_DIR, f"{s.replace('/', '_')}_lstm.keras"))]


def metric_card(title, value, color="white"):
    st.markdown(f'''<div class="card">
        <div class="card-title">{title}</div>
        <div class="card-value val-{color}">{value}</div>
    </div>''', unsafe_allow_html=True)


def show_results(res, label):
    """Карточки результата бэктеста."""
    color = "green" if res.net_profit > 0 else "red"
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Прибыль", f"${res.net_profit:.2f}", color)
    with c2: metric_card("ROI", f"{res.roi_pct:.2f}%", color)
    with c3: metric_card("Сделок", str(res.total_trades), "blue")
    with c4: metric_card("Win Rate", f"{res.win_rate*100:.1f}%", "amber")

    with st.expander("Подробнее"):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Комиссии", f"${res.total_fees:.2f}")
        d2.metric("Max Drawdown", f"{res.max_drawdown*100:.2f}%")
        d3.metric("Sharpe Ratio", f"{res.sharpe_ratio:.2f}")
        d4.metric("Win/Loss", f"{res.win_trades}/{res.loss_trades}")

    if res.equity_curve:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=res.equity_curve, mode="lines", name="Баланс",
            line=dict(color="#00e676" if "AI" in label else "#60a5fa", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,230,118,0.05)" if "AI" in label else "rgba(96,165,250,0.05)",
        ))
        fig.add_hline(y=res.investment, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                     annotation_text="Старт")
        fig.update_layout(template="plotly_dark", height=300,
                         margin=dict(l=0, r=0, t=30, b=0),
                         yaxis_title="Баланс ($)",
                         title=dict(text=f"Баланс — {label}", font_size=14))
        st.plotly_chart(fig, use_container_width=True)


# ─── Заголовок + статус ───
st.markdown("# ⚡ AI Grid Bot")

available_data = check_data()
available_models = check_models()

# Статус-бар
chips = []
if available_data:
    chips.append(f'<span class="status-chip chip-ready">✓ Данные: {len(available_data)} пар</span>')
else:
    chips.append('<span class="status-chip chip-missing">✕ Нет данных</span>')
if available_models:
    chips.append(f'<span class="status-chip chip-ready">✓ Модели: {len(available_models)} пар</span>')
else:
    chips.append('<span class="status-chip chip-missing">✕ Модели не обучены</span>')
st.markdown(f'<div class="status-bar">{"".join(chips)}</div>', unsafe_allow_html=True)


# ─── Вкладки (логичный порядок) ───
tab_start, tab_backtest, tab_optimize, tab_live = st.tabs([
    "🚀 Старт", "📈 Бэктест", "🔧 Оптимизация", "🔮 Прогноз",
])


# ═══════════════════════════════════════════════════════════════
# TAB: Старт — загрузка данных + обучение + статус
# ═══════════════════════════════════════════════════════════════
with tab_start:
    # Пошаговый гайд
    step1_done = len(available_data) > 0
    step2_done = len(available_models) > 0

    def step_html(num, label, desc, done):
        cls = "step-done" if done else "step-active" if (num == 1 and not done) or (num == 2 and step1_done and not done) else "step-pending"
        check = "✓" if done else str(num)
        return f'''<div class="step-row">
            <div class="step-num {cls}">{check}</div>
            <div class="step-text"><div class="step-label">{label}</div><div class="step-desc">{desc}</div></div>
        </div>'''

    st.markdown(
        step_html(1, "Загрузить данные", "Исторические свечи с Binance (бесплатно, без ключей)", step1_done) +
        step_html(2, "Обучить модель", "AI научится предсказывать движение цены", step2_done) +
        step_html(3, "Тестировать", "Запустить бэктест и найти лучшие настройки", False),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Шаг 1: Данные ──
    st.subheader("① Загрузка данных")

    d_col1, d_col2 = st.columns([2, 1])
    with d_col1:
        data_symbols = st.multiselect("Пары", SYMBOLS, default=SYMBOLS, key="d_sym",
                                      help="Какие криптовалюты загрузить")
    with d_col2:
        days = st.select_slider("Период", options=[30, 60, 90, 180, 365], value=90,
                                key="d_days", help="Больше = точнее тест")
        st.caption(f"{days} дней истории")

    if st.button("📥 Загрузить", type="primary", use_container_width=True, key="btn_data"):
        exchange = ccxt.binance({"enableRateLimit": True})
        progress = st.progress(0, text="Загружаю...")

        for idx, symbol in enumerate(data_symbols):
            progress.progress((idx) / len(data_symbols), text=f"{symbol}...")
            try:
                from ai.data_collector import fetch_historical_data, save_to_csv
                df = fetch_historical_data(exchange, symbol, "1h", days)
                save_to_csv(df, symbol, "1h")
                st.toast(f"✅ {symbol}: {len(df)} свечей")
            except Exception as e:
                st.error(f"❌ {symbol}: {e}")
            progress.progress((idx + 1) / len(data_symbols))

        progress.progress(1.0, text="Готово!")
        st.success(f"Загружено {len(data_symbols)} пар")
        st.rerun()

    # Показать имеющиеся данные
    if available_data:
        with st.expander(f"📂 Имеющиеся данные ({len(available_data)} пар)", expanded=False):
            for sym in available_data:
                csv_path = os.path.join(DATA_DIR, f"{sym.replace('/', '_')}_1h.csv")
                n = len(pd.read_csv(csv_path))
                st.write(f"**{sym}** — {n} свечей")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Шаг 2: Обучение ──
    st.subheader("② Обучение модели")

    if not available_data:
        st.info("Сначала загрузите данные (шаг 1)")
    else:
        t_col1, t_col2 = st.columns([2, 1])
        with t_col1:
            train_symbols = st.multiselect("Пары для обучения", available_data,
                                           default=available_data, key="t_sym")
        with t_col2:
            speed = st.radio("Скорость", ["Быстро (15 эпох)", "Точно (50 эпох)"],
                            key="t_speed", horizontal=True)
            epochs = 15 if "15" in speed else 50

        if st.button("🧠 Обучить", type="primary", use_container_width=True, key="btn_train"):
            try:
                import tensorflow as tf
            except ImportError:
                st.error("TensorFlow не установлен! Выполните: `pip install tensorflow`")
                st.stop()

            from ai.feature_engineer import prepare_training_data, FEATURE_COLUMNS
            from ai.model import build_lstm_model, train_model, save_model

            progress = st.progress(0, text="Обучаю...")

            for idx, symbol in enumerate(train_symbols):
                csv_path = os.path.join(DATA_DIR, f"{symbol.replace('/', '_')}_1h.csv")
                df = pd.read_csv(csv_path, parse_dates=["timestamp"])
                if len(df) < 200:
                    st.warning(f"⚠️ {symbol}: мало данных ({len(df)} < 200)")
                    continue

                progress.progress(idx / len(train_symbols), text=f"Обучаю {symbol}...")

                X_train, y_train, X_val, y_val, scaler = prepare_training_data(df)
                model = build_lstm_model(48, len(FEATURE_COLUMNS), n_classes=3)

                with st.spinner(f"Обучаю {symbol}..."):
                    history = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)

                save_model(model, scaler, symbol)
                val_acc = max(history.history["val_accuracy"])
                st.toast(f"✅ {symbol}: точность {val_acc*100:.1f}%")
                progress.progress((idx + 1) / len(train_symbols))

            progress.progress(1.0, text="Готово!")
            st.success("Модели обучены!")
            st.rerun()

        # Обученные модели
        if available_models:
            with st.expander(f"🧠 Обученные модели ({len(available_models)} пар)", expanded=False):
                for sym in available_models:
                    fpath = os.path.join(MODEL_DIR, f"{sym.replace('/', '_')}_lstm.keras")
                    size_mb = os.path.getsize(fpath) / 1024 / 1024
                    st.write(f"**{sym}** — {size_mb:.1f} MB")

        st.markdown("""
        <div class="card" style="margin-top:12px;">
            <div class="card-title">💡 Совет</div>
            <div style="font-size:14px; color:#bbb;">
                Обучение на CPU медленное. Для GPU используйте файл <code>train_colab.ipynb</code> в Google Colab,
                затем скопируйте модели в папку <code>models/</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB: Бэктест
# ═══════════════════════════════════════════════════════════════
with tab_backtest:
    if not available_data:
        st.info("⬅️ Перейдите на вкладку «Старт» и загрузите данные")
        st.stop()

    st.markdown("Проверьте, сколько заработал бы бот на реальных данных")

    col1, col2 = st.columns(2)
    with col1:
        bt_symbol = st.selectbox("Пара", available_data, key="bt_sym")
        bt_investment = st.number_input("Депозит ($)", 50, 10000, 300, step=50, key="bt_inv")

    with col2:
        preset_names = list(PRESETS.keys())
        bt_preset_name = st.radio("Профиль риска", preset_names, index=1,
                                  key="bt_preset", horizontal=True,
                                  help="Выберите стиль торговли")
        preset = PRESETS[bt_preset_name]
        st.markdown(f'''<div class="preset-card">
            <h4>{preset["icon"]} {bt_preset_name}</h4>
            <p>{preset["desc"]}<br>{preset["detail"]}</p>
            <p style="margin-top:8px;color:#60a5fa;font-size:14px;">
                {preset["grid_count"]} уровней · {preset["range_pct"]}% диапазон
            </p>
        </div>''', unsafe_allow_html=True)

    bt_grid_count = preset["grid_count"]
    bt_range_pct = preset["range_pct"]

    with st.expander("⚙️ Ручная настройка"):
        m1, m2 = st.columns(2)
        with m1:
            bt_grid_count = st.slider("Уровней сетки", 5, 30, preset["grid_count"], key="bt_gc",
                                     help="Больше = чаще сделки, мельче прибыль")
        with m2:
            bt_range_pct = st.slider("Диапазон %", 0.5, 6.0, preset["range_pct"], 0.1, key="bt_rp",
                                    help="Ширина ценового коридора")

    # Стратегия
    has_model = bt_symbol in available_models
    strategies = ["Grid + AI (сравнить)", "Только Grid", "Только AI Grid"]
    if not has_model:
        strategies = ["Только Grid"]
        st.caption("ℹ️ Модель не обучена — доступен только обычный Grid")

    bt_strategy = st.radio("Стратегия", strategies, horizontal=True, key="bt_strat")

    if st.button("▶ Запустить тест", type="primary", use_container_width=True, key="btn_bt"):
        csv_path = os.path.join(DATA_DIR, f"{bt_symbol.replace('/', '_')}_1h.csv")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])

        from ai.backtest import GridBacktest, AIGridBacktest

        results = {}

        run_grid = "Grid" in bt_strategy and "Только AI" not in bt_strategy
        run_ai = "AI" in bt_strategy and has_model

        if run_grid:
            with st.spinner("Тестирую Grid..."):
                grid_bt = GridBacktest(bt_investment, bt_grid_count, bt_range_pct)
                results["grid"] = grid_bt.run(df, bt_symbol)

        if run_ai:
            from ai.model import load_model
            model, scaler = load_model(bt_symbol)
            with st.spinner("Тестирую AI Grid..."):
                ai_bt = AIGridBacktest(bt_investment, bt_grid_count, bt_range_pct,
                                      model=model, scaler=scaler)
                results["ai_grid"] = ai_bt.run(df, bt_symbol)

        for name, res in results.items():
            st.divider()
            label = "Grid" if name == "grid" else "AI Grid"
            st.subheader(f"{'📊' if name == 'grid' else '🤖'} {label}")
            show_results(res, label)

        if "grid" in results and "ai_grid" in results:
            st.divider()
            diff = results["ai_grid"].net_profit - results["grid"].net_profit
            if diff > 0:
                st.success(f"🏆 AI Grid лучше на **${diff:.2f}**")
            elif diff < 0:
                st.warning(f"Grid лучше на **${abs(diff):.2f}**")
            else:
                st.info("Результаты одинаковые")


# ═══════════════════════════════════════════════════════════════
# TAB: Оптимизация
# ═══════════════════════════════════════════════════════════════
with tab_optimize:
    if not available_data:
        st.info("⬅️ Перейдите на вкладку «Старт» и загрузите данные")
        st.stop()

    st.markdown("Автоподбор лучших параметров сетки")

    o1, o2 = st.columns(2)
    with o1:
        opt_symbol = st.selectbox("Пара", available_data, key="opt_sym")
        opt_investment = st.number_input("Депозит ($)", 50, 10000, 300, step=50, key="opt_inv")
    with o2:
        opt_preset_name = st.radio("Диапазон поиска", list(PRESETS.keys()),
                                   index=1, key="opt_preset", horizontal=True)
        opt_use_ai = st.checkbox("С AI-предсказаниями", key="opt_ai",
                                help="Использовать обученную модель")

    opt_p = PRESETS[opt_preset_name]
    gc_min, gc_max, gc_step = opt_p["opt_gc"]
    rp_min, rp_max, rp_step = opt_p["opt_rp"]

    opt_metric = ("roi_pct", "ROI %")
    with st.expander("⚙️ Расширенные настройки"):
        ac1, ac2 = st.columns(2)
        with ac1:
            gc_min = st.number_input("Уровней (мин)", 3, 50, gc_min, key="o_gcn")
            gc_max = st.number_input("Уровней (макс)", 3, 50, gc_max, key="o_gcx")
            gc_step = st.number_input("Шаг", 1, 10, gc_step, key="o_gcs")
        with ac2:
            rp_min = st.number_input("Диапазон % (мин)", 0.5, 10.0, rp_min, 0.5, key="o_rpn")
            rp_max = st.number_input("Диапазон % (макс)", 0.5, 10.0, rp_max, 0.5, key="o_rpx")
            rp_step = st.number_input("Шаг %", 0.1, 2.0, rp_step, 0.1, key="o_rps")
        opt_metric = st.selectbox("Оптимизировать по", [
            ("roi_pct", "ROI % (рекомендуется)"),
            ("sharpe_ratio", "Sharpe Ratio (доходность/риск)"),
            ("net_profit", "Чистая прибыль ($)"),
            ("win_rate", "Win Rate (% прибыльных)"),
        ], format_func=lambda x: x[1], key="o_metric")

    grid_counts = list(range(int(gc_min), int(gc_max) + 1, int(gc_step)))
    range_pcts = [round(rp_min + i * rp_step, 1)
                  for i in range(int((rp_max - rp_min) / rp_step) + 1)]
    total = len(grid_counts) * len(range_pcts)

    st.caption(f"Будет проверено **{total}** комбинаций")

    if st.button("🔍 Найти лучшие настройки", type="primary", use_container_width=True, key="btn_opt"):
        csv_path = os.path.join(DATA_DIR, f"{opt_symbol.replace('/', '_')}_1h.csv")
        if not os.path.exists(csv_path):
            st.error("Нет данных!")
            st.stop()

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        from ai.backtest import GridOptimizer

        model, scaler = None, None
        if opt_use_ai:
            from ai.model import load_model
            model, scaler = load_model(opt_symbol)
            if model is None:
                st.error("Модель не обучена!")
                st.stop()

        optimizer = GridOptimizer(
            investment=opt_investment,
            grid_counts=grid_counts,
            range_pcts=range_pcts,
            use_ai=opt_use_ai,
            model=model, scaler=scaler,
        )

        progress = st.progress(0, text="Оптимизация...")

        def update_progress(current, total):
            progress.progress(current / total, text=f"Проверяю {current}/{total}...")

        with st.spinner("Оптимизация..."):
            results = optimizer.optimize(df, opt_symbol, metric=opt_metric[0],
                                        progress_callback=update_progress)
        progress.progress(1.0, text="Готово!")

        if not results:
            st.error("Нет результатов")
            st.stop()

        best = results[0]
        st.divider()

        # Результат — крупно
        color = "green" if best.net_profit > 0 else "red"
        st.markdown(f'''<div class="card" style="text-align:center; padding:28px;">
            <div style="font-size:15px; color:#888;">Лучшие параметры для {opt_symbol}</div>
            <div style="font-size:36px; font-weight:700; margin:12px 0;">
                {best.grid_count} уровней · {best.range_pct}%
            </div>
            <div class="card-value val-{color}">
                ${best.net_profit:.2f} (ROI {best.roi_pct:.2f}%)
            </div>
        </div>''', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Сделок", str(best.total_trades), "blue")
        with c2: metric_card("Win Rate", f"{best.win_rate*100:.1f}%", "amber")
        with c3: metric_card("Drawdown", f"{best.max_drawdown*100:.2f}%", "red")
        with c4: metric_card("Sharpe", f"{best.sharpe_ratio:.2f}", "blue")

        # Топ-5
        st.divider()
        st.subheader("Топ-5")
        medals = ["🥇", "🥈", "🥉", "4.", "5."]
        for i, r in enumerate(results[:5]):
            st.write(f"{medals[i]} **{r.grid_count} уровней, {r.range_pct}%** → "
                    f"ROI {r.roi_pct:.2f}%, прибыль ${r.net_profit:.2f}")

        with st.expander("📊 Heatmap"):
            try:
                heatmap_data = optimizer.results_to_heatmap(results, opt_metric[0])
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=[f"{x}%" for x in heatmap_data.columns],
                    y=[str(y) for y in heatmap_data.index],
                    colorscale="RdYlGn",
                    text=np.round(heatmap_data.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 11},
                ))
                fig.update_layout(xaxis_title="Диапазон %", yaxis_title="Уровней",
                                template="plotly_dark", height=400,
                                margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Не удалось построить heatmap: {e}")


# ═══════════════════════════════════════════════════════════════
# TAB: Live-прогноз
# ═══════════════════════════════════════════════════════════════
with tab_live:
    if not available_models:
        st.info("⬅️ Обучите модель на вкладке «Старт»")
        st.stop()

    st.markdown("Предсказание направления цены на ближайшие 6 часов")

    live_symbol = st.selectbox("Пара", available_models, key="live_sym")

    if st.button("🔮 Получить прогноз", type="primary", use_container_width=True, key="btn_live"):
        model_path = os.path.join(MODEL_DIR, f"{live_symbol.replace('/', '_')}_lstm.keras")
        if not os.path.exists(model_path):
            st.error("Модель не найдена!")
            st.stop()

        try:
            from ai.model import load_model
            from ai.feature_engineer import add_indicators, FEATURE_COLUMNS
            from ai.data_collector import update_data

            exchange = ccxt.binance({"enableRateLimit": True})

            with st.spinner("Загружаю данные с Binance..."):
                df = update_data(exchange, live_symbol, "1h")

            df = add_indicators(df).dropna().reset_index(drop=True)
            model, scaler = load_model(live_symbol)

            recent = df[FEATURE_COLUMNS].iloc[-48:].values
            scaled = scaler.transform(recent)
            X = np.expand_dims(scaled, axis=0)
            probs = model.predict(X, verbose=0)[0]

            directions = ["📉 Падение", "↔️ Боковик", "📈 Рост"]
            colors_map = ["red", "amber", "green"]
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            current_price = df["close"].iloc[-1]

            st.divider()

            # Результат — крупно
            st.markdown(f'''<div class="card" style="text-align:center; padding:28px;">
                <div style="font-size:15px; color:#888;">{live_symbol} · ${current_price:,.2f}</div>
                <div style="font-size:42px; margin:12px 0;">{directions[idx]}</div>
                <div class="card-value val-{colors_map[idx]}">
                    Уверенность: {confidence*100:.0f}%
                </div>
            </div>''', unsafe_allow_html=True)

            # Вероятности
            fig = go.Figure(go.Bar(
                x=["Падение", "Боковик", "Рост"],
                y=[p * 100 for p in probs],
                marker_color=["#ff5252", "#fbbf24", "#00e676"],
                text=[f"{p*100:.0f}%" for p in probs],
                textposition="outside",
                textfont=dict(size=16),
            ))
            fig.update_layout(template="plotly_dark", height=300,
                            margin=dict(l=0, r=0, t=10, b=0),
                            yaxis_range=[0, 100], yaxis_title="Вероятность %")
            st.plotly_chart(fig, use_container_width=True)

            # Рекомендация
            if idx == 2:
                st.success("**Рост** — можно расширить сетку вверх")
            elif idx == 0:
                st.warning("**Падение** — лучше сузить сетку или подождать")
            else:
                st.info("**Боковик** — идеальные условия для Grid бота")

            # Индикаторы
            with st.expander("📊 Технические индикаторы"):
                last = df.iloc[-1]
                i1, i2, i3, i4 = st.columns(4)
                i1.metric("RSI", f"{last['rsi']:.1f}")
                i2.metric("MACD", f"{last['macd']:.4f}")
                i3.metric("Волатильность", f"{last['atr_pct']*100:.2f}%")
                i4.metric("Позиция в канале", f"{last['bb_position']:.2f}")

        except ImportError as e:
            st.error(f"Не хватает зависимостей: {e}")
        except Exception as e:
            st.error(f"Ошибка: {e}")
