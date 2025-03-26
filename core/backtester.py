import os
import pandas as pd
import numpy as np
from typing import List
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


class Backtester:
    """
    Клас, що приймає стратегії (StrategyBase) та запускає їхній бектест.
    Збирає метрики, будує графіки, генерує HTML-звіти.
    """

    def __init__(self, strategies: List, results_path: str = "./results"):
        """
        :param strategies: список екземплярів класів (наслідуваних від StrategyBase)
        :param results_path: директорія для збереження результатів (csv, графіки, html)
        """
        self.strategies = strategies
        self.results_path = results_path
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(os.path.join(self.results_path, "screenshots"), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, "html"), exist_ok=True)

    def run_all(self):
        """
        Запускає бектест для кожної стратегії, зберігає метрики в CSV і графіки в PNG/HTML.
        """
        all_metrics = []

        for strat in self.strategies:
            strat_name = strat.__class__.__name__
            print(f"[Backtester] Running backtest for {strat_name} ...")

            pf = strat.run_backtest()
            metrics = strat.get_metrics()
            metrics["strategy"] = strat_name
            all_metrics.append(metrics)

            # Equity curve (mean по всіх символах)
            mean_nav = pf.value().mean(axis=1)
            fig_curve = px.line(mean_nav, title=f"Equity Curve - {strat_name}")
            fig_curve.update_layout(xaxis_title="Time", yaxis_title="Mean NAV")

            fig_curve_path = os.path.join(self.results_path, "screenshots", f"{strat_name}_equity.png")
            fig_curve.write_image(fig_curve_path)

            # Heatmap по total_return кожного символу
            ret_series = pf.total_return()
            ret_df = ret_series.to_frame(name="total_return").reset_index()

            if 'symbol' not in ret_df.columns:
                ret_df.rename(columns={'level_1': 'symbol'}, inplace=True)

            fig_heat = px.density_heatmap(
                ret_df,
                x="symbol",
                y="total_return",
                title=f"Heatmap - {strat_name}",
                color_continuous_scale="Viridis"
            )

            fig_heat_path = os.path.join(self.results_path, "screenshots", f"{strat_name}_heatmap.png")
            fig_heat.write_image(fig_heat_path)

            # Генеруємо HTML-звіт
            html_output_dir = os.path.join(self.results_path, "html")
            self.generate_html_report(strat_name, [fig_curve, fig_heat], html_output_dir)

        # Зберігаємо сукупний CSV з метриками
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(os.path.join(self.results_path, "metrics.csv"), index=False)
        print("[Backtester] All metrics saved to metrics.csv")

    def generate_html_report(self, strategy_name: str, figures: List, output_path: str):
        """
        Генерує інтерактивний .html звіт з переданих фігур Plotly.
        :param strategy_name: Назва стратегії
        :param figures: Список Plotly figure (equity, heatmap, тощо)
        :param output_path: Куди зберігати .html звіт
        """
        os.makedirs(output_path, exist_ok=True)
        html_parts = [pio.to_html(fig, full_html=False, include_plotlyjs='cdn') for fig in figures]

        full_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{strategy_name} – Звіт</title>
        </head>
        <body>
            <h1>{strategy_name} – Звіт</h1>
            {"<hr>".join(html_parts)}
        </body>
        </html>
        """

        with open(os.path.join(output_path, f"{strategy_name}_report.html"), "w", encoding="utf-8") as f:
            f.write(full_html)
