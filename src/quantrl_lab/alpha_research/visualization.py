import base64
import io
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from rich.console import Console

from .models import AlphaResult

console = Console()


class AlphaVisualizer:
    """Visualization tools for alpha research results with professional
    financial aesthetics."""

    # Premium color palette - inspired by Bloomberg/Reuters terminals
    COLORS = {
        "cyan": "#00D4FF",
        "orange": "#FF8C00",
        "green": "#00FF88",
        "red": "#FF3366",
        "purple": "#A855F7",
        "pink": "#FF6B9D",
        "yellow": "#FFD93D",
        "teal": "#14B8A6",
    }

    PALETTE = list(COLORS.values())

    # Background colors
    BG_DARK = "#0D1117"
    BG_CARD = "#161B22"
    BG_ELEVATED = "#21262D"

    # Text colors
    TEXT_PRIMARY = "#F0F6FC"
    TEXT_SECONDARY = "#8B949E"
    TEXT_MUTED = "#484F58"

    # Accent colors
    ACCENT_POSITIVE = "#3FB950"
    ACCENT_NEGATIVE = "#F85149"
    ACCENT_NEUTRAL = "#58A6FF"

    def __init__(self):
        """Initialize the visualizer with a premium dark theme."""
        plt.style.use("dark_background")

        plt.rcParams.update(
            {
                # Figure
                "figure.facecolor": self.BG_DARK,
                "figure.edgecolor": self.BG_DARK,
                "figure.dpi": 120,
                # Axes
                "axes.facecolor": self.BG_CARD,
                "axes.edgecolor": self.BG_ELEVATED,
                "axes.labelcolor": self.TEXT_SECONDARY,
                "axes.titlecolor": self.TEXT_PRIMARY,
                "axes.grid": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.linewidth": 0.8,
                # Grid
                "grid.color": self.BG_ELEVATED,
                "grid.linestyle": "-",
                "grid.linewidth": 0.5,
                "grid.alpha": 0.5,
                # Ticks
                "xtick.color": self.TEXT_SECONDARY,
                "ytick.color": self.TEXT_SECONDARY,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                # Text
                "text.color": self.TEXT_PRIMARY,
                "font.family": "sans-serif",
                "font.size": 10,
                # Title
                "axes.titlesize": 13,
                "axes.titleweight": "bold",
                "axes.titlelocation": "left",
                "axes.titlepad": 15,
                # Legend
                "legend.facecolor": self.BG_ELEVATED,
                "legend.edgecolor": self.TEXT_MUTED,
                "legend.fontsize": 9,
                "legend.framealpha": 0.9,
                # Lines
                "lines.linewidth": 2.0,
                "lines.antialiased": True,
            }
        )

        sns.set_palette(sns.color_palette(self.PALETTE))

    def plot_cumulative_returns(
        self, results: List[AlphaResult], title: str = "Strategy Performance", figsize: tuple = (14, 7)
    ) -> plt.Figure:
        """Plot cumulative returns with gradient fills and
        annotations."""
        fig, ax = plt.subplots(figsize=figsize)

        completed = [
            (r, r.equity_curve / r.equity_curve.iloc[0])
            for r in results
            if r.status == "completed" and r.equity_curve is not None
        ]

        if not completed:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14, color=self.TEXT_MUTED)
            return fig

        # Sort by final return for better legend ordering
        completed.sort(key=lambda x: x[1].iloc[-1], reverse=True)

        for i, (result, equity) in enumerate(completed):
            color = self.PALETTE[i % len(self.PALETTE)]
            final_ret = (equity.iloc[-1] - 1) * 100
            label = f"{result.job.strategy_name} ({result.job.indicator_name})"

            # Main line with glow effect
            ax.plot(
                equity.index,
                equity,
                color=color,
                linewidth=2.5,
                alpha=0.9,
                label=f"{label}  {final_ret:+.1f}%",
                zorder=10,
            )

            # Gradient fill
            ax.fill_between(equity.index, 1.0, equity, color=color, alpha=0.08, zorder=5)

            # End marker
            ax.scatter(
                [equity.index[-1]], [equity.iloc[-1]], color=color, s=60, edgecolor='white', linewidth=1.5, zorder=15
            )

        # Breakeven line
        ax.axhline(1.0, color=self.TEXT_MUTED, linestyle='--', linewidth=1, alpha=0.6, zorder=1)
        ax.text(
            ax.get_xlim()[0], 1.0, ' Breakeven', va='bottom', ha='left', fontsize=8, color=self.TEXT_MUTED, alpha=0.8
        )

        ax.set_title(title, fontsize=14, fontweight='bold', color=self.TEXT_PRIMARY)
        ax.set_ylabel("Growth of $1", fontsize=10, color=self.TEXT_SECONDARY)
        ax.set_xlabel("")

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.2f'))

        # Legend outside plot
        ax.legend(loc='upper left', frameon=True, fancybox=True)

        plt.tight_layout()
        return fig

    def plot_drawdowns(
        self, results: List[AlphaResult], title: str = "Drawdown Analysis", figsize: tuple = (14, 6)
    ) -> plt.Figure:
        """Plot drawdowns as filled areas with max drawdown
        annotations."""
        fig, ax = plt.subplots(figsize=figsize)

        for i, result in enumerate(results):
            if result.status != "completed" or result.equity_curve is None:
                continue

            equity = result.equity_curve
            running_max = equity.cummax()
            drawdown = (equity / running_max) - 1

            color = self.PALETTE[i % len(self.PALETTE)]
            label = f"{result.job.strategy_name} ({result.job.indicator_name})"

            # Fill area
            ax.fill_between(drawdown.index, drawdown * 100, 0, color=color, alpha=0.25, label=label)
            ax.plot(drawdown.index, drawdown * 100, color=color, linewidth=1.2, alpha=0.9)

            # Mark max drawdown point
            max_dd_idx = drawdown.idxmin()
            max_dd_val = drawdown.min() * 100
            ax.scatter(
                [max_dd_idx], [max_dd_val], color=color, s=40, marker='v', edgecolor='white', linewidth=1, zorder=10
            )

        ax.axhline(0, color=self.TEXT_MUTED, linestyle='-', linewidth=0.8, alpha=0.5)

        ax.set_title(title, fontsize=14, fontweight='bold', color=self.TEXT_PRIMARY)
        ax.set_ylabel("Drawdown (%)", fontsize=10, color=self.TEXT_SECONDARY)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

        ax.legend(loc='lower left', frameon=True, fancybox=True)

        plt.tight_layout()
        return fig

    def plot_ic_analysis(
        self, results: List[AlphaResult], title: str = "Information Coefficient Analysis", figsize: tuple = (12, 7)
    ) -> plt.Figure:
        """Plot IC as horizontal bars with value annotations."""
        fig, ax = plt.subplots(figsize=figsize)

        data = []
        for result in results:
            if result.status == "completed":
                data.append(
                    {
                        "name": f"{result.job.indicator_name} | {result.job.strategy_name}",
                        "ic": result.metrics.get("ic", 0),
                        "sharpe": result.metrics.get("sharpe_ratio", 0),
                    }
                )

        if not data:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14, color=self.TEXT_MUTED)
            return fig

        # Sort by IC
        data.sort(key=lambda x: x["ic"], reverse=True)

        names = [d["name"] for d in data]
        ics = [d["ic"] for d in data]

        y_pos = np.arange(len(names))

        # Create gradient colors based on IC value
        colors = [self.ACCENT_POSITIVE if ic >= 0 else self.ACCENT_NEGATIVE for ic in ics]

        bars = ax.barh(y_pos, ics, color=colors, alpha=0.8, height=0.6, edgecolor='none')

        # Add value labels
        for i, (bar, ic) in enumerate(zip(bars, ics)):
            width = bar.get_width()
            offset = 0.003 if width >= 0 else -0.003
            ha = 'left' if width >= 0 else 'right'
            ax.text(
                width + offset,
                bar.get_y() + bar.get_height() / 2,
                f'{ic:.4f}',
                ha=ha,
                va='center',
                fontsize=9,
                fontweight='bold',
                color=self.TEXT_PRIMARY,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()

        ax.axvline(0, color=self.TEXT_MUTED, linestyle='-', linewidth=1, alpha=0.5)

        ax.set_title(title, fontsize=14, fontweight='bold', color=self.TEXT_PRIMARY)
        ax.set_xlabel("Information Coefficient", fontsize=10, color=self.TEXT_SECONDARY)

        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_rolling_sharpe(
        self,
        results: List[AlphaResult],
        window: int = 60,
        title: str = "Rolling Sharpe Ratio",
        figsize: tuple = (14, 6),
    ) -> plt.Figure:
        """Plot rolling Sharpe ratio with trend highlighting."""
        fig, ax = plt.subplots(figsize=figsize)

        for i, result in enumerate(results):
            if result.status != "completed" or result.equity_curve is None:
                continue

            returns = result.equity_curve.pct_change().dropna()
            rolling_std = returns.rolling(window).std()
            rolling_mean = returns.rolling(window).mean()
            rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)

            color = self.PALETTE[i % len(self.PALETTE)]
            label = f"{result.job.strategy_name} ({result.job.indicator_name})"

            ax.plot(rolling_sharpe.index, rolling_sharpe, color=color, linewidth=1.8, alpha=0.9, label=label)

            # Add light fill for positive/negative regions
            ax.fill_between(
                rolling_sharpe.index,
                rolling_sharpe,
                0,
                where=(rolling_sharpe >= 0),
                color=self.ACCENT_POSITIVE,
                alpha=0.05,
            )
            ax.fill_between(
                rolling_sharpe.index,
                rolling_sharpe,
                0,
                where=(rolling_sharpe < 0),
                color=self.ACCENT_NEGATIVE,
                alpha=0.05,
            )

        ax.axhline(0, color=self.TEXT_MUTED, linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(1, color=self.ACCENT_POSITIVE, linestyle=':', linewidth=0.8, alpha=0.4)
        ax.axhline(-1, color=self.ACCENT_NEGATIVE, linestyle=':', linewidth=0.8, alpha=0.4)

        ax.set_title(f"{title} ({window}-day rolling)", fontsize=14, fontweight='bold', color=self.TEXT_PRIMARY)
        ax.set_ylabel("Sharpe Ratio", fontsize=10, color=self.TEXT_SECONDARY)
        ax.set_xlabel("")

        ax.legend(loc='upper left', frameon=True, fancybox=True)

        plt.tight_layout()
        return fig

    def plot_metrics_radar(
        self, results: List[AlphaResult], title: str = "Strategy Comparison", figsize: tuple = (10, 10)
    ) -> plt.Figure:
        """Plot radar/spider chart comparing key metrics across
        strategies."""
        completed = [r for r in results if r.status == "completed"]

        if not completed:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig

        # Metrics to compare (normalized 0-1)
        categories = ['Sharpe', 'Sortino', 'Win Rate', 'IC', 'Return']
        n_cats = len(categories)

        # Calculate angles for radar
        angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        ax.set_facecolor(self.BG_CARD)
        fig.patch.set_facecolor(self.BG_DARK)

        # Get min/max for normalization
        all_metrics = {cat: [] for cat in categories}
        for r in completed:
            m = r.metrics
            all_metrics['Sharpe'].append(m.get('sharpe_ratio', 0))
            all_metrics['Sortino'].append(m.get('sortino_ratio', 0))
            all_metrics['Win Rate'].append(m.get('win_rate', 0))
            all_metrics['IC'].append(m.get('ic', 0))
            all_metrics['Return'].append(m.get('total_return', 0))

        def normalize(val, vals):
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return 0.5
            return (val - min_v) / (max_v - min_v)

        for i, result in enumerate(completed[:6]):  # Limit to 6 strategies
            m = result.metrics
            values = [
                normalize(m.get('sharpe_ratio', 0), all_metrics['Sharpe']),
                normalize(m.get('sortino_ratio', 0), all_metrics['Sortino']),
                normalize(m.get('win_rate', 0), all_metrics['Win Rate']),
                normalize(m.get('ic', 0), all_metrics['IC']),
                normalize(m.get('total_return', 0), all_metrics['Return']),
            ]
            values += values[:1]

            color = self.PALETTE[i % len(self.PALETTE)]
            label = f"{result.job.strategy_name} ({result.job.indicator_name})"

            ax.plot(angles, values, color=color, linewidth=2, alpha=0.9, label=label)
            ax.fill(angles, values, color=color, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, color=self.TEXT_PRIMARY)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color=self.TEXT_MUTED)
        ax.grid(color=self.BG_ELEVATED, linestyle='-', linewidth=0.5)

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), frameon=True, fancybox=True)

        plt.title(title, fontsize=14, fontweight='bold', color=self.TEXT_PRIMARY, pad=20)
        plt.tight_layout()
        return fig

    def generate_html_report(self, results: List[AlphaResult], output_path: str) -> None:
        """Generate an interactive HTML report using Plotly charts."""
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except ImportError:
            raise ImportError("plotly is required for HTML report generation. Install with: uv sync --extra viz")

        completed = [r for r in results if r.status == "completed"]
        if not completed:
            console.print("[red]No completed results to report.[/red]")
            return

        # ── Summary stats from the top performer by Sharpe ──────────────────
        best = max(completed, key=lambda r: r.metrics.get('sharpe_ratio', -999))
        best_m = best.metrics
        best_label = f"{best.job.indicator_name} | {best.job.strategy_name}"

        _plotly_theme = dict(
            paper_bgcolor=self.BG_DARK,
            plot_bgcolor=self.BG_CARD,
            font=dict(family="Inter, sans-serif", color=self.TEXT_PRIMARY),
            xaxis=dict(gridcolor=self.BG_ELEVATED, linecolor=self.BG_ELEVATED, zeroline=False),
            yaxis=dict(gridcolor=self.BG_ELEVATED, linecolor=self.BG_ELEVATED, zeroline=False),
            margin=dict(l=60, r=30, t=50, b=50),
            legend=dict(bgcolor=self.BG_ELEVATED, bordercolor=self.TEXT_MUTED, borderwidth=1),
        )

        # ── Chart 1: Cumulative returns (interactive, toggleable) ────────────
        fig_ret = go.Figure()
        sorted_completed = sorted(
            completed,
            key=lambda r: r.equity_curve.iloc[-1] / r.equity_curve.iloc[0] if r.equity_curve is not None else 0,
            reverse=True,
        )
        for i, r in enumerate(sorted_completed):
            if r.equity_curve is None:
                continue
            equity = r.equity_curve / r.equity_curve.iloc[0]
            color = self.PALETTE[i % len(self.PALETTE)]
            label = f"{r.job.indicator_name} | {r.job.strategy_name}"
            final_ret = (equity.iloc[-1] - 1) * 100
            fig_ret.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity,
                    name=f"{label}  {final_ret:+.1f}%",
                    line=dict(color=color, width=2),
                    fill='tonexty' if i == 0 else None,
                    fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Growth: $%{y:.3f}<extra>" + label + "</extra>",
                )
            )
        fig_ret.add_hline(y=1.0, line_dash="dash", line_color=self.TEXT_MUTED, line_width=1, opacity=0.5)
        fig_ret.update_layout(**_plotly_theme, title="Strategy Performance (Growth of $1)", yaxis_tickprefix="$")
        chart_returns = pio.to_html(fig_ret, full_html=False, include_plotlyjs=False)

        # ── Chart 2: Drawdowns ───────────────────────────────────────────────
        fig_dd = go.Figure()
        for i, r in enumerate(completed):
            if r.equity_curve is None:
                continue
            dd = (r.equity_curve / r.equity_curve.cummax() - 1) * 100
            color = self.PALETTE[i % len(self.PALETTE)]
            label = f"{r.job.indicator_name} | {r.job.strategy_name}"
            fig_dd.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd,
                    name=label,
                    line=dict(color=color, width=1.5),
                    fill='tozeroy',
                    fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra>" + label + "</extra>",
                )
            )
        fig_dd.update_layout(**_plotly_theme, title="Drawdown Analysis", yaxis_ticksuffix="%")
        chart_dd = pio.to_html(fig_dd, full_html=False, include_plotlyjs=False)

        # ── Chart 3: Rolling Sharpe ──────────────────────────────────────────
        fig_rs = go.Figure()
        window = 60
        for i, r in enumerate(completed):
            if r.equity_curve is None:
                continue
            rets = r.equity_curve.pct_change().dropna()
            roll_sharpe = (rets.rolling(window).mean() / rets.rolling(window).std().replace(0, np.nan)) * np.sqrt(252)
            color = self.PALETTE[i % len(self.PALETTE)]
            label = f"{r.job.indicator_name} | {r.job.strategy_name}"
            fig_rs.add_trace(
                go.Scatter(
                    x=roll_sharpe.index,
                    y=roll_sharpe,
                    name=label,
                    line=dict(color=color, width=1.8),
                    hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra>" + label + "</extra>",
                )
            )
        fig_rs.add_hline(y=0, line_dash="dash", line_color=self.TEXT_MUTED, line_width=1, opacity=0.5)
        fig_rs.add_hline(y=1, line_dash="dot", line_color=self.ACCENT_POSITIVE, line_width=0.8, opacity=0.4)
        fig_rs.add_hline(y=-1, line_dash="dot", line_color=self.ACCENT_NEGATIVE, line_width=0.8, opacity=0.4)
        fig_rs.update_layout(**_plotly_theme, title=f"Rolling Sharpe Ratio ({window}-day)")
        chart_rs = pio.to_html(fig_rs, full_html=False, include_plotlyjs=False)

        # ── Chart 4: IC horizontal bar ───────────────────────────────────────
        ic_data = sorted(
            [
                {"label": f"{r.job.indicator_name} | {r.job.strategy_name}", "ic": r.metrics.get("ic", 0)}
                for r in completed
            ],
            key=lambda x: x["ic"],
        )
        ic_colors = [self.ACCENT_POSITIVE if d["ic"] >= 0 else self.ACCENT_NEGATIVE for d in ic_data]
        fig_ic = go.Figure(
            go.Bar(
                x=[d["ic"] for d in ic_data],
                y=[d["label"] for d in ic_data],
                orientation='h',
                marker_color=ic_colors,
                text=[f'{d["ic"]:.4f}' for d in ic_data],
                textposition='outside',
                hovertemplate="%{y}<br>IC: %{x:.4f}<extra></extra>",
            )
        )
        fig_ic.add_vline(x=0, line_color=self.TEXT_MUTED, line_width=1, opacity=0.5)
        fig_ic.update_layout(**_plotly_theme, title="Information Coefficient", xaxis_title="IC")
        chart_ic = pio.to_html(fig_ic, full_html=False, include_plotlyjs=False)

        # ── Chart 5: IC vs Sharpe scatter (replaces radar) ───────────────────
        fig_sc = go.Figure()
        for i, r in enumerate(completed):
            m = r.metrics
            ic_val = m.get('ic', 0)
            sharpe_val = m.get('sharpe_ratio', 0)
            label = f"{r.job.indicator_name} | {r.job.strategy_name}"
            color = self.PALETTE[i % len(self.PALETTE)]
            fig_sc.add_trace(
                go.Scatter(
                    x=[ic_val],
                    y=[sharpe_val],
                    mode='markers+text',
                    name=label,
                    marker=dict(color=color, size=12, line=dict(color='white', width=1)),
                    text=[r.job.indicator_name],
                    textposition='top center',
                    textfont=dict(size=9),
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "IC: %{x:.4f}<br>"
                        "Sharpe: %{y:.3f}<br>"
                        f"Return: {m.get('total_return', 0):.2%}<br>"
                        f"Win Rate: {m.get('win_rate', 0):.2%}"
                        "<extra></extra>"
                    ),
                )
            )
        fig_sc.add_vline(x=0, line_dash="dash", line_color=self.TEXT_MUTED, line_width=1, opacity=0.4)
        fig_sc.add_hline(y=0, line_dash="dash", line_color=self.TEXT_MUTED, line_width=1, opacity=0.4)
        fig_sc.update_layout(
            **_plotly_theme,
            title="Signal Quality vs Backtest Performance",
            xaxis_title="Information Coefficient (IC)  →  higher = more predictive",
            yaxis_title="Sharpe Ratio  →  higher = better risk-adjusted return",
            showlegend=False,
        )
        chart_scatter = pio.to_html(fig_sc, full_html=False, include_plotlyjs=False)

        # ── Metrics table rows (sorted by Sharpe desc) ───────────────────────
        table_rows = ""
        for r in sorted(completed, key=lambda x: x.metrics.get('sharpe_ratio', -999), reverse=True):
            m = r.metrics
            sharpe = m.get('sharpe_ratio', 0)
            ret = m.get('total_return', 0)
            sharpe_color = self.ACCENT_POSITIVE if sharpe > 0 else self.ACCENT_NEGATIVE
            ret_color = self.ACCENT_POSITIVE if ret > 0 else self.ACCENT_NEGATIVE
            table_rows += f"""
            <tr>
                <td><span class="badge">{r.job.indicator_name}</span></td>
                <td>{r.job.strategy_name}</td>
                <td style="color:{sharpe_color};font-weight:600" data-val="{sharpe:.4f}">{sharpe:.3f}</td>
                <td data-val="{m.get('sortino_ratio',0):.4f}">{m.get('sortino_ratio',0):.3f}</td>
                <td data-val="{m.get('calmar_ratio',0):.4f}">{m.get('calmar_ratio',0):.3f}</td>
                <td style="color:{ret_color};font-weight:600" data-val="{ret:.6f}">{ret:.2%}</td>
                <td style="color:{self.ACCENT_NEGATIVE}"
                    data-val="{m.get('max_drawdown',0):.6f}">{m.get('max_drawdown',0):.2%}</td>
                <td data-val="{m.get('win_rate',0):.6f}">{m.get('win_rate',0):.2%}</td>
                <td data-val="{m.get('ic',0):.6f}">{m.get('ic',0):.4f}</td>
            </tr>"""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alpha Research Report | QuantRL-Lab</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg:       {self.BG_DARK};
            --card:     {self.BG_CARD};
            --elevated: {self.BG_ELEVATED};
            --t1:       {self.TEXT_PRIMARY};
            --t2:       {self.TEXT_SECONDARY};
            --tm:       {self.TEXT_MUTED};
            --pos:      {self.ACCENT_POSITIVE};
            --neg:      {self.ACCENT_NEGATIVE};
            --blue:     {self.ACCENT_NEUTRAL};
            --cyan:     {self.COLORS['cyan']};
        }}
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--t1); line-height: 1.6; }}
        a {{ color: var(--cyan); }}

        /* ── Header ─────────────────────────────────────────────────────── */
        header {{
            background: linear-gradient(135deg, var(--card), var(--elevated));
            border-bottom: 1px solid var(--elevated);
            padding: 1.75rem 2rem;
        }}
        header h1 {{
            font-size: 1.6rem; font-weight: 700;
            background: linear-gradient(90deg, var(--cyan), var(--blue));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }}
        header .meta {{ color: var(--tm); font-size: 0.82rem; margin-top: 0.3rem; }}

        /* ── Layout ─────────────────────────────────────────────────────── */
        main {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
        @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

        /* ── KPI cards ───────────────────────────────────────────────────── */
        .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem; margin-bottom: 1.5rem; }}
        .kpi {{
            background: var(--card); border: 1px solid var(--elevated); border-radius: 10px;
            padding: 1.25rem; text-align: center;
            transition: transform .15s, box-shadow .15s;
        }}
        .kpi:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,.35); }}
        .kpi .v {{ font-size: 1.75rem; font-weight: 700; color: var(--cyan); line-height: 1.1; }}
        .kpi .v.pos {{ color: var(--pos); }} .kpi .v.neg {{ color: var(--neg); }}
        .kpi .l {{ font-size: 0.7rem; text-transform: uppercase;
            letter-spacing: 1.2px; color: var(--tm); margin-top: .4rem; }}

        /* ── Winner banner ───────────────────────────────────────────────── */
        .winner {{
            background: linear-gradient(90deg, rgba(0,212,255,.08), rgba(63,185,80,.08));
            border: 1px solid rgba(0,212,255,.25); border-radius: 10px;
            padding: 1rem 1.5rem; margin-bottom: 1.5rem;
            display: flex; align-items: center; gap: 1rem;
        }}
        .winner .icon {{ font-size: 1.4rem; }}
        .winner .lbl {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: var(--tm); }}
        .winner .name {{ font-size: 1rem; font-weight: 600; color: var(--cyan); }}

        /* ── Plot cards ──────────────────────────────────────────────────── */
        .card {{
            background: var(--card); border: 1px solid var(--elevated);
            border-radius: 10px; overflow: hidden; margin-bottom: 1.5rem;
        }}
        .card-hdr {{
            background: var(--elevated); padding: .75rem 1.25rem;
            border-bottom: 1px solid var(--bg); font-weight: 600; font-size: .9rem;
        }}
        .card-body {{ padding: .5rem; }}
        .card-body .plotly-graph-div {{ border-radius: 6px; }}

        /* ── Table ───────────────────────────────────────────────────────── */
        .tbl-wrap {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: .875rem; }}
        thead th {{
            background: var(--elevated); padding: .75rem 1rem;
            text-align: left; font-size: .75rem; text-transform: uppercase;
            letter-spacing: .5px; color: var(--tm); border-bottom: 1px solid var(--bg);
            cursor: pointer; user-select: none; white-space: nowrap;
        }}
        thead th:hover {{ color: var(--t1); }}
        thead th.asc::after  {{ content: ' ↑'; color: var(--cyan); }}
        thead th.desc::after {{ content: ' ↓'; color: var(--cyan); }}
        tbody td {{ padding: .75rem 1rem; border-bottom: 1px solid var(--elevated); }}
        tbody tr:hover {{ background: var(--elevated); }}
        .badge {{
            display: inline-block;
            background: linear-gradient(135deg, rgba(0,212,255,.2), rgba(0,212,255,.08));
            color: var(--cyan); padding: .2rem .65rem; border-radius: 20px;
            font-size: .78rem; font-weight: 500;
        }}

        /* ── Footer ──────────────────────────────────────────────────────── */
        footer {{
            text-align: center; padding: 1.5rem 2rem; color: var(--tm);
            font-size: .8rem; border-top: 1px solid var(--elevated); margin-top: 1rem;
        }}
    </style>
</head>
<body>
<header>
    <h1>Alpha Research Report</h1>
    <div class="meta">Generated: {timestamp} &nbsp;·&nbsp;
        QuantRL-Lab &nbsp;·&nbsp; {len(completed)} strategies evaluated</div>
</header>

<main>
    <!-- KPI cards — top performer stats, not averages -->
    <div class="kpi-row">
        <div class="kpi"><div class="v">{len(completed)}</div><div class="l">Strategies Tested</div></div>
        <div class="kpi">
            <div class="v pos">{best_m.get('sharpe_ratio',0):.2f}</div>
            <div class="l">Best Sharpe</div></div>
        <div class="kpi">
            <div class="v {'pos' if best_m.get('total_return',0)>0 else 'neg'}">
                {best_m.get('total_return',0):.2%}</div>
            <div class="l">Best Return</div></div>
        <div class="kpi">
            <div class="v pos">{best_m.get('ic',0):.4f}</div>
            <div class="l">Best IC</div></div>
        <div class="kpi">
            <div class="v pos">{best_m.get('win_rate',0):.2%}</div>
            <div class="l">Best Win Rate</div></div>
        <div class="kpi">
            <div class="v neg">{best_m.get('max_drawdown',0):.2%}</div>
            <div class="l">Best Max DD</div></div>
    </div>

    <!-- Winner banner -->
    <div class="winner">
        <div class="icon">🏆</div>
        <div>
            <div class="lbl">Top Performer by Sharpe</div>
            <div class="name">{best_label}</div>
        </div>
    </div>

    <!-- Cumulative returns (full width) -->
    <div class="card">
        <div class="card-hdr">📈 Strategy Performance — click legend to toggle</div>
        <div class="card-body">{chart_returns}</div>
    </div>

    <!-- Drawdown + Rolling Sharpe -->
    <div class="grid-2">
        <div class="card">
            <div class="card-hdr">📉 Drawdown Analysis</div>
            <div class="card-body">{chart_dd}</div>
        </div>
        <div class="card">
            <div class="card-hdr">📊 Rolling Sharpe ({window}-day)</div>
            <div class="card-body">{chart_rs}</div>
        </div>
    </div>

    <!-- IC bar + IC vs Sharpe scatter -->
    <div class="grid-2">
        <div class="card">
            <div class="card-hdr">🎯 Information Coefficient</div>
            <div class="card-body">{chart_ic}</div>
        </div>
        <div class="card">
            <div class="card-hdr">🔬 Signal Quality vs Backtest Performance</div>
            <div class="card-body">{chart_scatter}</div>
        </div>
    </div>

    <!-- Detailed metrics table (sortable) -->
    <div class="card">
        <div class="card-hdr">📋 Detailed Metrics — click any column header to sort</div>
        <div class="card-body tbl-wrap">
            <table id="metrics-table">
                <thead>
                    <tr>
                        <th>Indicator</th><th>Strategy</th>
                        <th>Sharpe</th><th>Sortino</th><th>Calmar</th>
                        <th>Return</th><th>Max DD</th><th>Win Rate</th><th>IC</th>
                    </tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
    </div>
</main>

<footer>Built with QuantRL-Lab · Alpha Research Module</footer>

<script>
// ── Sortable table ────────────────────────────────────────────────────────────
(function () {{
    const table = document.getElementById('metrics-table');
    const headers = table.querySelectorAll('thead th');
    let sortCol = -1, sortAsc = true;

    headers.forEach((th, col) => {{
        th.addEventListener('click', () => {{
            if (sortCol === col) {{ sortAsc = !sortAsc; }}
            else {{ sortCol = col; sortAsc = true; }}

            headers.forEach(h => h.classList.remove('asc', 'desc'));
            th.classList.add(sortAsc ? 'asc' : 'desc');

            const tbody = table.querySelector('tbody');
            const rows  = Array.from(tbody.querySelectorAll('tr'));

            rows.sort((a, b) => {{
                const aCell = a.querySelectorAll('td')[col];
                const bCell = b.querySelectorAll('td')[col];

                // Prefer data-val for numeric columns, fallback to textContent
                const aRaw = aCell.dataset.val ?? aCell.textContent.trim();
                const bRaw = bCell.dataset.val ?? bCell.textContent.trim();

                const aNum = parseFloat(aRaw);
                const bNum = parseFloat(bRaw);
                const numeric = !isNaN(aNum) && !isNaN(bNum);

                const cmp = numeric ? aNum - bNum : aRaw.localeCompare(bRaw);
                return sortAsc ? cmp : -cmp;
            }});

            rows.forEach(r => tbody.appendChild(r));
        }});
    }});
}})();
</script>
</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html_content)

        console.print(f"[green]✓ Report saved to {output_path}[/green]")

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string with high
        quality."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=self.BG_DARK, edgecolor='none')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
