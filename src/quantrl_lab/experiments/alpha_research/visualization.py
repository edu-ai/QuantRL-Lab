import base64
import io
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from rich.console import Console

from .core import AlphaResult

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
        """Generate a premium HTML report with all visualizations."""

        # Generate plots
        fig_returns = self.plot_cumulative_returns(results)
        img_returns = self._fig_to_base64(fig_returns)
        plt.close(fig_returns)

        fig_dd = self.plot_drawdowns(results)
        img_dd = self._fig_to_base64(fig_dd)
        plt.close(fig_dd)

        fig_ic = self.plot_ic_analysis(results)
        img_ic = self._fig_to_base64(fig_ic)
        plt.close(fig_ic)

        fig_roll = self.plot_rolling_sharpe(results)
        img_roll = self._fig_to_base64(fig_roll)
        plt.close(fig_roll)

        fig_radar = self.plot_metrics_radar(results)
        img_radar = self._fig_to_base64(fig_radar)
        plt.close(fig_radar)

        # Compute summary statistics
        completed = [r for r in results if r.status == "completed"]

        if not completed:
            console.print("[red]No completed results to report.[/red]")
            return

        best_sharpe = max(r.metrics.get('sharpe_ratio', -999) for r in completed)
        best_result = next(r for r in completed if r.metrics.get('sharpe_ratio', -999) == best_sharpe)
        best_strategy = f"{best_result.job.strategy_name} ({best_result.job.indicator_name})"

        avg_return = np.mean([r.metrics.get('total_return', 0) for r in completed])
        avg_ic = np.mean([r.metrics.get('ic', 0) for r in completed])
        max_dd = min(r.metrics.get('max_drawdown', 0) for r in completed)

        # Build metrics table rows
        table_rows = ""
        for r in completed:
            m = r.metrics
            sharpe = m.get('sharpe_ratio', 0)
            sharpe_color = (
                self.ACCENT_POSITIVE if sharpe > 0 else self.ACCENT_NEGATIVE if sharpe < 0 else self.TEXT_MUTED
            )
            ret = m.get('total_return', 0)
            ret_color = self.ACCENT_POSITIVE if ret > 0 else self.ACCENT_NEGATIVE if ret < 0 else self.TEXT_MUTED

            table_rows += f"""
            <tr>
                <td><span class="strategy-badge">{r.job.indicator_name}</span></td>
                <td>{r.job.strategy_name}</td>
                <td style="color: {sharpe_color}; font-weight: 600;">{sharpe:.3f}</td>
                <td>{m.get('sortino_ratio', 0):.3f}</td>
                <td>{m.get('calmar_ratio', 0):.3f}</td>
                <td style="color: {ret_color}; font-weight: 600;">{ret:.2%}</td>
                <td style="color: {self.ACCENT_NEGATIVE};">{m.get('max_drawdown', 0):.2%}</td>
                <td>{m.get('win_rate', 0):.2%}</td>
                <td>{m.get('ic', 0):.4f}</td>
            </tr>
            """

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alpha Research Report | QuantRL-Lab</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: {self.BG_DARK};
            --bg-card: {self.BG_CARD};
            --bg-elevated: {self.BG_ELEVATED};
            --text-primary: {self.TEXT_PRIMARY};
            --text-secondary: {self.TEXT_SECONDARY};
            --text-muted: {self.TEXT_MUTED};
            --accent-positive: {self.ACCENT_POSITIVE};
            --accent-negative: {self.ACCENT_NEGATIVE};
            --accent-blue: {self.ACCENT_NEUTRAL};
            --cyan: {self.COLORS['cyan']};
        }}

        * {{ box-sizing: border-box; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }}

        /* Header */
        .report-header {{
            background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
            border-bottom: 1px solid var(--bg-elevated);
            padding: 2rem 0;
            margin-bottom: 2rem;
        }}

        .report-header h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(90deg, var(--cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .report-header .timestamp {{
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }}

        /* Metric Cards */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--bg-elevated);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }}

        .metric-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--cyan);
            line-height: 1.2;
        }}

        .metric-card .value.positive {{ color: var(--accent-positive); }}
        .metric-card .value.negative {{ color: var(--accent-negative); }}

        .metric-card .label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }}

        /* Winner Banner */
        .winner-banner {{
            background: linear-gradient(90deg, rgba(0, 212, 255, 0.1), rgba(63, 185, 80, 0.1));
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .winner-banner .icon {{
            font-size: 1.5rem;
        }}

        .winner-banner .content {{
            flex: 1;
        }}

        .winner-banner .title {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .winner-banner .strategy {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--cyan);
        }}

        /* Plot Cards */
        .plot-card {{
            background: var(--bg-card);
            border: 1px solid var(--bg-elevated);
            border-radius: 12px;
            margin-bottom: 1.5rem;
            overflow: hidden;
        }}

        .plot-card .card-header {{
            background: var(--bg-elevated);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--bg-dark);
            font-weight: 600;
            font-size: 0.95rem;
        }}

        .plot-card .card-body {{
            padding: 1rem;
        }}

        .plot-card img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
        }}

        /* Data Table */
        .table-container {{
            background: var(--bg-card);
            border: 1px solid var(--bg-elevated);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 2rem;
        }}

        .table-container .header {{
            background: var(--bg-elevated);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--bg-dark);
            font-weight: 600;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .data-table th {{
            background: var(--bg-elevated);
            padding: 0.875rem 1rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            border-bottom: 1px solid var(--bg-dark);
        }}

        .data-table td {{
            padding: 0.875rem 1rem;
            border-bottom: 1px solid var(--bg-elevated);
            font-size: 0.9rem;
        }}

        .data-table tbody tr:hover {{
            background: var(--bg-elevated);
        }}

        .strategy-badge {{
            display: inline-block;
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 212, 255, 0.1));
            color: var(--cyan);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }}

        /* Grid Layout */
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }}

        @media (max-width: 992px) {{
            .plot-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        /* Footer */
        .report-footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--bg-elevated);
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="report-header">
        <div class="container">
            <h1>📊 Alpha Research Report</h1>
            <div class="timestamp">Generated: {timestamp} | QuantRL-Lab</div>
        </div>
    </div>

    <div class="container">
        <!-- Metrics Summary -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="value">{len(completed)}</div>
                <div class="label">Strategies Tested</div>
            </div>
            <div class="metric-card">
                <div class="value positive">{best_sharpe:.2f}</div>
                <div class="label">Best Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="value {'positive' if avg_return > 0 else 'negative'}">{avg_return:.2%}</div>
                <div class="label">Avg Total Return</div>
            </div>
            <div class="metric-card">
                <div class="value">{avg_ic:.4f}</div>
                <div class="label">Avg IC</div>
            </div>
            <div class="metric-card">
                <div class="value negative">{max_dd:.2%}</div>
                <div class="label">Max Drawdown</div>
            </div>
        </div>

        <!-- Winner Banner -->
        <div class="winner-banner">
            <div class="icon">🏆</div>
            <div class="content">
                <div class="title">Top Performing Strategy</div>
                <div class="strategy">{best_strategy}</div>
            </div>
        </div>

        <!-- Main Performance Chart -->
        <div class="plot-card">
            <div class="card-header">📈 Strategy Performance</div>
            <div class="card-body">
                <img src="data:image/png;base64,{img_returns}" alt="Cumulative Returns">
            </div>
        </div>

        <!-- Grid: Drawdown and Rolling Sharpe -->
        <div class="plot-grid">
            <div class="plot-card">
                <div class="card-header">📉 Drawdown Analysis</div>
                <div class="card-body">
                    <img src="data:image/png;base64,{img_dd}" alt="Drawdowns">
                </div>
            </div>
            <div class="plot-card">
                <div class="card-header">📊 Rolling Sharpe Ratio</div>
                <div class="card-body">
                    <img src="data:image/png;base64,{img_roll}" alt="Rolling Sharpe">
                </div>
            </div>
        </div>

        <!-- Grid: IC and Radar -->
        <div class="plot-grid">
            <div class="plot-card">
                <div class="card-header">🎯 Information Coefficient</div>
                <div class="card-body">
                    <img src="data:image/png;base64,{img_ic}" alt="IC Analysis">
                </div>
            </div>
            <div class="plot-card">
                <div class="card-header">🕸️ Strategy Comparison</div>
                <div class="card-body">
                    <img src="data:image/png;base64,{img_radar}" alt="Radar Chart">
                </div>
            </div>
        </div>

        <!-- Detailed Metrics Table -->
        <div class="table-container">
            <div class="header">📋 Detailed Metrics</div>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Indicator</th>
                        <th>Strategy</th>
                        <th>Sharpe</th>
                        <th>Sortino</th>
                        <th>Calmar</th>
                        <th>Return</th>
                        <th>Max DD</th>
                        <th>Win Rate</th>
                        <th>IC</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </div>

    <div class="report-footer">
        Built with QuantRL-Lab | Alpha Research Module
    </div>
</body>
</html>
"""

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
