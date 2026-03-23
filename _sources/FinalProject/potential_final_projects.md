# Potential Final Project Ideas

Details are coming soon. In the meantime, here are some project categories to get you thinking.

## Auto-Research for Financial Forecasting

[Autoresearch](https://github.com/karpathy/autoresearch) is a framework by Andrej Karpathy that automates the ML experimentation loop. An AI agent reads your training code, forms hypotheses for improvement, modifies the code, runs fixed-length training experiments, and evaluates whether the changes helped — all autonomously. It can run ~100 experiments overnight on a single GPU.

[![Karpathy's autoresearch announcement](../assets/karpathy_autoresearch_tweet.jpg)](https://x.com/karpathy/status/2030371219518931079)

*Andrej Karpathy's announcement of autoresearch. [View original tweet.](https://x.com/karpathy/status/2030371219518931079)*

For this class, you could apply auto-research to financial time series forecasting using the [Financial Time Series Forecasting Repository (FTSFR)](https://jeremybejarano.com/ftsfr/). FTSFR is an open benchmark that evaluates a suite of classical and modern forecasting methods on financial datasets — including arbitrage basis spreads, banking indicators, and asset returns. You would choose one of the FTSFR datasets and one of the forecasting methods, then use auto-research to systematically improve model performance.

## Startup Pitch with Working Prototype

Another class of project is something you might pitch to a VC firm — a product idea for an AI tool in finance, backed by a working prototype. The prototype should be the smallest possible case that demonstrates the idea is feasible. Think of it as a proof of concept you could demo in a meeting.

For inspiration on what makes a compelling AI startup idea, see this talk:

<iframe width="560" height="315" src="https://www.youtube.com/embed/TwDJhUJL-5o?si=fG40CJG8Ur_r5r3z" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Benchmarks

Design and build a benchmark harness, then run evaluations across models and report results. The goal is to create a reproducible evaluation framework for a specific AI capability relevant to finance, run it against multiple models, and report the results. Example benchmark ideas include: quant interview question performance (visual and non-visual), SEC filings information retrieval, financial time series prediction, academic paper table or plot extraction, and PDF-to-markdown conversion accuracy.

For an example of what professional AI benchmarking looks like, see [Artificial Analysis](https://artificialanalysis.ai/models?intelligence=coding-index). For more context on how benchmarks work and why they matter, see these videos:

<iframe width="560" height="315" src="https://www.youtube.com/embed/d5EltXhbcfA?si=y0IMX-RErDPzn5MH" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/CQGuvf6gSrM?si=k9qJQSEaFxHjQrkm" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Agent Use Cases

Build an AI agent that solves a specific problem in finance workflows. Examples include: a fact-checking agent, a plot format or common-sense checker (e.g., verifying that financial time series are smooth with no spurious breakpoints), or tools that enhance developer workflows like test-driven development with hold-out evaluation.

## Web App

Build a finance-focused AI web application. This could be a dashboard, an internal tool, or a consumer-facing product prototype that leverages generative or agentic AI.

---

More details and specific project options will be provided soon.
