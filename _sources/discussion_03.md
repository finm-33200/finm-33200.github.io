# Discussion 3: AI Application Development & Financial Data Integration

**Duration:** 3 hours
**Format:** Hands-on workshop

In this workshop, we'll build a production-ready AI chat application that integrates real financial data. You'll learn the fundamentals of modern web development with Next.js, deploy a ChatKit-powered assistant, and integrate SEC filings and market data APIs to create a **Company Research Assistant**.

## Learning Objectives

By the end of this session, you will be able to:

- **Understand AI chat application architecture** for financial services
- **Compare local vs. cloud AI development** approaches
- **Build and deploy Next.js applications** on Vercel
- **Integrate ChatKit** to create production-ready chat interfaces
- **Access financial data** from SEC EDGAR and Yahoo Finance APIs
- **Implement AI function calling** to connect LLMs with real-time data
- **Deploy production applications** with proper environment configuration

## What You'll Build

A **Company Research Assistant** that can:
- Answer natural language questions about any public company
- Fetch and analyze SEC filings (10-K, 10-Q, 8-K)
- Retrieve real-time stock prices and market data
- Provide cited sources for all information
- Stream responses in real-time

## Prerequisites

- Basic knowledge of JavaScript/TypeScript
- Familiarity with React (helpful but not required)
- [OpenAI API key](https://platform.openai.com/api-keys) (will walk through setup)
- Node.js 18.17+ installed
- GitHub account
- Code editor (VS Code recommended)

## Session Outline

### Part 1: Introduction & Architecture (30 min)
Overview of AI chat applications, ChatKit framework, and the architecture we'll build.

### Part 2: Next.js Fundamentals (45 min)
Learn Next.js basics, API routes, environment variables, and Vercel deployment.

### Part 3: ChatKit Integration (1h 45 min)
Hands-on implementation: Clone starter app, integrate financial APIs, create custom tools, and deploy.

## Course Materials

```{toctree}
:maxdepth: 1
discussions/chat_applications_overview.md
discussions/intro_to_nextjs.md
discussions/intro_to_chatkit.md
```

## Additional Resources

### Documentation
- [OpenAI ChatKit](https://openai.github.io/chatkit-js/)
- [Next.js Documentation](https://nextjs.org/docs)
- [SEC EDGAR API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [Vercel Platform](https://vercel.com/docs)

### Repositories
- [ChatKit Starter App](https://github.com/openai/openai-chatkit-starter-app)
- [ChatKit Advanced Samples](https://github.com/openai/openai-chatkit-advanced-samples)

### Tools
- [ChatKit Studio](https://chatkit.studio/) - Visual workflow builder
- [OpenAI Platform](https://platform.openai.com/) - API keys and settings

## Project Extensions

After completing the base workshop, consider extending your application with:

1. **Portfolio Tracking** - Let users save and monitor multiple stocks
2. **Historical Analysis** - Compare company performance over time
3. **News Integration** - Add recent news about companies
4. **Visualization** - Charts for stock prices and financial metrics
5. **Multi-Company Comparison** - Compare competitors side-by-side
6. **Alert System** - Notify users of significant company events
7. **Document Search** - Semantic search through SEC filings using RAG

## Assessment

Students should be able to:
1. Explain the architecture of their AI application
2. Deploy a functional Company Research Assistant
3. Demonstrate integration with at least 2 financial data APIs
4. Implement proper error handling and citations
5. Discuss considerations for production deployment

---

**Ready to get started?** Begin with the [Chat Applications Overview](discussions/chat_applications_overview.md)!

