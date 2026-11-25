import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv


sys.stdout.reconfigure(encoding="utf-8")  # Windows console pe special characters ke liye

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

SYSTEM_PROMPT = """You are a research agent. Given a topic, you research it on the web and write a well-sourced report.

Tools:
- web_search(query): find relevant pages. Returns titles, URLs and snippets.
- read_webpage(url): read the full text of one page.

Process:
1. PLAN: Before using any tool, list 3-5 sub-questions you need to answer.
2. RESEARCH: Search for each sub-question. Read the most relevant pages instead of relying on snippets alone. Use specific queries, and try a different query if results are weak.
3. CHECK: After each step, decide whether you have enough information. If a sub-question is still unanswered, keep researching. If you have enough, stop and write the report. Do not search just to search.
4. REPORT: Write the final report in Markdown.

Rules:
- Use only information found through your tools. Never invent facts, numbers or URLs.
- If sources disagree, or you cannot find something, say so clearly.
- Prefer recent and reliable sources. Note the date of data when it matters.
- Every key claim must cite a source URL.
- Be efficient: do not read the same page twice or repeat the same search.

Final report format:
# <Topic>
## Summary (3-5 lines)
## Key Findings (bullet points, each with a source link)
## Details (one section per sub-question)
## Gaps and Uncertainties
## Sources (list of URLs used)
"""


# ------------- functions -------------
def web_search(query: str)-> str:
    """Search the web and return results as plain text for the LLM"""

    try:
        response = tavily.search(query=query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"
    
    results = response.get("results", [])
    if not results:
        return "no results found"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r['title']}\n"
            f"  URL: {r['url']}\n"
            f"  {r['content'][:300]}"
        )

    return "\n\n".join(lines)

def execute_tool(name: str, arguments: str) -> str:
    """LLM ka tool call asli Python function mein badalke chalata hai"""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'"

    try:
        args = json.loads(arguments)
        return func(**args)
    except Exception as e:
        return f"Error running {name}: {e}"

MAX_PAGE_CHARS = 6000   # LLM ko kitna text bhejna hai (tokens bachane ke liye)

def read_webpage(url: str) -> str:
    """Download a webpage and return its readable text."""
    headers = {"User-Agent": "Mozilla/5.0 (research-agent learning project)"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        return f"Could not read page: {e}"

    if "text/html" not in r.headers.get("Content-Type", ""):
        return "Skipped: not an HTML page (maybe a PDF or file)."

    soup = BeautifulSoup(r.text, "html.parser")

    # kachra hatao: script, style, navigation, footer waghera
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    if not text:
        return "Page had no readable text."

    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n...[truncated]"

    return text
#-------------Tool schema---------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query. Returns titles, URLs and short snippets of top results. Use this to find pages relevant to the research topic.",
            "parameters": {
                "type": "object",
                "properties":{
                    "query": {
                        "type": "string",
                        "description": "A specific search query, e.g. 'Python developer salary Bangalore 2026'",
                    }
                },
                "required": ["query"],
            },
            
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": "Download and read the full text of one webpage. Use this on a promising URL from web_search results when the snippet is not detailed enough. Returns plain text (truncated if long).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL starting with http:// or https://",
                    }
                },
                "required": ["url"],
            },
        }
    }
]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_webpage": read_webpage,
}

MAX_STEPS = 8

def run_agent(topic: str)-> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": topic},
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n[step {step} Sending to LLM...]")
        resp = client.chat.completions.create(
            model = "openai/gpt-oss-120b",
            messages=messages,
            tools=TOOLS
        )

        msg = resp.choices[0].message

        if not msg.tool_calls:
            print("[done] LLM has give the final answer")
            return msg.content

        messages.append({
             "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments,
                    },
                }
                for c in msg.tool_calls
            ],
        })

        for call in msg.tool_calls:
            print(f"  -> {call.function.name}({call.function.arguments})")
            result = execute_tool(call.function.name, call.function.arguments)
            print(f"  <- {len(result)} chars result")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return "Max steps reach ho gaye, report poori nahi hui."


if __name__ == "__main__":
    while True:
        user_input = input("Your research topic (or 'quit'): ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Bye")
            break

        print(run_agent(user_input))
