"""Convert a Markdown file to a styled PDF."""

import sys
import markdown
from weasyprint import HTML

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/md_to_pdf.py <input.md> [output.pdf]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".md", ".pdf")

    with open(input_path) as f:
        md_content = f.read()

    html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: A4;
        margin: 1.5cm 2cm;
        @bottom-center {{
            content: counter(page);
            font-size: 9pt;
            color: #666;
        }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.5;
        color: #1a1a1a;
        max-width: 100%;
    }}
    h1 {{
        font-size: 20pt;
        border-bottom: 2px solid #e67e22;
        padding-bottom: 8px;
        margin-top: 0;
    }}
    h2 {{
        font-size: 15pt;
        color: #e67e22;
        border-bottom: 1px solid #eee;
        padding-bottom: 4px;
        margin-top: 24px;
    }}
    h3 {{
        font-size: 12pt;
        color: #333;
        margin-top: 18px;
    }}
    h4 {{
        font-size: 11pt;
        color: #555;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0;
        font-size: 9.5pt;
    }}
    th {{
        background-color: #1a1a2e;
        color: #fff;
        padding: 6px 8px;
        text-align: left;
        font-weight: 600;
    }}
    td {{
        padding: 5px 8px;
        border-bottom: 1px solid #ddd;
    }}
    tr:nth-child(even) {{
        background-color: #f8f8f8;
    }}
    strong {{
        color: #e67e22;
    }}
    code {{
        background-color: #f4f4f4;
        padding: 1px 4px;
        border-radius: 3px;
        font-size: 9.5pt;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }}
    hr {{
        border: none;
        border-top: 1px solid #ddd;
        margin: 20px 0;
    }}
    p {{
        margin: 6px 0;
    }}
    ul, ol {{
        margin: 6px 0;
        padding-left: 20px;
    }}
    li {{
        margin: 3px 0;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML(string=full_html).write_pdf(output_path)
    print(f"PDF written to {output_path}")

if __name__ == "__main__":
    main()
