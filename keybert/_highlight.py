from rich.console import Console
from rich.highlighter import RegexHighlighter
import string


class NullHighlighter(RegexHighlighter):
    base_style = ""
    highlights = [r""]

def clean_text(astr):
    clean_str = astr.translate(str.maketrans('', '', string.punctuation))
    clean_str = ' '.join(clean_str.split()).lower()
    return clean_str

def highlight_document(doc, keywords):
    highlighted_text = clean_text(doc)
    for kwd in keywords:
      kwd = clean_text(kwd)
      highlighted_text=highlighted_text.replace(kwd,f'[black on #FFFF00]{kwd}[/]')
    console = Console(highlighter=NullHighlighter())
    console.print(highlighted_text)
