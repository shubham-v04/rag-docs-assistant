import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.ingestion.loader import LoadedDocument

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^```[^\n]*\n.*?^```[ \t]*$", re.MULTILINE | re.DOTALL)
# a code block bigger than this is cut by lines instead of kept whole
MAX_CODE_FACTOR = 2


# Technical docs are organised by headings, and an answer usually lives inside one
# section. Splitting on headers first keeps each section together and gives every
# chunk a header path to cite. Long sections are then cut with the recursive splitter
# on paragraph boundaries, and fenced code blocks are never cut mid-block unless they
# are far larger than a chunk. MarkdownHeaderTextSplitter is not used because it strips
# leading whitespace, which destroys indentation in code examples.
def chunk_document(doc: LoadedDocument) -> list[Document]:
    settings = get_settings()
    prose_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    chunks: list[Document] = []
    for header_path, body in split_sections(doc.text):
        units = section_units(body, prose_splitter, settings.chunk_size)
        for text in pack(units, settings.chunk_size):
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": doc.source,
                        "title": doc.title,
                        "header_path": header_path,
                        "chunk_index": len(chunks),
                    },
                )
            )
    return chunks


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split on h1-h3 headings, ignoring '#' lines inside code fences."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    path: list[str] = []
    in_fence = False

    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            path = path[: level - 1] + [match.group(2)]
            sections.append((" > ".join(path), []))
        sections[-1][1].append(line)

    return [
        (p, "\n".join(lines).strip()) for p, lines in sections if "".join(lines).strip()
    ]


def section_units(
    body: str, prose_splitter: RecursiveCharacterTextSplitter, size: int
) -> list[str]:
    """Break a section into pieces that are safe to pack: prose parts and whole code blocks."""
    units: list[str] = []
    pos = 0
    for match in FENCE_RE.finditer(body):
        units += prose_units(body[pos : match.start()], prose_splitter, size)
        units += code_units(match.group(0), size * MAX_CODE_FACTOR)
        pos = match.end()
    units += prose_units(body[pos:], prose_splitter, size)
    return units


def prose_units(
    text: str, splitter: RecursiveCharacterTextSplitter, size: int
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return splitter.split_text(text) if len(text) > size else [text]


def code_units(block: str, limit: int) -> list[str]:
    if len(block) <= limit:
        return [block]
    lines = block.splitlines()
    opener, closer, body = lines[0], lines[-1], lines[1:-1]
    parts: list[list[str]] = [[]]
    length = 0
    for line in body:
        if length + len(line) > limit and parts[-1]:
            parts.append([])
            length = 0
        parts[-1].append(line)
        length += len(line) + 1
    return ["\n".join([opener, *part, closer]) for part in parts]


def pack(units: list[str], size: int) -> list[str]:
    """Greedily merge neighbouring units so small pieces do not become tiny chunks."""
    packed: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 2 > size:
            packed.append(current)
            current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        packed.append(current)
    return packed
