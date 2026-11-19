"""Restore the delivered PowerPoint from the current editable master.

Edit assets/presentation_master.pptx in PowerPoint or Impress to update future
copies. Keeping an editable master preserves changes made directly in the deck.
"""
from pathlib import Path
import argparse
import posixpath
import shutil
from zipfile import ZipFile

from lxml import etree

HERE = Path(__file__).resolve().parent
MASTER = HERE / 'assets' / 'presentation_master.pptx'
OUTPUT = HERE / 'Sleep_Apnea_SVM_Software_Presentation.pptx'


def strip_annotations(path: Path) -> None:
    """Remove speaker notes and reviewer comments, including package references."""
    with ZipFile(path) as archive:
        parts = [(entry, archive.read(entry.filename)) for entry in archive.infolist()]
    removed = {
        entry.filename for entry, _ in parts
        if entry.filename.startswith(('ppt/notesSlides/', 'ppt/notesMasters/',
                                      'ppt/comments/', 'ppt/threadedComments/'))
        or entry.filename in ('ppt/commentAuthors.xml', 'ppt/authors.xml')
    }
    temporary = path.with_suffix('.pptx.tmp')
    with ZipFile(temporary, 'w') as archive:
        for entry, data in parts:
            name = entry.filename
            if name in removed:
                continue
            root = None
            if name.endswith('.rels'):
                root = etree.fromstring(data)
                base = '' if name == '_rels/.rels' else posixpath.dirname(name).replace('/_rels', '')
                for relationship in list(root):
                    if relationship.get('TargetMode') == 'External':
                        continue
                    target = relationship.get('Target', '')
                    resolved = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(base, target))
                    if resolved in removed:
                        root.remove(relationship)
            elif name == '[Content_Types].xml':
                root = etree.fromstring(data)
                for declaration in list(root):
                    if declaration.get('PartName', '').lstrip('/') in removed:
                        root.remove(declaration)
            elif name == 'ppt/presentation.xml':
                root = etree.fromstring(data)
                for element in root.findall('{http://schemas.openxmlformats.org/presentationml/2006/main}notesMasterIdLst'):
                    root.remove(element)
            if root is not None:
                data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
            archive.writestr(entry, data)
    temporary.replace(path)


def build(source: Path = MASTER, output: Path = OUTPUT) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f'Presentation master is missing: {source}')
    if source.resolve() == output.resolve():
        raise ValueError('Use a different output path to preserve the presentation master.')
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    strip_annotations(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=MASTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(build(args.source, args.output))
