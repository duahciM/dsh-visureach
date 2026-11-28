"""Prepare fixed new real windows without using any automatic proposals."""

from pathlib import Path
from shutil import copy2

from PIL import Image, ImageDraw

from dacf_research.prepare_inputs import ROOT, digest, read, write
from evidenweft_phase1.source_review import native_stream, timeline

H = Path(__file__).resolve().parent


def inventory():
    media = Path('C:/Users/Administrator/Desktop/视觉插件/素材')
    cases = read(ROOT / 'partial_modules_stage3/inputs/source_inventory.json')
    (H / 'inputs').mkdir()
    for case, filename in (('new-pan', '7.mp4'), ('new-local', '9.mp4')):
        path = H / 'inputs' / filename
        copy2(media / filename, path)
        info, clock = timeline(path)
        indices = [round((len(clock)-1)*i/8) for i in range(1,8)]
        rows = [clock[i] for i in indices]
        boxes = {r['pts']: i for i, r in enumerate(rows)}
        sheet = Image.new('RGB', (1155, 680), '#171b22')
        for row, data in native_stream(path, info, clock):
            if row['pts'] not in boxes:
                continue
            image = Image.frombytes('RGB', (info['width'], info['height']), data)
            image.thumbnail((375, 196))
            x, y = (boxes[row['pts']] % 3)*385, (boxes[row['pts']] // 3)*225
            sheet.paste(image, (x, y))
            ImageDraw.Draw(sheet).text((x+8,y+197), f"PTS {row['pts']} · frame {row['frame']} · {info['width']} x {info['height']}", fill='white')
        sheet.save(H / 'inputs' / f'{case}.png')
        cases[case] = {'path':path.relative_to(ROOT).as_posix(), 'sha256':digest(path), 'pts':[r['pts'] for r in rows], 'explicit_goal_queries':[]}
    write(H / 'inputs/source_inventory.json',cases)


if __name__ == '__main__':
    inventory()
