"""Update the user architecture document and preserve the full previous version."""
from pathlib import Path
import os

from dacf_research.prepare_inputs import digest,write
from partial_modules_stage1.fixtures import HERE

TARGET=Path('C:/Users/Administrator/Desktop/视觉插件/文档/视频观察插件-架构与功能说明.md')


def main():
    original=TARGET.read_bytes()
    bom=original.startswith(b'\xef\xbb\xbf')
    before=original.decode('utf-8-sig')
    eol='\r\n' if '\r\n' in before else '\n'
    text=before.replace('\r\n','\n')

    def replace(one,next_text):
        nonlocal text
        if text.count(one)!=1:
            raise ValueError(f'Expected one v8 heading or description: {one}')
        text=text.replace(one,next_text,1)

    begin='> **文档 v8 · 2026-11-27。**'
    replacement=('> **文档 v9 · 2026-11-28。** 当前可安装实现为 **VisuReach 0.6.0 离线版**。'
                 '独立研究延伸至 DACF-AMR Round 6、工具说明 Round 7 及 M2/M5 部分实现阶段一。'
                 '研究输出与正式工具面分别验收，真实源时标及原画质由各自入口保留。实时方向按既有冻结范围保全。')
    line=next(line for line in text.splitlines() if line.startswith(begin))
    replace(line,replacement)
    replace('[独立研究清单](/D:/codex/p1-transition/dacf_round2/run_manifest.json)',
            '[独立研究起点](/D:/codex/p1-transition/dacf_round2/run_manifest.json) · [阶段一实测](/D:/codex/p1-transition/partial_modules_stage1/STAGE1_REPORT.md)')
    replace('第 24 章说明当前研究入口与产品边界。','第 24 章保留独立研究的历史入口。第 25 章更新 M2–M6 的实际能力、Round 7 结项和阶段一验证。')
    replace('| **24** | **当前独立研究：EvidenWeft 阶段一与 DACF 两轮** |',
            '| **24** | **独立研究起点：EvidenWeft 阶段一与 DACF 第一、二轮** |\n| **25** | **M2–M6 实际实现、Round 7 结项及阶段一** |')
    replace('# 24　当前独立研究 · EvidenWeft 与 DACF','# 24　独立研究起点 · EvidenWeft 与 DACF 第一、二轮')
    replace('第 23 章记录当前发布形态与运行边界，第 24 章记录独立研究的位置。第 16–22 章按历史时间点查阅。',
            '第 23 章记录正式发布边界，第 24 章记录前两轮研究，第 25 章记录最新模块实现和试验。第 16–22 章按历史时间点查阅。')
    section=(HERE/'ARCHITECTURE_DELTA.md').read_text(encoding='utf8').strip()
    replace('## 附：本文件与《研究综合档案》的分工（更新）',section+'\n\n---\n\n## 附：本文件与《研究综合档案》的分工（更新）')
    for i,label in enumerate(('身份识别','空间关系','差异发现','状态与关系','独立证据判断'),3):
        header=f'## 3.{i} M{i-1}　{label}　【部分】'
        replace(header,header+'\n\n本节保留完整规格。正式已实现、研究原型与未验证部分见第 25 章实测对照。')
    old_sha=digest(TARGET)
    with (HERE/'architecture-v8.md').open('xb') as archive:
        archive.write(original)
    payload=text.replace('\n',eol).encode('utf-8-sig' if bom else 'utf-8')
    temporary=TARGET.with_suffix('.round7-stage1-update.tmp')
    with temporary.open('xb') as file:
        file.write(payload)
    if TARGET.read_bytes()!=original:
        raise RuntimeError('Architecture document changed during the update; the verified replacement is still separate.')
    os.replace(temporary,TARGET)
    assert TARGET.read_bytes()==payload
    write(HERE/'architecture_update.json',{'old_version':'v8','new_version':'v9','original_sha256':old_sha,'updated_sha256':digest(TARGET),
           'source_backup':str(HERE/'architecture-v8.md'),'actual_document':str(TARGET),'history_preserved':'Sections 16–22 retained verbatim'})


if __name__=='__main__':main()
