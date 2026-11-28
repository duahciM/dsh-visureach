"""Verified exact local source reads and source-versioned PNG receipts."""
from fractions import Fraction
from hashlib import sha256
import re
import subprocess
from time import perf_counter

from PIL import Image

from dacf_research.prepare_inputs import digest


class Reader:
    def __init__(self, source, signature, stream, clock, folder):
        self.source = source
        self.signature = signature
        self.stream = stream
        self.clock = {r['pts']: r for r in clock}
        self.folder = folder
        self.folder.mkdir()
        self.receipts = {}
        self.failed = {}

    def read(self, pts):
        if pts in self.receipts:
            return self.receipts[pts], False
        row = self.clock[pts]
        w, h = self.stream['width'], self.stream['height']
        target = self.folder/f'pts-{pts}.png'
        start = perf_counter()
        origin = next(iter(self.clock.values()))['seconds']
        seek = max(0., row['seconds']-origin-0.30)
        attempts = []
        for skip in dict.fromkeys((seek, 0.)):
            command = ['ffmpeg','-hide_banner','-loglevel','info','-xerror','-threads','1','-filter_threads','1']
            if skip:
                command += ['-ss', f'{skip:.6f}']
            command += ['-copyts','-noautorotate','-i',str(self.source),'-map','0:v:0','-vf',
                        f"showinfo,select=eq(pts\\,{pts}),showinfo,format=rgb24",'-fps_mode','passthrough',
                        '-frames:v','1','-f','rawvideo','-']
            try:
                process = subprocess.run(command, capture_output=True, timeout=30)
            except subprocess.TimeoutExpired:
                attempts.append({'seek_seconds':skip,'frames_at_input_filter':None,'output_pts':[],
                                 'verified':False,'state':'timeout'})
                continue
            error = process.stderr.decode('utf-8', errors='replace')
            witnessed = [int(p) for p in re.findall(r'Parsed_showinfo_2[^\n]+?\bpts:\s*(-?\d+)\s+pts_time:',error)]
            seen = len(re.findall(r'Parsed_showinfo_0[^\n]+?\bpts:\s*(-?\d+)\s+pts_time:',error))
            base = re.search(r'config in time_base: ([\d/]+)',error)
            verified = (process.returncode == 0 and base and Fraction(base[1]) == Fraction(self.stream['time_base'])
                        and witnessed == [pts] and len(process.stdout) == w*h*3)
            attempts.append({'seek_seconds':skip,'frames_at_input_filter':seen,'output_pts':witnessed,
                             'verified':bool(verified),'exit_code':process.returncode})
            if not verified:
                continue
            pixels = process.stdout
            Image.frombytes('RGB',(w,h),pixels).save(target)
            with Image.open(target) as decoded:
                assert decoded.tobytes() == pixels
            receipt = row | {'id':f'{self.signature}:{row["frame"]}:{pts}:{self.stream["time_base"]}',
                             'time_base':self.stream['time_base'],'source_frame':row['frame'],
                             'source_sha256':self.signature,'native_size':[w,h],'path':str(target.resolve()),
                             'rgb_sha256':sha256(pixels).hexdigest(),'png_sha256':digest(target), 'png_bytes':target.stat().st_size,
                             'decode_attempts':attempts,'wall_seconds':round(perf_counter()-start,6)}
            self.receipts[pts] = receipt
            return receipt, True
        self.failed[pts] = attempts
        raise ValueError(f'Could not verify source PTS {pts}')
