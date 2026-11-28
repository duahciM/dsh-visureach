from pathlib import Path
from dacf_round6.native import trial,H

if __name__=='__main__':
    area=H/'wire';area.mkdir()
    trial(area/'P3-wire','fixture','P3','A2',
        'Connection and system-field verification only. Please reply with a brief JSON acknowledgment. No video observation required.',
        H/'wire.mjs')
