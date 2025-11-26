import base64, os
from pathlib import Path
bundle_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath('lightroom_clone/main_window.py')), '..'))
icon_path = os.path.normpath(os.path.join(bundle_root, 'icons', 'check.png'))
check_icon_css = ''
try:
    with open(icon_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('ascii')
    check_icon_css = f'image: url("data:image/png;base64,{encoded}");'
except Exception as exc:
    print('err', exc)
print(len(check_icon_css))
print(check_icon_css[:80])
