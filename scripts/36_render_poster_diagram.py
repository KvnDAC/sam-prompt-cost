"""Rasterize the hand-drawn poster method diagram (SVG) to a transparent print PNG.

Uses a local Chromium browser (Edge or Chrome) in headless mode so text renders with
the system Arial, as in the poster. The SVG stays the editable source.
"""
import argparse,re,shutil,subprocess,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SVG=ROOT/'docs/waimlap_2026/poster_figures/method_diagram.svg'
CANDIDATES=[r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
  r'C:\Program Files\Google\Chrome\Application\chrome.exe','msedge','google-chrome','chromium']

def browser(explicit):
    for c in ([explicit] if explicit else CANDIDATES):
        if c and (Path(c).is_file() or shutil.which(c)):return c
    raise FileNotFoundError('No Edge/Chrome found; pass --browser')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--scale',type=int,default=3);ap.add_argument('--browser')
    a=ap.parse_args();svg=SVG.read_text(encoding='utf-8')
    w,h=(int(v) for v in re.search(r'viewBox="0 0 (\d+) (\d+)"',svg).groups())
    out=SVG.with_suffix('.png')
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Edge's crash handler may still hold the profile
        page=Path(tmp)/'diagram.html'
        page.write_text(f'<!doctype html><html><body style="margin:0;background:transparent">{svg}</body></html>',encoding='utf-8')
        # A throwaway profile keeps an already-open browser from absorbing the headless call.
        subprocess.run([browser(a.browser),'--headless',f'--user-data-dir={Path(tmp)/"profile"}','--no-first-run',
          '--disable-gpu','--hide-scrollbars','--default-background-color=00000000',
          f'--force-device-scale-factor={a.scale}',f'--window-size={w},{h}',f'--screenshot={out}',page.as_uri()],check=True,timeout=120)
    print(out)
if __name__=='__main__':main()
