"""Capture real app screenshots using deterministic synthetic demonstration input."""
from pathlib import Path
import argparse
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / 'assets'


def capture_region(page, top_locator, bottom_locator, path, padding=12):
    top_locator.wait_for(state='visible')
    page.wait_for_timeout(450)
    top = top_locator.bounding_box()
    bottom = bottom_locator.bounding_box()
    assert top and bottom
    region = {'x': 70, 'y': max(0, top['y'] - padding), 'width': 1220,
              'height': bottom['y'] + bottom['height'] - max(0, top['y'] - padding) + padding}
    if region['y'] + region['height'] > page.viewport_size['height']:
        raise ValueError(f'Screenshot region exceeds viewport: {region}')
    page.screenshot(path=str(path), clip=region, animations='disabled')
    print(path.name, region, flush=True)


def main(url, executable):
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox'])
        # A tall viewport avoids Streamlit rerun scroll restoration affecting crops.
        page = browser.new_page(viewport={'width': 1360, 'height': 4800}, device_scale_factor=1)
        page.goto(url, wait_until='domcontentloaded')
        example = page.get_by_role('button', name='Add example data', exact=True)
        example.wait_for(timeout=30000)
        example.click()
        page.wait_for_timeout(900)
        expander = page.locator('[data-testid="stExpander"]').filter(
            has=page.get_by_text('Optional biosignals — EEG parameters, ECG and SpO₂ context', exact=True))
        expander.locator('summary').click()
        page.get_by_text('6 EEG feature window(s)', exact=True).wait_for()
        expander.screenshot(path=str(OUT / 'app_eeg_inputs.png'), animations='disabled')
        print('app_eeg_inputs.png', expander.bounding_box(), flush=True)
        page.get_by_role('button', name='Run screening', exact=True).click()
        page.wait_for_timeout(900)
        latest = page.get_by_role('heading', name='Latest screening', exact=True)
        latest.wait_for()
        # The capture includes the genuine synthetic warning and per-window table.
        capture_region(page, latest,
                       page.get_by_role('button', name='Download EEG window predictions CSV', exact=True),
                       OUT / 'app_eeg_predictions.png')
        page.get_by_role('tab', name='Model and performance', exact=True).click()
        page.get_by_role('button', name='Show performance metrics', exact=True).click()
        heading = page.get_by_text('EEG subject-wise held-out performance', exact=True)
        heading.wait_for()
        table = page.get_by_role('tabpanel', name='Model and performance').locator('[data-testid="stDataFrame"]').first
        table.screenshot(path=str(OUT / 'app_eeg_performance.png'), animations='disabled')
        print('app_eeg_performance.png', table.bounding_box(), flush=True)
        browser.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8517')
    parser.add_argument('--browser', default='/opt/google/chrome/chrome')
    args=parser.parse_args()
    main(args.url,args.browser)
