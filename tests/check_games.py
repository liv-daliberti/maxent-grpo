from pathlib import Path
from playwright.sync_api import sync_playwright
import json, subprocess
ROOT=Path(__file__).resolve().parents[1]
errors=[]
with sync_playwright() as p:
    b=p.chromium.launch(args=['--no-sandbox','--disable-dev-shm-usage'])
    page=b.new_page(viewport={'width':1200,'height':1000})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'public/games.html').as_uri())
    def tab(name):page.locator('#tab-'+name).click()
    assert page.locator('.discoveries,[data-reset-game],#reset-puzzle').count()==0
    assert page.get_by_text('Show all solutions',exact=True).count()==0
    assert page.locator('.game-message:not([hidden])').count()==0
    def countdown(s):
        page.locator('#expression').fill(s)
        page.locator('#expression').press('Enter')
    tab('countdown');countdown('3+6')
    assert page.locator('#countdown-message').get_attribute('data-kind')=='error'
    countdown('3+6+9');countdown('6+3+9')
    assert page.locator('#countdown-count').inner_text()=='1 of 8 solutions discovered'
    countdown('(6*9)/3')
    assert page.locator('#countdown-count').inner_text().startswith('2 of ')
    for bad in ['3*6*9','3+3+6+6','alert(1)','3/(6-6)','3+6+9)', '3 6 9','3**6+9']:
        countdown(bad);assert page.locator('#countdown-message').get_attribute('data-kind')=='error',bad
    page.locator('#clear-expression').click();page.locator('[data-tile="3"]').click()
    assert page.locator('#expression').input_value()=='3'
    tab('python');page.locator('#run-python').click()
    assert page.locator('#python-count').inner_text()=='1 of 32 solutions discovered'
    page.locator('#run-python').click()
    assert page.locator('#python-count').inner_text().startswith('1 of ')
    for i,v in enumerate([41,7,3,2]):page.select_option('#divisor-'+str(i),str(v))
    page.locator('#run-python').click()
    assert page.locator('#python-count').inner_text().startswith('2 of ')
    page.select_option('#divisor-0','1');page.locator('#run-python').click()
    assert page.locator('#python-message').get_attribute('data-kind')=='error'
    tab('algebra')
    for action in ['C','F']:page.locator('[data-action="'+action+'"]').click()
    page.locator('#check-algebra').click()
    assert page.locator('#equation').inner_text()=='x = 16'
    page.locator('#check-algebra').click()
    assert page.locator('#algebra-count').inner_text()=='1 of 9 solutions discovered'
    page.locator('#restart-algebra').click()
    for action in ['F','E']:page.locator('[data-action="'+action+'"]').click()
    page.locator('#check-algebra').click()
    assert page.locator('#algebra-count').inner_text().startswith('2 of ')
    page.locator('#restart-algebra').click()
    page.locator('[data-action="A"]').click();page.locator('[data-action="F"]').click()
    assert 'earlier equation' in page.locator('#algebra-message').inner_text()
    assert page.locator('#step-count').inner_text()=='1 / 4 steps'
    page.locator('#undo-action').click()
    assert page.locator('#step-count').inner_text()=='0 / 4 steps'
    for i in range(5):page.locator('[data-action="C"]').click()
    assert 'four actions' in page.locator('#algebra-message').inner_text()
    tab('pantry');page.locator('#pantry-form button[type=submit]').click()
    assert 'two and four' in page.locator('#pantry-message').inner_text()
    assert page.locator('.pantry-totals').is_visible()
    assert page.locator('#amount-4').is_disabled()
    for v in [4,5]:page.locator(f'[name="ingredient"][value="{v}"]').check()
    page.locator('#pantry-form button[type=submit]').click()
    assert page.locator('#pantry-message').get_attribute('data-kind')=='error'
    assert '100 g' in page.locator('#pantry-totals').inner_text()
    page.locator('#suggest-amounts').click()
    assert page.locator('#pantry-ready').get_attribute('data-ready')=='true'
    page.locator('#pantry-form button[type=submit]').click()
    assert page.locator('#pantry-message').get_attribute('data-kind')=='success'
    assert page.locator('#pantry-ready').get_attribute('data-ready')=='true'
    page.select_option('#amount-4','50');page.select_option('#amount-5','50')
    assert page.locator('#pantry-ready').get_attribute('data-ready')=='false'
    page.locator('#suggest-amounts').click()
    page.locator('#pantry-form button[type=submit]').click()
    assert page.locator('#pantry-count').inner_text().startswith('1 of ')
    for v in [4,5]:page.locator(f'[name="ingredient"][value="{v}"]').uncheck()
    for v in [0,3]:page.locator(f'[name="ingredient"][value="{v}"]').check()
    page.select_option('#amount-0','50');page.select_option('#amount-3','75')
    assert page.locator('#pantry-ready').get_attribute('data-ready')=='true'
    page.locator('#pantry-form button[type=submit]').click()
    assert page.locator('#pantry-count').inner_text().startswith('2 of ')
    page.locator('[name="ingredient"][value="2"]').check()
    page.locator('#pantry-form button[type=submit]').click()
    assert 'outside their targets' in page.locator('#pantry-message').inner_text()
    page.locator('#suggest-amounts').click()
    assert 'No allowed quantities' in page.locator('#pantry-message').inner_text()
    assert page.locator('#pantry-count').inner_text().startswith('2 of ')
    tab('graph')
    for i,v in enumerate(['0','1','2']):page.select_option('#color'+str(i),v)
    page.locator('#check-coloring').click()
    assert page.locator('#graph-count').inner_text()=='1 of 6 solutions discovered'
    page.locator('#tab-graph').focus();page.keyboard.press('ArrowRight')
    assert page.locator('#tab-countdown').get_attribute('aria-selected')=='true'
    assert page.locator('#countdown-count').inner_text().startswith('2 of ')
    page.reload();tab('countdown')
    assert page.locator('#countdown-count').inner_text().startswith('0 of ')
    for width in [1200,390,320]:
        page.set_viewport_size({'width':width,'height':900})
        for game in ['graph','countdown','python','algebra','pantry']:
            tab(game)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),(width,game)
            assert page.locator('#panel-'+game).is_visible()
            assert page.locator('[data-game-panel]:visible').count()==1
            if width in [1200,390]:page.screenshot(path=f'/tmp/game-{game}-{width}.png',full_page=True)
    page.goto((ROOT/'public/index.html').as_uri())
    assert page.locator('figure').count()==5
    assert page.locator('.games-link').first.get_attribute('href')=='games.html'
    tab('countdown');countdown('(6*9)/3')
    assert page.locator('#countdown-count').inner_text().startswith('1 of ')
    page.locator('#concentrated').click();page.locator('[data-draw="8"]').click()
    assert '8 replies' in page.locator('#sample-results').inner_text()
    assert not errors,errors
    b.close()
print(json.dumps({'status':'passed','checks':['two distinct solutions and duplicate detection in each new game','incorrect answers rejected','safe arithmetic parser','algebra repeated-state and step limits','pantry quantity witness','discovery counting and tab state','keyboard tab navigation','all five panels at 1200/390/320px','article integration','existing sampler retained','no browser errors']},indent=2))
