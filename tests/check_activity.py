"""Functional browser checks for mathematical and interactive behavior."""
from pathlib import Path
from itertools import permutations
import json
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]/'public'
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1200,'height':1000})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'index.html').as_uri())
    assert page.locator('figure').count()==5
    assert page.locator('figure img').evaluate_all('(imgs)=>imgs.every(i=>i.complete && i.naturalWidth>0)')
    page.locator('#check-coloring').click()
    assert 'Choose a color' in page.locator('#puzzle-status').inner_text()
    for i in range(3):page.select_option(f'#color{i}','0')
    page.locator('#check-coloring').click()
    assert 'Not valid' in page.locator('#puzzle-status').inner_text()
    for j,perm in enumerate(permutations('012'),1):
        for i,v in enumerate(perm):page.select_option(f'#color{i}',v)
        page.locator('#check-coloring').click()
        assert page.locator('#graph-count').inner_text()==f'{j} of 6 solutions discovered'
    page.locator('#check-coloring').click()
    assert 'already discovered' in page.locator('#puzzle-status').inner_text()
    assert page.locator('#graph-count').inner_text()=='6 of 6 solutions discovered'
    assert page.get_by_text('Show all solutions',exact=True).count()==0
    assert page.locator('#reset-puzzle,.discoveries').count()==0

    result=page.evaluate('''() => {
      for (let c=0;c<=100;c++) {
        const q=ModeDemo.qFor(c);
        if(Math.abs(q.reduce((a,b)=>a+b,0)-1)>1e-12)throw Error('probability normalization');
        let last=0;
        for(let k=1;k<=128;k++) {const v=ModeDemo.expected(q,k);if(v<last-1e-12||v>6+1e-12)throw Error('discovery bounds');last=v;}
      }
      const q=ModeDemo.qFor(0);
      if(Math.abs(ModeDemo.expected(q,1)-.9)>1e-12)throw Error('one draw');
      if(ModeDemo.sample(q,8,()=>.99)[6]!==8)throw Error('invalid sample handling');
      if(ModeDemo.sample(q,8,()=>0)[0]!==8)throw Error('repeat handling');
      return {even8:ModeDemo.expected(q,8),concentrated8:ModeDemo.expected(ModeDemo.qFor(100),8)};
    }''')
    assert abs(result['even8']-4.365056849765626)<1e-10
    assert abs(result['concentrated8']-1.211961325920005)<1e-10
    assert '1.21' in page.locator('#expectations').text_content()
    page.locator('#concentrated').click()
    assert page.locator('#share').inner_text()=='97.0%'
    assert '1.21' in page.locator('#expectations').text_content()
    page.locator('#sampler summary').click()
    assert page.locator('#expectations').is_visible()
    assert page.locator('#prediction').count()==0
    assert '97 use solution A' in page.locator('#distribution-summary').inner_text()
    assert page.locator('[data-draw],#sample-results,#reply-tiles').count()==0
    page.locator('#sampler').screenshot(path='/tmp/blog-sampler.png')
    page.locator('#even').click()
    assert page.locator('#share').inner_text()=='16.7%'
    assert '4.37' in page.locator('#expectations').inner_text()
    paths=page.locator('#sampling-chart polyline').evaluate_all('(es)=>es.map(e=>e.getAttribute("points"))')
    assert paths[0]==paths[1], 'Even distribution should match reference curve'
    page.evaluate("window.dispatchEvent(new CustomEvent('openai:set_globals',{detail:{globals:{widgetState:{solutionCurve:{concentration:100,counts:[8,0,0,0,0,0,0],k:8}}}}}))")
    assert page.locator('#share').inner_text()=='97.0%'
    assert page.locator('[data-draw],#sample-results,#reply-tiles').count()==0
    page.locator('#even').click()
    page.locator('#concentration').focus()
    page.keyboard.press('ArrowRight')
    assert page.locator('#concentration').input_value()=='1'
    page.screenshot(path='/tmp/blog-desktop.png',full_page=True)
    for width in [390,320]:
        page.set_viewport_size({'width':width,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Horizontal page overflow'
        for el in ['#puzzle','#sampler']:
            box=page.locator(el).bounding_box()
            assert box['x']>=0 and box['x']+box['width']<=width
        page.locator('#sampler').screenshot(path=f'/tmp/blog-sampler-{width}.png')
    for filename in ['puzzle.html','sampler.html']:
        page.goto((ROOT/filename).as_uri())
        assert page.locator('section.interactive').count()==1
    assert not errors,errors
    browser.close()
print(json.dumps({'status':'passed','checks':['all six valid colorings','invalid and incomplete coloring rejection','duplicate handling','removed reveal and collection controls','probability normalization','analytic discovery expectations','sampling boundaries','curve controls and legacy saved-state restoration','keyboard range control','loaded images','desktop and 390/320px layouts','standalone embeds','no browser errors'],**result},indent=2))
