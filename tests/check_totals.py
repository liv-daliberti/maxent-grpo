from pathlib import Path
from itertools import product
from fractions import Fraction
import json, subprocess
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
catalogs=json.loads(subprocess.check_output(['node','-e',"const g=require('./public/games.js');console.log(JSON.stringify(Object.fromEntries(Object.entries(g.catalogs()).map(([k,v])=>[k,[...v]]))))"],cwd=ROOT,text=True))
assert {k:len(v) for k,v in catalogs.items()}=={'countdown':8,'python':32,'algebra':9,'pantry':14}
# Independent full-support check: every proper-divisor vector is reachable.
factors=[[d for d in range(2,n) if n%d==0] for n in [18,82,91,93]]
assert {','.join(map(str,v)) for v in product(*factors)}=={k for k,_ in catalogs['python']}
# Enumerate all permitted quantities independently, with exact decimal fractions.
nutrition=[[Fraction(s) for s in row] for row in [
 ['584','21.5','10.8','0'],['97','.74','4.62','0'],['45','.941','3.1','86.6'],
 ['27','.83','2.1','6'],['47','.91','2','9'],['571','18.9','7.22','0']]]
supports=set()
for grams in product(*[[0]+list(range(50,m+1,25)) for m in [100,125,150,100,150,125]]):
    if not 2<=sum(g>0 for g in grams)<=4 or not 125<=sum(grams)<=200:continue
    e,p,f,s=[sum(grams[i]*nutrition[i][j]/100 for i in range(6)) for j in range(4)]
    if Fraction('239.4')<=e<=Fraction('432.25') and p>=Fraction('6.423') and f>=Fraction('3.478') and s<=Fraction('37.6'):
        supports.add(''.join('1' if g else '0' for g in grams))
assert supports=={k for k,_ in catalogs['pantry']}

errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(args=['--no-sandbox','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1200,'height':1000})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'public/games.html').as_uri())
    for game,entries in catalogs.items():
        page.locator('#tab-'+game).click()
        total=len(entries)
        assert page.locator('#'+game+'-count').inner_text()==f'0 of {total} solutions discovered'
        def submit(witness):
            if game=='countdown':
                page.locator('#expression').fill(witness);page.locator('#expression').press('Enter')
            elif game=='python':
                for i,v in enumerate(witness):page.select_option('#divisor-'+str(i),str(v))
                page.locator('#run-python').click()
            elif game=='algebra':
                page.locator('#restart-algebra').click()
                for a in witness:page.locator('[data-action="'+a+'"]').click()
                page.locator('#check-algebra').click()
            else:
                for i in range(6):page.locator(f'[name="ingredient"][value="{i}"]').set_checked(i in witness)
                page.locator('#suggest-amounts').click()
                page.locator('#pantry-form button[type=submit]').click()
        for n,(_,witness) in enumerate(entries,1):
            submit(witness)
            assert page.locator('#'+game+'-count').inner_text()==f'{n} of {total} solutions discovered'
            assert page.locator('#'+game+'-progress').get_attribute('value')==str(n)
        assert page.locator('#'+game+'-remaining').inner_text()=='Collection complete'
        assert f'All {total} solutions discovered!' in page.locator('#'+game+'-message').inner_text()
        submit(entries[-1][1])
        assert page.locator('#'+game+'-count').inner_text()==f'{total} of {total} solutions discovered'
        assert 'already discovered' in page.locator('#'+game+'-message').inner_text()
    page.reload()
    page.locator('#tab-countdown').click()
    for e in ['--3+6+9','3-(-6)+9','+(3+6+9)','3+6','3*6*9']:
        page.locator('#expression').fill(e);page.locator('#expression').press('Enter')
        assert page.locator('#countdown-message').get_attribute('data-kind')=='error'
        assert page.locator('#countdown-count').inner_text()=='0 of 8 solutions discovered'
    for width in [1200,390,320]:
        page.set_viewport_size({'width':width,'height':900})
        for game in catalogs:
            page.locator('#tab-'+game).click()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert page.locator('#'+game+'-count').is_visible()
        if width==390:page.screenshot(path='/tmp/discovery-progress-mobile.png',full_page=True)
    page.goto((ROOT/'public/index.html').as_uri())
    for game,entries in catalogs.items():
        page.locator('#tab-'+game).click()
        assert page.locator('#'+game+'-count').inner_text()==f'0 of {len(entries)} solutions discovered'
    assert not errors,errors
    browser.close()
print(json.dumps({'status':'passed','totals':{k:len(v) for k,v in catalogs.items()},'checks':['every enumerated solution reachable through the UI','all completion messages','duplicates never advance progress','fresh visit restores discovery goal','extra unary signs rejected','independent exact pantry enumeration','all possible Python output vectors reachable','article and games page','desktop and mobile overflow','no browser errors']},indent=2))
