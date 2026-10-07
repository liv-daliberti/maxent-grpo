'use strict';
/* Small, deterministic verifiers for the paper's Appendix C example games. */
const GameRules=(()=>{
  const gcd=(a,b)=>b?gcd(b,a%b):Math.abs(a);
  function fraction(n,d){if(!d)throw Error('Division by zero is not allowed.');const g=gcd(n,d);return [n/g*Math.sign(d),Math.abs(d)/g];}
  function countdown(input){
    const text=input.replaceAll('×','*').replaceAll('÷','/').trim();
    if(!text||text.length>120||!/^[0-9+*/()\s-]+$/.test(text))throw Error('Use 3, 6 and 9 with +, −, ×, ÷ and parentheses.');
    const tokens=text.match(/\d+|[()+*/-]/g);let at=0;
    function atom(depth){
      if(depth>20)throw Error('Try a shorter expression.');
      const t=tokens[at++];
      if(t==='+'||t==='-')throw Error('Keep the tiles positive. Put + or − between two expressions, not in front of a tile or group.');
      if(t==='('){const v=expr(depth+1);if(tokens[at++]!==')')throw Error('Close every parenthesis.');return v;}
      if(!t||!/^\d+$/.test(t)||!['3','6','9'].includes(t))throw Error('Use each of the tiles 3, 6 and 9 exactly once.');
      return {n:+t,d:1,key:t,used:[+t]};
    }
    function combine(a,op,b){
      let n,d;
      if(op==='+'){n=a.n*b.d+b.n*a.d;d=a.d*b.d;}
      if(op==='-'){n=a.n*b.d-b.n*a.d;d=a.d*b.d;}
      if(op==='*'){n=a.n*b.n;d=a.d*b.d;}
      if(op==='/'){n=a.n*b.d;d=a.d*b.n;}
      [n,d]=fraction(n,d);
      const keys=[a.key,b.key];if(op==='+'||op==='*')keys.sort();
      return {n,d,key:({'+':'add','-':'sub','*':'mul','/':'div'}[op])+`(${keys.join(',')})`,used:[...a.used,...b.used]};
    }
    function term(depth){let v=atom(depth);while(tokens[at]==='*'||tokens[at]==='/'){const op=tokens[at++];v=combine(v,op,atom(depth));}return v;}
    function expr(depth){let v=term(depth);while(tokens[at]==='+'||tokens[at]==='-'){const op=tokens[at++];v=combine(v,op,term(depth));}return v;}
    const v=expr(0);
    if(at!==tokens.length)throw Error('Check the operators and parentheses.');
    if(v.used.sort().join(',')!=='3,6,9')throw Error('Use each tile exactly once.');
    if(v.n!==18*v.d)throw Error(`That equals ${v.d===1?v.n:v.n+'/'+v.d}, not 18. Try again.`);
    return {key:v.key,label:text.replaceAll('*','×').replaceAll('/','÷')+' = 18'};
  }
  const inputs=[18,82,91,93],factorChoices=[1,2,3,6,7,9,13,31,41];
  function python(ds){
    if(ds.length!==4||ds.some(d=>!Number.isInteger(d)||d<1||d>100))throw Error('Choose four integer divisors.');
    const outputs=inputs.map(n=>ds.slice(0,3).find(d=>n%d===0)??ds[3]);
    const valid=outputs.map((d,i)=>d>1&&d<inputs[i]&&inputs[i]%d===0);
    return {key:outputs.join(','),label:'['+outputs.join(', ')+']',outputs,valid,correct:valid.every(Boolean)};
  }
  const actionNames={A:'Divide by 2',B:'Subtract 9',C:'Add 9',D:'Add 1',E:'Add 18',F:'Multiply by 2'};
  function algebra(ids){
    if(ids.length>4)throw Error('Use at most four actions.');
    let s=[.5,-9,-1];const states=[s],seen=new Set([s.join(',')]);
    for(const id of ids){
      let [a,b,c]=s;
      if(id==='A')s=[a/2,b/2,c/2];else if(id==='F')s=[a*2,b*2,c*2];
      else if(['B','C','D','E'].includes(id)){const v={B:-9,C:9,D:1,E:18}[id];s=[a,b+v,c+v];}
      else throw Error('Choose an action from the menu.');
      if(seen.has(s.join(',')))throw Error('That returns to an earlier equation. Repeated states are not allowed.');
      states.push(s);seen.add(s.join(','));
    }
    return {key:ids.join(';'),label:ids.map(id=>actionNames[id]).join(' → '),states,correct:s[0]===1&&s[1]===0};
  }
  const pantry=[
    {name:'Almonds',max:100,n:[584000,21500,10800,0]},
    {name:'Banana',max:125,n:[97000,740,4620,0]},
    {name:'Carrots',max:150,n:[45000,941,3100,86600]},
    {name:'Grape tomatoes',max:100,n:[27000,830,2100,6000]},
    {name:'Navel orange',max:150,n:[47000,910,2000,9000]},
    {name:'Sunflower seeds',max:125,n:[571000,18900,7220,0]}
  ];
  const pantryCache=new Map();
  function checkAmounts(amounts){
    const mass=amounts.reduce((a,b)=>a+b,0),totals=[0,0,0,0];
    amounts.forEach((g,i)=>pantry[i].n.forEach((v,j)=>totals[j]+=g*v));
    const [e,p,f,s]=totals;
    return {correct:mass>=125&&mass<=200&&e>=23940000&&e<=43225000&&p>=642300&&f>=347800&&s<=3760000,mass,totals:totals.map(v=>v/100000)};
  }
  function pantryAttempt(amounts){
    if(!Array.isArray(amounts)||amounts.length!==pantry.length||amounts.some((g,i)=>!Number.isInteger(g)||g<0||g>pantry[i].max||(g!==0&&(g<50||g%25!==0))))
      return {correct:false,reason:'Use 50 g or more of each selected ingredient, in 25 g steps, within the available amount.'};
    const selected=amounts.map((g,i)=>g?i:-1).filter(i=>i>=0);
    if(selected.length<2||selected.length>4)return {correct:false,reason:'Choose between two and four ingredients.'};
    const result=checkAmounts(amounts);
    return {...result,amounts,key:amounts.map(g=>g?'1':'0').join(''),label:selected.map(i=>pantry[i].name+' '+amounts[i]+' g').join(' + '),reason:result.correct?'':'Some totals are outside their targets. Adjust the amounts or try different ingredients.'};
  }
  function plan(selected){
    const key=pantry.map((_,i)=>selected.includes(i)?'1':'0').join('');
    if(pantryCache.has(key))return pantryCache.get(key);
    if(selected.length<2||selected.length>4)return {correct:false,reason:'Choose between two and four ingredients.'};
    let answer=null;
    function search(i,amounts,mass){
      if(answer||mass>200)return;
      if(i===6){const c=checkAmounts(amounts);if(c.correct)answer={...c,amounts,key,label:selected.map(j=>pantry[j].name).join(' + ')};return;}
      const choices=selected.includes(i)?Array.from({length:(pantry[i].max-50)/25+1},(_,j)=>50+j*25):[0];
      for(const g of choices)search(i+1,[...amounts,g],mass+g);
    }
    search(0,[],0);
    const result=answer||{correct:false,reason:'No allowed quantities meet all five bounds for this ingredient set. Try a different mix.'};
    pantryCache.set(key,result);return result;
  }
  let catalogCache;
  function catalogs(){
    if(catalogCache)return catalogCache;
    const result={countdown:new Map(),python:new Map(),algebra:new Map(),pantry:new Map()};
    function expressions(numbers){
      if(numbers.length===1)return [String(numbers[0])];
      const out=[];
      for(let mask=1;mask<(1<<numbers.length)-1;mask++){
        const left=numbers.filter((_,i)=>mask&(1<<i)),right=numbers.filter((_,i)=>!(mask&(1<<i)));
        for(const a of expressions(left))for(const b of expressions(right))for(const op of '+-*/')out.push(`(${a}${op}${b})`);
      }
      return out;
    }
    for(const e of expressions([3,6,9]))try{const r=countdown(e);result.countdown.set(r.key,e);}catch{}
    for(const a of factorChoices)for(const b of factorChoices)for(const c of factorChoices)for(const d of factorChoices){const r=python([a,b,c,d]);if(r.correct)result.python.set(r.key,[a,b,c,d]);}
    function paths(ids){
      if(ids.length)try{const r=algebra(ids);if(r.correct)result.algebra.set(r.key,ids);}catch{return;}
      if(ids.length<4)for(const id of Object.keys(actionNames))paths([...ids,id]);
    }
    paths([]);
    for(let mask=0;mask<64;mask++){
      const selected=pantry.map((_,i)=>i).filter(i=>mask&(1<<i)),r=plan(selected);
      if(r.correct)result.pantry.set(r.key,selected);
    }
    catalogCache=result;return result;
  }
  const solutionTotals=()=>Object.fromEntries(Object.entries(catalogs()).map(([game,solutions])=>[game,solutions.size]));
  return {countdown,python,inputs,factorChoices,algebra,actionNames,pantry,plan,checkAmounts,pantryAttempt,catalogs,solutionTotals};
})();
if(typeof module!=='undefined')module.exports=GameRules;
if(typeof document!=='undefined')(()=>{
  const $=id=>document.getElementById(id);
  if(!$('game-lab'))return;
  const discoveries={countdown:new Map(),python:new Map(),algebra:new Map(),pantry:new Map()};
  const totals=GameRules.solutionTotals();
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function message(game,text,kind='neutral'){$(game+'-message').textContent=text;$(game+'-message').dataset.kind=kind;$(game+'-message').hidden=false;}
  function clearFeedback(game){$(game+'-message').hidden=true;}
  function renderProgress(game){
    const n=discoveries[game].size,total=totals[game],remaining=total-n;
    $(game+'-count').textContent=`${n} of ${total} solutions discovered`;
    $(game+'-progress').max=total;$(game+'-progress').value=n;
    $(game+'-remaining').textContent=remaining?`${remaining} left to find`:'Collection complete';
  }
  Object.keys(discoveries).forEach(renderProgress);
  function accept(game,result){
    const repeat=discoveries[game].has(result.key);
    if(!repeat)discoveries[game].set(result.key,result.label);
    const complete=discoveries[game].size===totals[game];
    message(game,repeat?'Correct, but already discovered. Your progress stays the same.':complete?`All ${totals[game]} solutions discovered! You completed this game.`:'Correct — a new solution!','success');
    renderProgress(game);
  }
  const tabs=[...document.querySelectorAll('[role="tab"][data-game]')];
  function switchGame(id,focus=false){
    for(const tab of tabs){const active=tab.dataset.game===id;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;if(active&&focus)tab.focus();}
    document.querySelectorAll('[data-game-panel]').forEach(p=>p.hidden=p.dataset.gamePanel!==id);
  }
  tabs.forEach((tab,i)=>{
    tab.onclick=()=>switchGame(tab.dataset.game);
    tab.onkeydown=e=>{let next;if(e.key==='ArrowRight')next=(i+1)%tabs.length;if(e.key==='ArrowLeft')next=(i+tabs.length-1)%tabs.length;if(e.key==='Home')next=0;if(e.key==='End')next=tabs.length-1;if(next!==undefined){e.preventDefault();switchGame(tabs[next].dataset.game,true);}};
  });
  if(tabs.some(t=>t.dataset.game===location.hash.slice(1)))switchGame(location.hash.slice(1));
  $('countdown-form').onsubmit=e=>{e.preventDefault();try{accept('countdown',GameRules.countdown($('expression').value));}catch(err){message('countdown',err.message,'error');}};
  document.querySelectorAll('[data-tile]').forEach(b=>b.onclick=()=>{
    const input=$('expression');const at=input.selectionStart,end=input.selectionEnd;
    input.setRangeText(b.dataset.tile,at,end,'end');input.focus();
  });
  $('clear-expression').onclick=()=>{$('expression').value='';$('expression').focus();clearFeedback('countdown');};
  function ruleValues(){return [0,1,2,3].map(i=>+$('divisor-'+i).value);}
  function showRule(){const d=ruleValues();$('python-code').textContent=`lambda n: ${d[0]} if n % ${d[0]} == 0 else ${d[1]} if n % ${d[1]} == 0 else ${d[2]} if n % ${d[2]} == 0 else ${d[3]}`;$('python-results').replaceChildren();clearFeedback('python');}
  [0,1,2,3].forEach(i=>$('divisor-'+i).onchange=showRule);showRule();
  $('run-python').onclick=()=>{
    const r=GameRules.python(ruleValues());
    $('python-results').innerHTML='<table><thead><tr><th>Input n</th><th>Output d</th><th>Proper divisor?</th></tr></thead><tbody>'+r.outputs.map((d,i)=>`<tr><th>${GameRules.inputs[i]}</th><td>${d}</td><td>${r.valid[i]?'✓ Yes':'✕ No'}</td></tr>`).join('')+'</tbody></table>';
    if(r.correct)accept('python',r);else message('python','At least one output fails. Every d must be greater than 1, less than n, and divide n exactly.','error');
  };
  let path=[];
  const format=n=>Number.isInteger(n)?String(n):String(n);
  function equation([a,b,c]){return (a===1?'x':a===.5?'x/2':format(a)+'x')+(b<0?' − '+format(-b):b>0?' + '+format(b):'')+' = '+format(c);}
  function renderAlgebra(){const r=GameRules.algebra(path);$('equation').textContent=equation(r.states.at(-1));$('algebra-path').innerHTML=r.states.map((s,i)=>'<li>'+(i?escape(GameRules.actionNames[path[i-1]])+': ':'Start: ')+escape(equation(s))+'</li>').join('');$('step-count').textContent=path.length+' / 4 steps';$('undo-action').disabled=path.length===0;}
  document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>{try{GameRules.algebra([...path,b.dataset.action]);path.push(b.dataset.action);renderAlgebra();message('algebra','Applied to both sides. Check your path once x is isolated.');}catch(e){message('algebra',e.message,'error');}});
  $('undo-action').onclick=()=>{path.pop();renderAlgebra();message('algebra','Last action undone.');};
  $('restart-algebra').onclick=()=>{path=[];renderAlgebra();clearFeedback('algebra');};
  $('check-algebra').onclick=()=>{const r=GameRules.algebra(path);if(r.correct)accept('algebra',r);else message('algebra','Keep going: finish with exactly x on the left.','error');};renderAlgebra();
  const pantryTargets=[
    {name:'Weight',unit:'g',min:125,max:200,target:'125–200 g'},
    {name:'Energy',unit:'kcal',min:239.4,max:432.25,target:'239.4–432.25 kcal'},
    {name:'Protein',unit:'g',min:6.423,max:Infinity,target:'At least 6.423 g'},
    {name:'Fiber',unit:'g',min:3.478,max:Infinity,target:'At least 3.478 g'},
    {name:'Sodium',unit:'mg',min:0,max:37.6,target:'At most 37.6 mg'}
  ];
  const pantryAmounts=()=>GameRules.pantry.map((_,i)=>document.querySelector('[name="ingredient"][value="'+i+'"]').checked?+$('amount-'+i).value:0);
  function renderPantry(){
    const amounts=pantryAmounts(),n=amounts.filter(Boolean).length,r=GameRules.checkAmounts(amounts),values=[r.mass,...r.totals];
    let met=0;
    $('pantry-selected').textContent=n+' ingredient'+(n===1?'':'s')+' selected · choose 2–4';
    $('pantry-totals').innerHTML=pantryTargets.map((t,i)=>{
      const value=values[i],ok=value>=t.min&&value<=t.max;if(ok)met++;
      const state=value<t.min?'Too low':value>t.max?'Too high':'Met';
      return '<tr data-met="'+ok+'"><th scope="row">'+t.name+'</th><td><strong>'+value.toLocaleString('en-US',{maximumFractionDigits:5})+' '+t.unit+'</strong><span class="target-status">'+(ok?'✓ ':'')+state+'</span></td><td>'+t.target+'</td></tr>';
    }).join('');
    const ready=GameRules.pantryAttempt(amounts).correct;
    $('pantry-ready').textContent=!n?'Choose ingredients to start your plan.':ready?'All 5 targets met. Check your plan to record this combination.':n<2||n>4?'Choose between two and four ingredients.':met+' of 5 targets met. Adjust amounts or change ingredients.';
    $('pantry-ready').dataset.ready=String(ready);
  }
  $('pantry-form').onsubmit=e=>{
    e.preventDefault();const r=GameRules.pantryAttempt(pantryAmounts());
    if(r.correct)accept('pantry',r);else message('pantry',r.reason,'error');
  };
  $('suggest-amounts').onclick=()=>{
    const selected=[...document.querySelectorAll('[name="ingredient"]:checked')].map(el=>+el.value),r=GameRules.plan(selected);
    if(!r.correct){message('pantry',r.reason,'error');return;}
    r.amounts.forEach((g,i)=>{$('amount-'+i).value=String(g);});renderPantry();
    message('pantry','Suggested amounts meet every target. Check your plan to record this combination.');
  };
  document.querySelectorAll('[name="ingredient"]').forEach(el=>el.onchange=()=>{
    const input=$('amount-'+el.value);input.disabled=!el.checked;input.value=el.checked?'50':'0';
    renderPantry();clearFeedback('pantry');
  });
  GameRules.pantry.forEach((_,i)=>$('amount-'+i).onchange=()=>{
    if($('amount-'+i).value==='0'){
      document.querySelector('[name="ingredient"][value="'+i+'"]').checked=false;$('amount-'+i).disabled=true;
    }
    renderPantry();clearFeedback('pantry');
  });
  renderPantry();
})();
