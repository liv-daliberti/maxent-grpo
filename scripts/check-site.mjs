import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
const root = resolve('public');
const article = readFileSync(resolve(root, 'index.html'), 'utf8');
if (/<(?:script|iframe|form|input|button|select)\b|\son[a-z]+\s*=/i.test(article)) {
  throw new Error('The article must use native HTML interactions without JavaScript or forms.');
}
for (const game of ['1. Graph coloring', '2. Countdown', '3. Python factors', '4. Algebra steps', '5. Pantry planning']) {
  if (!article.includes(`<summary>${game}</summary>`)) throw new Error(`Missing HTML puzzle: ${game}`);
}
for (const graphic of ['game-graph-coloring', 'game-countdown', 'game-python', 'game-algebra', 'game-pantry', 'graph-six-colorings']) {
  if (!article.includes(`figures/${graphic}.png`)) throw new Error(`Missing puzzle graphic: ${graphic}`);
}
if (!/<details open(?:="")?><summary>1\. Graph coloring<\/summary>/.test(article)) {
  throw new Error('The graph-coloring diagram must be visible when the page opens.');
}
for (const name of ['index.html', 'games.html', 'puzzle.html', 'sampler.html']) {
  const page = readFileSync(resolve(root, name), 'utf8');
  if (/citp|chatgpt\.site/i.test(page)) throw new Error(`Old branding in ${name}`);
  for (const [, url] of page.matchAll(/(?:src|href)="([^"]+)"/g)) {
    if (/^(?:https?:|data:|#)/.test(url)) continue;
    const path = url.split(/[?#]/)[0];
    if (!existsSync(resolve(dirname(resolve(root, name)), path))) throw new Error(`Missing ${url} in ${name}`);
  }
}
if (!readFileSync(resolve(root, 'paper.pdf')).subarray(0, 5).equals(Buffer.from('%PDF-'))) throw new Error('Missing paper PDF');
console.log('Static blog validated: HTML-only article, five native puzzle reveals, local figures and paper, and retained companion pages.');
