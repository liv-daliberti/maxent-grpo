import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
const root = resolve('public');
const article = readFileSync(resolve(root, 'index.html'), 'utf8');
if (/<(?:script|iframe)\b|\son[a-z]+\s*=/i.test(article)) {
  throw new Error('The article must use native HTML interactions without JavaScript.');
}
for (const game of ['1. Graph coloring', '2. Countdown', '3. Python factors', '4. Algebra steps', '5. Pantry planning']) {
  if (!article.includes(`<summary>${game}</summary>`)) throw new Error(`Missing HTML puzzle: ${game}`);
}
for (const game of ['graph', 'countdown', 'python', 'algebra', 'pantry']) {
  if (!article.includes(`play-game game-${game}`)) throw new Error(`Missing playable builder: ${game}`);
}
if (!/<details[^>]*open[^>]*><summary>1\. Graph coloring<\/summary>/.test(article)) {
  throw new Error('The graph-coloring game must be open initially.');
}
for (const [, type] of article.matchAll(/<input[^>]*type="([^"]+)"/g)) {
  if (type !== 'checkbox') throw new Error('Unexpected input type');
}
for (const [, type] of article.matchAll(/<button[^>]*type="([^"]+)"/g)) {
  if (type !== 'reset') throw new Error('Only native reset buttons are allowed');
}
if (/<form[^>]*(?:action|method)=/i.test(article)) throw new Error('Puzzle forms must not submit data');
const css = readFileSync(resolve(root, 'drupal-theme.css'), 'utf8');
if (!css.includes(':has(') || !css.includes(':checked')) throw new Error('Missing required game state CSS');
for (const name of ['index.html', 'reading.html', 'games.html', 'puzzle.html', 'sampler.html']) {
  const page = readFileSync(resolve(root, name), 'utf8');
  if (/citp|chatgpt\.site/i.test(page)) throw new Error(`Old branding in ${name}`);
  for (const [, url] of page.matchAll(/(?:src|href)="([^"]+)"/g)) {
    if (/^(?:https?:|data:|#)/.test(url)) continue;
    const path = url.split(/[?#]/)[0];
    if (!existsSync(resolve(dirname(resolve(root, name)), path))) throw new Error(`Missing ${url} in ${name}`);
  }
}
if (!readFileSync(resolve(root, 'paper.pdf')).subarray(0, 5).equals(Buffer.from('%PDF-'))) throw new Error('Missing paper PDF');
console.log('Static blog validated: HTML-only article, five playable HTML/CSS puzzles, local figures and paper, and retained companion pages.');
