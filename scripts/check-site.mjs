import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
const root = resolve('public');
const article = readFileSync(resolve(root, 'index.html'), 'utf8');
if (/<(?:script|iframe)\b|\son[a-z]+\s*=/i.test(article)) {
  throw new Error('The article must use native HTML interactions without JavaScript.');
}
for (const game of ['graph', 'countdown', 'python', 'algebra', 'pantry']) {
  if (!article.includes(`play-game game-${game}`)) throw new Error(`Missing playable builder: ${game}`);
}
const inputs = [...article.matchAll(/<input\b[^>]*>/g)].map(m => m[0]);
const tabs = inputs.filter(tag => tag.includes('type="radio"'));
if (tabs.length !== 5 || tabs.filter(tag => /\bchecked(?:=|\s|>)/.test(tag)).length !== 1 ||
    !tabs.some(tag => tag.includes('id="mx-game-graph"') && /\bchecked(?:=|\s|>)/.test(tag))) {
  throw new Error('Five native game tabs must start on Graph coloring');
}
for (const tag of inputs) {
  if (!tag.includes('type="checkbox"') && !(tag.includes('type="radio"') && tag.includes('name="mx-game"'))) {
    throw new Error('Unexpected input type');
  }
}
if (article.includes('Compare the exact average discoveries')) throw new Error('Removed comparison dropdown returned');
for (const name of ['02-same-accuracy', '04-measured-concentration', '03-feedback']) {
  if (!article.includes(`src="figures/${name}.png"`)) throw new Error(`Missing original figure: ${name}`);
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
