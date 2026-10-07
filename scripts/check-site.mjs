import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
const root = resolve('public');
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
console.log('Static blog validated: article, five games, interactive graph, and local paper.');
