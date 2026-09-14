"""Fixed three-hidden n5 graph laws with balanced label/position streams.

The law is stratified, not uniform over all graph identities: each round visits
all remaining ordered visible-color groups once; each group's independent
round visits its remaining hidden-position groups once. Identities are uniform
without replacement within each joint group. Depleted groups are removed;
there is no structural fallback. Each support has a quota-independent prefix.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from functools import lru_cache
from itertools import combinations,permutations,product
import hashlib,json,random
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
for directory in (HERE,HERE.parent):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
from modebench_level3_discrete import _graph_prompt,graph_completion_count,row_identity
SCHEMA='modebench_level3_graph_candidate_v8'
SUPPORTS=frozenset({4,5,6,8,9,12,18})
PRESETS={0:'n5_three_hidden_minimal_independent_visible_anchors',
         1:'n5_three_hidden_simple_forest_optional_visible_edge',
         2:'n5_three_hidden_minimal_anchors_with_legal_visible_edge',
         3:'n5_three_hidden_any_forest_with_legal_visible_edge'}
AVAILABILITIES={4:(1,2,2),6:(1,2,3),8:(2,2,2),9:(1,3,3),12:(2,2,3),18:(2,3,3)}
SAMPLER_LAW={'primary_groups':'ordered visible color pairs, including three equal-color pairs only where structurally eligible',
 'primary_schedule':'seeded shuffled rounds over all nonempty color groups',
 'secondary_groups':'three hidden vertex positions in ascending order',
 'secondary_schedule':'independent seeded shuffled rounds over nonempty hidden-position groups within each color group',
 'within_joint_group':'uniform random permutation of all eligible identities after fixed exclusions; without replacement',
 'depletion':'remove empty joint/color groups deterministically; never change n, hidden count, or catalogue law',
 'uniform_over_all_remaining_identities':False,'quota_independent_support_prefix':True}

def _seed(*parts):return int.from_bytes(hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).digest(),'big')
def identity(n,edges,partial):return ('graph_coloring',n,tuple(sorted(tuple(sorted(edge)) for edge in edges)),''.join('?' if c is None else str(c) for c in partial))
def group_key(item):
    partial=item[3]
    return tuple(c for c in partial if c!='?'),tuple(i for i,c in enumerate(partial) if c=='?')

@lru_cache(maxsize=28)
def catalogue(difficulty,support):
    """Enumerate the entire fixed semantic identity support, without RNG/outcomes."""
    if difficulty not in PRESETS or support not in SUPPORTS:raise ValueError('expected graph tier0..3 and registered support')
    found=set()
    if difficulty==3:
        hh=list(combinations(range(3),2));hv=list(product(range(3),range(3,5)));templates=[]
        for hmask in range(7):  # Three hidden vertices: every 0..2-edge graph is a forest.
            for vmask in range(64):
                edges=[(3,4)]+[pair for j,pair in enumerate(hh) if hmask>>j&1]+[pair for j,pair in enumerate(hv) if vmask>>j&1]
                if graph_completion_count(5,[(u+1,v+1) for u,v in edges],[None,None,None,1,2])==support:templates.append(edges)
        for hidden in combinations(range(5),3):
            visible=[v for v in range(5) if v not in hidden];mapping=(*hidden,*visible)
            for c1,c2 in permutations((1,2,3),2):
                partial=[None]*5;partial[visible[0]]=c1;partial[visible[1]]=c2
                for edges in templates:found.add(identity(5,[(mapping[u]+1,mapping[v]+1) for u,v in edges],partial))
    else:
        for hidden in permutations(range(5),3):
            visible=[v for v in range(5) if v not in hidden]
            for colors in product((1,2,3),repeat=2):
                partial=[None if v in hidden else colors[visible.index(v)] for v in range(5)];choices=[]
                if support==5:
                    for left in visible:
                        for right in visible:
                            if partial[left]!=partial[right]:choices.append([(hidden[0],hidden[1]),(hidden[1],hidden[2]),(hidden[0],left),(hidden[2],right)])
                elif difficulty==1:
                    forbidden={4:2,6:2,8:1,9:2,12:0,18:0}[support]
                    for anchors in combinations(visible,forbidden):
                        if len({partial[v] for v in anchors})!=forbidden:continue
                        edges=[(hidden[0],v) for v in anchors]
                        if support in (4,6,8,12,18):edges.append((hidden[0],hidden[1]))
                        if support in (4,8,12):edges.append((hidden[1],hidden[2]))
                        choices.append(edges)
                else:
                    neighbors=[[vs for vs in combinations(visible,3-available) if len({partial[v] for v in vs})==3-available] for available in AVAILABILITIES[support]]
                    for anchors in product(*neighbors):choices.append([(v,a) for v,group in zip(hidden,anchors) for a in group])
                visible_edge=tuple(visible) if colors[0]!=colors[1] else None
                for edges in choices:
                    variants=[edges] if difficulty!=2 else []
                    if difficulty in (1,2) and visible_edge:variants.append([*edges,visible_edge])
                    for variant in variants:found.add(identity(5,[(u+1,v+1) for u,v in variant],partial))
    return frozenset(found)

def identity_stream(support,excluded,seed,difficulty):
    available=sorted(catalogue(difficulty,support)-set(excluded));groups=defaultdict(lambda:defaultdict(list))
    for item in available:
        color,hidden=group_key(item);groups[color][hidden].append(item)
    for color,subgroups in groups.items():
        for hidden,items in subgroups.items():random.Random(_seed(SCHEMA,seed,difficulty,support,color,hidden,'identities')).shuffle(items)
    pending={color:[] for color in groups};rounds=Counter();color_round=0
    while groups:
        colors=sorted(groups);random.Random(_seed(SCHEMA,seed,difficulty,support,color_round,'color_round')).shuffle(colors);color_round+=1
        for color in colors:
            if not pending[color]:
                pending[color]=sorted(groups[color]);random.Random(_seed(SCHEMA,seed,difficulty,support,color,rounds[color],'position_round')).shuffle(pending[color]);rounds[color]+=1
            hidden=pending[color].pop();items=groups[color][hidden];item=items.pop()
            if not items:
                del groups[color][hidden]
                if not groups[color]:del groups[color]
            yield item

def graph_stream(support,excluded,seed,difficulty):
    for _,n,edges,text in identity_stream(support,excluded,seed,difficulty):
        yield n,[list(pair) for pair in edges],[None if c=='?' else int(c) for c in text]

def build_pool(domain,target,excluded,seed,tag,difficulty,multiplier=4):
    if domain!='graph_coloring' or type(difficulty) is not int or difficulty not in PRESETS:raise ValueError('graph v8 requires graph_coloring and integer tier0..3')
    if type(multiplier) is not int or multiplier<1 or any(type(count) is not int or count<0 or type(support) is not int or support not in SUPPORTS for support,count in target.items()):raise ValueError('registered integer supports and nonnegative integer quotas required')
    required=Counter({s:c*multiplier for s,c in target.items() if c});rows=[]
    for support,count in sorted(required.items()):
        stream=graph_stream(support,excluded,seed,difficulty)
        for index in range(count):
            try:n,edges,partial=next(stream)
            except StopIteration as error:raise RuntimeError(f'graph v8 tier{difficulty}/support{support} catalogue exhausted; no topology/size fallback') from error
            spec={'verifier':domain,'n':n,'edges':edges,'partial_colors':partial,'source':SCHEMA,'instance_id':f'{tag}-{seed}-support-{support}-{index}','num_completions':support,'num_solutions':graph_completion_count(n,edges,[None]*n)}
            rows.append({'problem':_graph_prompt(n,edges,partial),'answer':json.dumps(spec,sort_keys=True),'modebench_task':domain,'answer_mode_count':support,'answer_mode_split':tag,'level3_difficulty':difficulty,'level3_generator':SCHEMA,'level3_graph_preset':PRESETS[difficulty],'level3_cell_index':index})
    rows.sort(key=lambda row:_seed(SCHEMA,seed,'output_order',row_identity(domain,row)))
    ids={row_identity(domain,row) for row in rows}
    if len(ids)!=len(rows) or ids&set(excluded) or Counter(r['answer_mode_count'] for r in rows)!=required:raise RuntimeError('graph v8 identity/histogram invariant failed')
    return rows
