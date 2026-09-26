#!/usr/bin/env python3
"""External, durable pacing for one authorized Hub repository's commit requests.

Model hashing, upload construction, byte verification and retirement stay in the
unchanged archive engine. LFS transfers remain parallel; only final commit POSTs
share this gate. No uncertain or unclassified request is automatically retried.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl,hashlib,json,math,os,re,stat,threading,time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

REPO='od2961/maxent-grpo-models'
COMMIT_PATH='/api/models/'+REPO+'/commit/main'
SPACING_SECONDS=20
COOLDOWN_SECONDS=3605
SCHEMA='model-archive-repository-commit-pacing-v1'
_TLS=threading.local()

class CommitPacingUnsafe(Exception):pass
class RepositoryCommitRateLimited(Exception):pass
class RepositoryCommitCooldown(Exception):pass

def require(ok,message):
    if not ok:raise CommitPacingUnsafe(message)
def utc(epoch=None):return datetime.fromtimestamp(time.time() if epoch is None else epoch,timezone.utc).isoformat()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def granular_commit_limit(status,body):
    if status!=429:return False
    try:error=json.loads(body).get('error','')
    except (ValueError,AttributeError):return False
    return isinstance(error,str) and 'rate limit for repository commits (256 per hour)' in error and 'about 1 hour' in error

def sanitize(body,secret):
    if secret:body=body.replace(secret,'[credential omitted]')
    body=re.sub(r'https?://\S+','[URL omitted]',body)
    return re.sub(r'\b(?:hf_|sk-)[A-Za-z0-9_\-]+','[credential omitted]',body)[:1000]

def safe_headers(headers,secret):
    result={}
    for key,value in headers.items():
        if str(key).lower() not in ('date','ratelimit','ratelimit-policy','retry-after'):continue
        value=str(value)
        if len(value)<=512 and re.fullmatch(r'[A-Za-z0-9 ,;=._"()\-:+]*',value) and not (secret and secret in value):
            result[str(key)]=sanitize(value,secret)
    return result


def validate_state(state):
    require(state.get('schema')==SCHEMA and state.get('repo_id')==REPO
        and state.get('minimum_spacing_seconds')==SPACING_SECONDS
        and state.get('granular_cooldown_seconds')==COOLDOWN_SECONDS,'pacing_policy_changed')
    for key in ('last_attempt_at','cooldown_until'):
        value=state.get(key);require(type(value) in (int,float) and math.isfinite(value) and value>=0,'invalid_pacing_time')
    require(type(state.get('next_sequence')) is int and state['next_sequence']>=0,'invalid_pacing_sequence')
    require(state.get('blocked_reason') in (None,'uncertain_request','unclassified_response','unparseable_success'),'invalid_pacing_block')
    require(state.get('pending_attempt') is None or type(state['pending_attempt']) is int,'invalid_pending_attempt')
    return state

def admission_delay(state,current):
    validate_state(state)
    require(state['blocked_reason'] is None and state['pending_attempt'] is None,'unresolved_prior_commit_attempt')
    if state['cooldown_until']>current:raise RepositoryCommitCooldown('repository_commit_cooldown_active')
    return max(0,state['last_attempt_at']+SPACING_SECONDS-current)

def read(path):
    path=Path(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno());require(stat.S_ISREG(info.st_mode) and info.st_size<=65536,'unsafe_pacing_metadata')
        return json.loads(stream.read())

def save(path,value,*,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path if immutable else path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    if not immutable:os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

class CommitPacer:
    def __init__(self,directory,*,clock=time.time,sleep=time.sleep):
        self.directory=Path(directory).absolute();self.clock=clock;self.sleep=sleep
        require(self.directory.resolve()==self.directory and not self.directory.is_symlink(),'unsafe_pacing_directory')
        self.directory.mkdir(parents=True,exist_ok=True)
    @contextmanager
    def locked(self):
        fd=os.open(self.directory/'gate.lock',os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            yield
    def bootstrap(self,*,initial_cooldown_until,evidence_pin):
        require(type(initial_cooldown_until) in (int,float) and math.isfinite(initial_cooldown_until),'invalid_initial_cooldown')
        require(set(evidence_pin)=={'path','sha256'} and digest(evidence_pin['path'])==evidence_pin['sha256'],'cooldown_evidence_changed')
        registration={'schema':SCHEMA,'repo_id':REPO,'minimum_spacing_seconds':SPACING_SECONDS,
            'granular_cooldown_seconds':COOLDOWN_SECONDS,'source_sha256':digest(__file__),
            'initial_cooldown_until':initial_cooldown_until,'initial_evidence':evidence_pin}
        with self.locked():
            path=self.directory/'registration.json';state_path=self.directory/'state.json'
            if path.exists():
                require(read(path)==registration,'pacing_registration_changed')
                return validate_state(read(state_path))
            require(not state_path.exists() and not list((self.directory/'attempts').glob('*')),'unregistered_pacing_state')
            save(path,registration,immutable=True)
            state={'schema':SCHEMA,'repo_id':REPO,'minimum_spacing_seconds':SPACING_SECONDS,
                'granular_cooldown_seconds':COOLDOWN_SECONDS,'last_attempt_at':0,'cooldown_until':initial_cooldown_until,
                'next_sequence':0,'pending_attempt':None,'blocked_reason':None}
            save(state_path,state);return state
    def snapshot(self):
        with self.locked():
            registration=read(self.directory/'registration.json')
            require(registration['source_sha256']==digest(__file__),'pacing_source_changed')
            return validate_state(read(self.directory/'state.json'))
    def transact(self,summary,sender,secret):
        require(isinstance(summary,str) and 0<len(summary)<=512 and '\n' not in summary,'missing_exact_commit_context')
        with self.locked():
            registration=read(self.directory/'registration.json')
            require(registration['source_sha256']==digest(__file__),'pacing_source_changed')
            state=validate_state(read(self.directory/'state.json'))
            while True:
                try:delay=admission_delay(state,self.clock())
                except RepositoryCommitCooldown:
                    refusal={'schema':'paced-commit-not-sent-v1','status':'deferred_before_http','summary':summary,
                        'repo_id':REPO,'at_utc':utc(self.clock()),'cooldown_until':state['cooldown_until'],
                        'last_sequence':state['next_sequence']-1,'pid':os.getpid()}
                    save(self.directory/'not_sent'/(uuid4().hex+'.json'),refusal,immutable=True)
                    raise
                if not delay:break
                self.sleep(min(delay,30))
            sequence=state['next_sequence'];stamp=self.clock()
            prefix=self.directory/'attempts'/f'{sequence:06d}'
            intent={'schema':'paced-repository-commit-attempt-v1','sequence':sequence,'repo_id':REPO,
                'method':'POST','host':'huggingface.co','path':COMMIT_PATH,'summary':summary,
                'attempted_at':stamp,'attempted_at_utc':utc(stamp),'pid':os.getpid()}
            save(prefix.with_suffix('.intent.json'),intent,immutable=True)
            state.update(last_attempt_at=stamp,next_sequence=sequence+1,pending_attempt=sequence)
            save(self.directory/'state.json',state)
            try:response=sender()
            except BaseException as error:
                result={'status':'uncertain','sequence':sequence,'summary':summary,'at_utc':utc(self.clock()),'error_type':type(error).__name__}
                save(prefix.with_suffix('.result.json'),result,immutable=True)
                state.update(blocked_reason='uncertain_request');save(self.directory/'state.json',state)
                raise
            status=response.status_code
            result={'sequence':sequence,'summary':summary,'http_status':status,'at_utc':utc(self.clock()),
                'headers':safe_headers(response.headers,secret)}
            if status==200:
                try:commit=response.json().get('commitOid')
                except (ValueError,AttributeError,TypeError):commit=None
                if re.fullmatch('[0-9a-f]{40}',commit or ''):
                    result.update(status='committed',commit_sha=commit);state.update(pending_attempt=None)
                else:
                    result.update(status='unparseable_success');state.update(blocked_reason='unparseable_success')
            elif granular_commit_limit(status,response.text):
                until=self.clock()+COOLDOWN_SECONDS
                result.update(status='definitively_rejected_granular_limit',sanitized_error=sanitize(response.text,secret),
                    cooldown_until=until,cooldown_until_utc=utc(until))
                state.update(cooldown_until=until,pending_attempt=None)
            else:
                result.update(status='unclassified_response',sanitized_error=sanitize(response.text,secret))
                state.update(blocked_reason='unclassified_response')
            save(prefix.with_suffix('.result.json'),result,immutable=True)
            save(self.directory/'state.json',state)
            if result['status']=='definitively_rejected_granular_limit':
                raise RepositoryCommitRateLimited('repository_commit_limit_256_per_hour')
            require(result['status']=='committed','unclassified_commit_response_requires_reconciliation')
            return response

def install_pacing(pacer,secret):
    """Install a process-local wrapper; each original SDK request is sent at most once."""
    import requests
    from huggingface_hub import HfApi,configure_http_backend
    require(bool(secret),'missing_private_credential')
    current=HfApi.create_commit
    if getattr(current,'_maxent_pacer_directory',None) is not None:
        require(current._maxent_pacer_directory==str(pacer.directory),'different_pacer_already_installed');return
    original=current
    def create_commit(self,*args,**kwargs):
        require(not args and kwargs.get('repo_id')==REPO and kwargs.get('repo_type')=='model','unmapped_commit_target')
        require(kwargs.get('revision') in (None,'main') and not kwargs.get('create_pr') and not kwargs.get('run_as_future'),'unmapped_commit_mode')
        summary=kwargs.get('commit_message','')
        require(isinstance(summary,str) and (summary.startswith('Archive ') or re.fullmatch(r'Index [0-9]+ verified experimental models',summary)
            or summary=='Publish final715-model catalog and complete scientific archive guides'),'unmapped_commit_purpose')
        prior=getattr(_TLS,'summary',None);_TLS.summary=summary
        try:return original(self,*args,**kwargs)
        finally:_TLS.summary=prior
    create_commit._maxent_pacer_directory=str(pacer.directory)
    HfApi.create_commit=create_commit
    class PacedSession(requests.Session):
        def request(self,method,url,*args,**kwargs):
            parsed=urlsplit(url)
            if method.upper()=='POST' and parsed.scheme=='https' and parsed.netloc=='huggingface.co' and parsed.path==COMMIT_PATH:
                require(not parsed.query and not kwargs.get('params'),'unexpected_commit_query')
                summary=getattr(_TLS,'summary',None)
                kwargs['allow_redirects']=False
                return pacer.transact(summary,lambda:super(PacedSession,self).request(method,url,*args,**kwargs),secret)
            return super().request(method,url,*args,**kwargs)
    configure_http_backend(backend_factory=PacedSession)
