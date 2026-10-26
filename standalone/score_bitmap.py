"""Paired large-library bitmap experiment; reuse verified NPU tree and both frozen CPUs."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import large_search as retained,score_bitmap_sources as source
g=retained.g;r=retained.r;simd=retained.simd;digest=retained.corpus.digest
OWN=('standalone/score_bitmap.py','standalone/score_bitmap_sources.py')
PASS='actual_score_bitmap_large_candidate_retained'
MISS='actual_score_bitmap_large_correct_but_not_retained'
MIN_REDUCTION=.05
ATOMIC_DOC='https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMT-API/atomic_operations/asc_atomic_or.md'


def decision(rows):
    r.require(len(rows)==6,'Six paired rounds required')
    cpu=retained.gate([{**row,'npu':row['candidate']} for row in rows]);cases={}
    for case in ('A','B'):
        old=[row['baseline']['summary'][case]['median_batch_ms'] for row in rows]
        new=[row['candidate']['summary'][case]['median_batch_ms'] for row in rows]
        old95=[row['baseline']['summary'][case]['p95_batch_ms'] for row in rows]
        new95=[row['candidate']['summary'][case]['p95_batch_ms'] for row in rows]
        reduction=1-statistics.median(new)/statistics.median(old);wins=sum(a<b for a,b in zip(new,old))
        cases[case]={'baseline_medians_ms':old,'candidate_medians_ms':new,'baseline_p95_ms':old95,'candidate_p95_ms':new95,
            'wins':wins,'median_reduction_fraction':reduction,'passed':wins==6 and reduction>=MIN_REDUCTION and statistics.median(new95)<statistics.median(old95)}
    return {'retain_candidate':cpu['performance_target_met_at_this_scale'] and all(c['passed'] for c in cases.values()),
        'minimum_predeclared_median_reduction':MIN_REDUCTION,'versus_verified_npu':cases,'versus_frozen_cpus':cpu}


def actual_atomic_headers(toolkit):
    paths={p.resolve() for p in toolkit.rglob('device_atomic_functions.h') if p.is_file()}
    r.require(len(paths)==1,'Unique actual SIMT atomic header required')
    path=next(iter(paths))
    families={pattern:{p.resolve() for p in toolkit.rglob(pattern) if p.is_file()}
        for pattern in ('device_atomic_functions*.h','device_functions*.h')}
    def declarations_in(pattern,name):
        return [{'path':str(p),'line':i,'text':line.strip()} for p in sorted(families[pattern])
            for i,line in enumerate(p.read_text(encoding='utf-8',errors='replace').splitlines(),1) if re.search(r'\b'+name+r'\s*\(',line)]
    declarations=declarations_in('device_atomic_functions*.h','asc_atomic_or')
    r.require(declarations,'Actual asc_atomic_or declarations missing; stop before build')
    popc=path.parent/'device_functions.h';r.require(popc.is_file(),'Actual integer-math header missing')
    popc_declarations=declarations_in('device_functions*.h','__popc')
    r.require(popc_declarations,'Actual __popc declarations missing; stop before build')
    return {'path':str(path),'sha256':digest(path),'declarations':declarations,'official_reference':ATOMIC_DOC,
        'popc_header':{'path':str(popc),'sha256':digest(popc),'declarations':popc_declarations},
        'files':{str(p):digest(p) for group in families.values() for p in sorted(group)},
        'scope':'950 SIMT uint32 UB atomic OR; target compiler must verify address-space overload. Not SIMD scalar GM-only OR.'}


def build_install(a):
    project=a.output/'project';vendor=a.vendors/source.VENDOR
    r.require(not vendor.exists(),'Independent candidate vendor exists; preserve it')
    metadata=source.prepare(a.original,project);logs=a.output/'compiler_logs';logs.mkdir()
    env=os.environ.copy();env['ASCENDC_BUILD_LOG_DIR']=str(logs)
    g.execute(['bash','build.sh'],a.output/'build.log',env,project,timeout=1800)
    old=simd.t.resource.cache(a.original/'build_out/CMakeCache.txt');new=simd.t.resource.cache(project/'build_out/CMakeCache.txt')
    r.require(old and new.get('vendor_name')==source.VENDOR and {k:v for k,v in old.items() if k!='vendor_name'}==
        {k:v for k,v in new.items() if k!='vendor_name'},'Actual950 target/build settings changed')
    packages=list((project/'build_out').glob('*.run'));r.require(len(packages)==1,'Unique isolated package required')
    listing=subprocess.run(['bash',str(packages[0]),'--list'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (a.output/'package_list.log').write_bytes(listing.stdout)
    r.require(listing.returncode==0 and set(re.findall(r'vendors/([A-Za-z0-9_]+)/',listing.stdout.decode(errors='replace')))=={source.VENDOR},'Not an isolated package')
    g.execute(['bash',packages[0],'--install-path='+str(a.vendors.parent)],a.output/'install.log',timeout=300)
    header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    for name,args in simd.fused.PROTOTYPES.items():
        found=re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',text,flags=re.S)
        r.require(len(found)==1 and re.sub(r'\s+','',found[0])==re.sub(r'\s+','',args),'Actual candidate ABI differs: '+name)
    for name,raw in source.candidate_sources(a.original).items():r.require((project/name).read_bytes()==raw,'Candidate source changed')
    objects=simd.fused.chain.artifacts(vendor,project,'bm25_fused_score_top_k',header.name)
    compiler_logs,resources=simd.compiler_resources(a.output)
    return {'vendor':str(vendor),'project':str(project),'metadata':metadata,'device_objects':objects,
        'compiler_logs':compiler_logs,'simt_resources':resources,'missing_resource_fields':'UNKNOWN; no occupancy or spill inference'}


def baseline_evidence(a):
    path=a.completed/'report.json';old=json.loads(path.read_text(encoding='utf-8'))
    r.require(old.get('status')=='actual_large_search_target_met_on_declared_data','Completed large-scale hardware baseline required')
    r.require(old['source_sha256']==g.checks.sources(retained.OWN+retained.HOST),'Verified large implementation changed')
    proofs={str(path):digest(path)}
    for key in ('previous_sha256','installation_sha256'):
        simd.close.scale.unchanged(old[key],'Verified baseline artifacts changed');proofs.update(old[key])
    wanted={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in retained.source.fused_sources(a.original).items()}
    r.require(old['operators']['fused']['metadata']['source_sha256']==wanted,'Baseline leaf differs from accepted source')
    selected=[t for t in old['tiers'] if t['docs']==a.docs and t['status']=='six_rounds_checked']
    r.require(len(selected)==1,'Requested database was not measured in accepted run');tier=selected[0]
    r.require(len(tier['per_round'])==6,'Accepted scale lacks six complete rounds');prior={}
    for row in tier['per_round']:
        for role in ('npu','cpu','cpu_directory'):
            link=row[role];p=Path(link['directory'])/'report.json'
            r.require(digest(p)==link['report_sha256'],'Accepted per-round report changed');proofs[str(p)]=digest(p)
            data=json.loads(p.read_text(encoding='utf-8'))
            r.require(data['summary']==link['summary'] and digest(Path(data['command'][0]))==data['runner_sha256'],'Accepted runner/summary changed')
            proofs[str(Path(data['command'][0]).resolve())]=data['runner_sha256']
            if role not in prior:prior[role]=data
    r.require(retained.gate(tier['per_round'])==tier['decision'],'Accepted CPU gate does not recompute')
    command=prior['npu']['command'];inputs=Path(command[1]).parent
    r.require(len(command)==14 and command[5]==str(old['device']) and command[7:12]==['10','8','24','5','30'],'Baseline command contract differs')
    description,index=retained.corpus.validate(inputs)
    r.require(description==tier['workload'] and description['docs']==a.docs and description['queries']==8 and description['query_length']==3 and description['k']==10,'Accepted workload differs')
    r.require({case:description['cases'][case]['terms'] for case in ('A','B')}==old['fixed_query_cases'],'Fixed A/B IDs differ')
    proofs[str(inputs/'preparation_report.json')]=digest(inputs/'preparation_report.json')
    return old,tier,prior,inputs,description,index,proofs


def command_for(previous,destination,vendors=None):
    command=list(previous['command']);command[4]=str(destination)
    if vendors is not None:command[6]=str(vendors)
    return command


def adversarial_documents():
    # Different posting threads set32 different bits of owner0; include tail8192.
    for doc in range(8193):
        row=[4]
        if doc%256==0:row += [0,0]
        if doc<1025:row.append(1)
        if doc<1024:row.append(2)
        if doc<1023:row.append(3)
        yield row


def run(a):
    r.require(sys.platform=='linux' and not a.output.exists() and a.output.is_relative_to(ROOT/'results'),'A5 Linux, fresh results output required')
    r.require(all(not a.output.is_relative_to(p) and not p.is_relative_to(a.output) for p in (a.completed,a.original,a.vendors)),'Independent output required')
    old,tier,prior,inputs,description,index,proofs=baseline_evidence(a)
    runner=Path(prior['npu']['command'][0]);baseline_vendors=Path(prior['npu']['command'][6]);device=old['device']
    disk_needed=24*4*a.docs*8*4+1024**3
    r.require(disk_needed<shutil.disk_usage(a.output.parent).free,'Disk budget:24 processes x four immutable full-score checkpoints')
    plan=retained.layout(a.docs,index['vocab_size'],index['posting_count'])
    r.require(plan['cpu_peak_known_bytes']+512*1024**2<retained.host_available_bytes()*4//5,'Current host memory budget insufficient')
    own=g.checks.sources(OWN+retained.OWN+retained.HOST);a.output.mkdir(parents=True)
    report={'status':'running','scope':'Only score-time bitmap changes in accepted large leaf. Reuse CPU/tree/host/data.',
        'source_sha256':own,'previous_sha256':proofs,'device':device,'docs':a.docs,'workload':description,
        'memory_plan':plan,'disk_budget_bytes':disk_needed,'per_round':[],
        'predeclared_retention_rule':'A/B:6/6 faster than paired NPU,>=5% median reduction, lower median p95; pass both frozen CPUs six-round gate.',
        'baseline_large_report':str(a.completed/'report.json')}
    def save():r.write_json(a.output/'report.json',report)
    def check():
        simd.close.scale.unchanged(proofs,'Verified baseline changed')
        r.require(g.checks.sources(OWN+retained.OWN+retained.HOST)==own,'Source changed during run')
        if 'candidate_sha256' in report:simd.close.scale.unchanged(report['candidate_sha256'],'Candidate artifacts changed')
    save()
    try:
        atomic=actual_atomic_headers(a.toolkit);report['actual_atomic_header']=atomic;proofs[atomic['path']]=atomic['sha256']
        proofs.update(atomic['files']);save()
        cpuinfo=Path('/proc/cpuinfo').read_bytes();(a.output/'cpu_info.txt').write_bytes(cpuinfo)
        report['host_context']={'cpu_info_sha256':digest(a.output/'cpu_info.txt'),'allowed_cpu_affinity':sorted(os.sched_getaffinity(0)),
            'openmp_environment':{k:v for k,v in os.environ.items() if k.startswith(('OMP_','GOMP_'))}}
        report['management_before']=retained.accepted.prior.management_snapshot(a.output,'bitmap_before');save()
        g.execute([runner,'--memory',device,a.output/'device_memory.json'],a.output/'memory.log',g.environment(baseline_vendors))
        memory=json.loads((a.output/'device_memory.json').read_text(encoding='utf-8'));report['device_memory_probe']=memory;save()
        r.require(plan['known_device_bytes_upper_bound']+plan['debug_score_bytes']+1024**3<=memory['free']*4//5,'Actual device memory budget insufficient')
        report['candidate']=build_install(a);save();check()
        vendors=g.aliases(a.output/'runtime/vendors',a.vendors/source.VENDOR,baseline_vendors/'bm25_topk_merge')
        project=a.output/'project';files=[p for p in (a.vendors/source.VENDOR).rglob('*') if p.is_file() and p.suffix in ('.so','.o','.json','.h')]
        files += [project/name for name in source.candidate_sources(a.original)]
        files += [project/name for name in ('bm25_project.json','CMakePresets.json','op_kernel/CMakeLists.txt','build.sh')]
        report['candidate_sha256']={str(p.resolve()):digest(p) for p in files};save();check()
        env=g.environment(vendors);selector=ROOT/'results/fused_simd_close2/selector_host_build/bm25_fused_simd_selector_check'
        r.require(str(selector.resolve()) in proofs,'Retained54-call raw checker not fingerprinted')
        g.execute([selector,vendors,device,a.output/'selector_checks'],a.output/'selector.log',env)
        raw=json.loads((a.output/'selector_checks/report.json').read_text(encoding='utf-8'))
        r.require(raw['status']=='actual_simd_selector_guards_and_strict_bit_rankings_passed' and raw['calls']==54,'Raw score/zero/guard checks failed')
        report['affected_checks']={'raw_selector':raw};save();check()
        fixture=a.output/'adversarial_input';original_docs=a.output/'adversarial_tokens.jsonl'
        with original_docs.open('w',encoding='utf-8',newline='\n') as f:
            for row in adversarial_documents():f.write(json.dumps(row)+'\n')
        cases={'A':[[0,0,7],[1,7,7],[2,7,7],[3,7,7]]*2,'B':[[7,7,7],[0,2,0],[2,3,2],[1,3,1]]*2}
        retained.corpus.prepare(fixture,8193,8,950,cases,corpus=original_docs)
        desc,idx=retained.corpus.validate(fixture)
        for label,vendor in (('baseline',baseline_vendors),('candidate',vendors)):
            destination=a.output/('affected_'+label)
            command=[runner,fixture/'index',fixture/'A.txt',fixture/'B.txt',destination,device,vendor,10,8,24,1,10,
                fixture/'A/expected_scores.bin',fixture/'B/expected_scores.bin']
            check();g.execute(command,a.output/('affected_'+label+'.log'),g.environment(vendor))
            report['affected_checks'][label]=retained.verify_run(destination,fixture,desc,idx,'npu',device,1,10);save();check()
        r.require(report['affected_checks']['baseline']['output_sha256']==report['affected_checks']['candidate']['output_sha256'],'Affected full output bits differ')
        def measure(role,path):
            path.mkdir(parents=True);out=path/'actual';npu=role in ('baseline','candidate')
            previous=prior['npu'] if npu else prior[role]
            vendor=(vendors if role=='candidate' else baseline_vendors) if npu else None
            command=command_for(previous,out,vendor);check()
            g.execute(command,path/'run.log',g.environment(vendor) if npu else os.environ.copy(),timeout=3600)
            result=retained.verify_run(out,inputs,description,index,'npu' if npu else role,device);check()
            result.update(command=command,runner_sha256=digest(Path(command[0])),log_sha256=digest(path/'run.log'))
            r.write_json(path/'report.json',result)
            print(f'N={a.docs} {role} complete Search: '+json.dumps(result['summary'],ensure_ascii=False),flush=True)
            return {'directory':str(path),'summary':result['summary'],'report_sha256':digest(path/'report.json')}
        for i in range(6):
            retained.corpus.validate(inputs)
            order=('cpu','baseline','candidate','cpu_directory') if i%2==0 else ('cpu_directory','candidate','baseline','cpu')
            row={'round':i+1,'process_order':list(order)}
            for role in order:row[role]=measure(role,a.output/f'round{i+1}'/role)
            for checkpoint in g.checks.CHECKPOINTS:
                left=Path(row['baseline']['directory'])/'actual'/checkpoint;right=Path(row['candidate']['directory'])/'actual'/checkpoint
                for file in left.iterdir():
                    if file.suffix=='.bin':r.require(retained.same_stream(file,right/file.name),'Paired full-score/tree bits differ')
            report['per_round'].append(row);save()
        retained.corpus.validate(inputs);report['decision']=decision(report['per_round'])
        report['management_after']=retained.accepted.prior.management_snapshot(a.output,'bitmap_after')
        report['status']=PASS if report['decision']['retain_candidate'] else MISS;check();save()
        print('Paired large-scale bitmap decision: '+json.dumps(report['decision'],ensure_ascii=False),flush=True)
        print('Verified large baseline remains installed; retain candidate only if all declared gates pass.',flush=True)
        return 0 if report['status']==PASS else 2
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));save()
        logs=list(a.output.rglob('*.log'));recent=max(logs,key=lambda p:p.stat().st_mtime) if logs else None
        if recent:print('Latest stage log '+str(recent)+'\n'+recent.read_text(encoding='utf-8',errors='replace')[-24000:],flush=True)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,path in {'completed':'results/large_scale_close1','original':'results/Bm25FusedScoreTopKUbFix1','vendors':'results/custom_opp/vendors'}.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    p.add_argument('--toolkit',type=Path,default=Path('/usr/local/Ascend/cann-9.2.0'))
    p.add_argument('--docs',type=int,default=10000000);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    try:
        r.require(a.docs>=1000000,'Large article count required, not increased batch');return run(a)
    except (Exception,KeyboardInterrupt) as error:print('FAILED: '+str(error)+'; baseline retained, no fabricated device result.',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
